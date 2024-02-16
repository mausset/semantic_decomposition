import lightning as pl
import torch
from torch import nn
from einops import rearrange

from x_transformers import Encoder


class SA(pl.LightningModule):

    def __init__(
        self, input_dim, slot_dim, n_slots=7, n_iters=3, implicit=False, eps=1e-8
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.implicit = implicit
        self.eps = eps

        self.scale = input_dim**-0.5

        self.mu = nn.Parameter(torch.randn(slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.slots_logsigma = nn.Parameter(self.slots_logsigma.squeeze())

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 2, input_dim),
        )

        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.norm_recurrent = nn.LayerNorm(slot_dim)

    def step(self, slots, k, v):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps

        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjd,bij->bid", v, attn)
        slots = rearrange(slots, "b n d -> (b n) d")
        updates = rearrange(updates, "b n d -> (b n) d")
        slots = self.gru(updates, slots)
        slots = rearrange(slots, "(b n) d -> b n d", n=n)

        slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

    def forward(self, x, sample):
        b, t, _, _ = x.shape

        if sample is None:
            sample = torch.randn((b, t, self.n_slots, self.slot_dim), device=x.device)
        slots = self.mu + self.slots_logsigma.exp() * sample

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        k = rearrange(k, "b t n d -> (b t) n d")
        v = rearrange(v, "b t n d -> (b t) n d")
        slots = rearrange(slots, "b t n d -> (b t) n d")

        for _ in range(self.n_iters):
            slots = self.step(slots, k, v)

        if self.implicit:
            slots = self.step(slots.detach(), k, v)

        slots = rearrange(slots, "(b t) n d -> b t n d", t=t)

        return slots, sample

    def recurrent(self, x, init_slots):
        b, t, _, _ = x.shape

        if init_slots is None:
            sample = torch.randn((b, self.n_slots, self.slot_dim), device=x.device)
            init_slots = self.mu + self.slots_logsigma.exp() * sample
        slots = init_slots

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        results = []
        for i in range(t):
            for _ in range(self.n_iters):
                slots = self.step(slots, k[:, i], v[:, i])
            if self.implicit:
                slots = self.step(slots.detach(), k[:, i], v[:, i])
            slots = self.norm_recurrent(slots)
            results.append(slots)
            slots = slots.detach()

        slots = torch.stack(results, dim=1)

        return slots, None


class ICASALayer(pl.LightningModule):
    "Transformer layer with inverted cross attention."

    def __init__(self, input_dim, slot_dim, eps=1e-8):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.eps = eps

        self.scale = input_dim**-0.5

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.slot_norm = nn.LayerNorm(slot_dim)

        self.encoder_transformer = Encoder(
            dim=slot_dim, depth=1, ff_glu=True, ff_dropout=0.0
        )

    def forward(self, x, slots):

        q = self.inv_cross_q(self.slot_norm(slots))
        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps

        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjd,bij->bid", v, attn)

        slots = self.slot_norm(slots + updates)
        slots = self.encoder_transformer(updates)

        return slots


class ICASA(pl.LightningModule):

    def __init__(self, input_dim, slot_dim, depth, n_slots=5, eps=1e-8):
        super().__init__()

        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.depth = depth
        self.n_slots = n_slots
        self.eps = eps

        self.mu = nn.Parameter(torch.randn(slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.slots_logsigma = nn.Parameter(self.slots_logsigma.squeeze())

        self.layers = nn.ModuleList(
            [ICASALayer(input_dim, slot_dim, eps) for _ in range(depth)]
        )

    def forward(self, x, sample):

        if sample is None:
            sample = torch.randn(
                (x.shape[0], self.n_slots, self.slot_dim), device=x.device
            )

        slots = self.mu + self.slots_logsigma.exp() * sample
        results = []
        for i in range(x.shape[1]):
            for layer in self.layers:
                slots = layer(x[:, i], slots)
            results.append(slots)

            slots = slots.detach()

        slots = torch.stack(results, dim=1)
        return slots, sample

    def non_recurrent(self, x, slots):

        for layer in self.layers:
            slots = layer(x, slots)

        return slots
