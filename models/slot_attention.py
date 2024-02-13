import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


class ISA(pl.LightningModule):

    def __init__(
        self, input_dim, slot_dim, n_iters, n_slots=5, min_slots=None, eps=1e-8
    ):
        super().__init__()

        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.min_slots = min_slots
        self.eps = eps

        if self.min_slots is not None:
            assert self.min_slots <= self.n_slots

        self.mu = nn.Parameter(torch.randn(slot_dim))
        self.sigma = nn.Parameter(torch.randn(slot_dim))

        self.in_to_kv = nn.Linear(input_dim, slot_dim * 2, bias=False)
        self.slot_to_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.ffn = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, input_dim),
        )

        self.input_norm = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_ffn = nn.LayerNorm(slot_dim)

    def step(self, slots, k, v):
        _, n, _ = slots.shape

        q = self.slot_to_q(self.norm_slots(slots))
        attn = F.softmax(torch.einsum("bkd,bqd->bkq", k, q), dim=-1)
        attn = attn / torch.sum(attn + self.eps, dim=-2, keepdim=True)
        updates = torch.einsum("bvq,bvd->bqd", attn, v)

        # TODO: I hate this
        updates = rearrange(updates, "bt n d -> (bt n) d")
        slots = rearrange(slots, "bt n d -> (bt n) d")
        slots = self.gru(updates, slots)
        updates = rearrange(updates, "(bt n) d -> bt n d", n=n)
        slots = rearrange(slots, "(bt n) d -> bt n d", n=n)

        slots = slots + self.ffn(self.norm_ffn(slots))

        return slots

    def forward(self, x, init_slots=None, time=1):
        n_slots = self.n_slots
        if self.min_slots is not None:
            n_slots = torch.randint(self.min_slots, self.n_slots + 1).item()

        if init_slots is None:
            init_slots = self.mu + self.sigma * torch.randn(
                (x.shape[0] // time, n_slots, self.slot_dim), device=x.device
            )
            init_slots = repeat(init_slots, "b n d -> b t n d", t=time)

        init_slots = rearrange(init_slots, "b t n d -> (b t) n d")
        slots = init_slots.clone()

        x = self.input_norm(x)
        k, v = self.in_to_kv(x).chunk(2, dim=-1)
        k = k * (self.slot_dim**-0.5)

        for _ in range(self.n_iters - 1):
            slots = self.step(slots, k, v)
        slots = self.step(slots.detach(), k, v)

        slots = rearrange(slots, "(b t) n d -> b t n d", t=time)
        init_slots = rearrange(init_slots, "(b t) n d -> b t n d", t=time)

        return slots, init_slots.detach()


class ISARecurrent(pl.LightningModule):

    def __init__(
        self, input_dim, slot_dim, n_iters, n_slots=5, min_slots=None, eps=1e-8
    ):
        super().__init__()

        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.min_slots = min_slots
        self.eps = eps

        if self.min_slots is not None:
            assert self.min_slots <= self.n_slots

        self.mu = nn.Parameter(torch.randn(slot_dim))
        self.sigma = nn.Parameter(torch.randn(slot_dim))

        self.in_to_kv = nn.Linear(input_dim, slot_dim * 2, bias=False)
        self.slot_to_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.ffn = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, input_dim),
        )

        self.input_norm = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_ffn = nn.LayerNorm(slot_dim)
        self.norm_recurrent = nn.LayerNorm(slot_dim)

    def step(self, slots, k, v):
        _, n, _ = slots.shape

        q = self.slot_to_q(self.norm_slots(slots))
        attn = F.softmax(torch.einsum("bkd,bqd->bkq", k, q), dim=-1)
        attn = attn / torch.sum(attn + self.eps, dim=-2, keepdim=True)
        updates = torch.einsum("bvq,bvd->bqd", attn, v)

        # TODO: I hate this
        updates = rearrange(updates, "b n d -> (b n) d")
        slots = rearrange(slots, "b n d -> (b n) d")
        slots = self.gru(updates, slots)
        updates = rearrange(updates, "(b n) d -> b n d", n=n)
        slots = rearrange(slots, "(b n) d -> b n d", n=n)

        slots = slots + self.ffn(self.norm_ffn(slots))

        return slots

    def iterate(self, slots, k, v):
        for _ in range(self.n_iters - 1):
            slots = self.step(slots, k, v)
        slots = self.step(slots.detach(), k, v)
        return slots

    def forward(self, x, init_slots=None):
        _, t, _, _ = x.shape

        n_slots = self.n_slots
        if self.min_slots is not None:
            n_slots = torch.randint(self.min_slots, self.n_slots + 1).item()

        if init_slots is None:
            init_slots = self.mu + self.sigma * torch.randn(
                (x.shape[0], n_slots, self.slot_dim), device=x.device
            )

        slots = init_slots.clone()

        x = self.input_norm(x)
        k, v = self.in_to_kv(x).chunk(2, dim=-1)
        k = k * (self.slot_dim**-0.5)

        results = []
        for i in range(t):
            slots = self.iterate(slots, k[:, i], v[:, i])
            results.append(slots)
            slots = self.norm_recurrent(slots.detach())  # Maybe not necessary

        slots = torch.stack(results, dim=1)

        return slots, init_slots.detach()
