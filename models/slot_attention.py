import lightning as pl
import torch
from torch import nn
from einops import rearrange

from .utils import SwiGLUFFN, ConceptBank


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

        self.concept_bank = ConceptBank(slot_dim)

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, input_dim),
        )

        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

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

    def forward(self, x):
        _, t, _, _ = x.shape

        x = rearrange(x, "b t n d -> (b t) n d")

        x = self.norm_input(x)

        init_slots = self.concept_bank(x)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        for _ in range(self.n_iters):
            slots = self.step(slots, k, v)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots
            slots = self.step(slots, k, v)

        slots = rearrange(slots, "(b t) n d -> b t n d", t=t)

        return slots


class SAV2(pl.LightningModule):
    "Transformer layer with inverted cross attention."

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_slots=8,
        n_iters=3,
        implicit=True,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.implicit = implicit
        self.eps = eps

        self.scale = input_dim**-0.5

        self.concept_bank = ConceptBank(slot_dim)

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_ica = nn.LayerNorm(slot_dim)
        self.norm_ffn = nn.LayerNorm(slot_dim)

        self.ffn = SwiGLUFFN(slot_dim)

    def step(self, slots, k, v):

        q = self.inv_cross_q(self.norm_slots(slots))

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        updates = torch.einsum("bjd,bij->bid", v, attn)

        slots = self.norm_ica(slots + updates)

        slots = self.norm_ffn(self.ffn(slots) + slots)

        return slots

    def forward(self, x):
        _, t, _, _ = x.shape

        x = rearrange(x, "b t n d -> (b t) n d")

        x = self.norm_input(x)

        init_slots = self.concept_bank(x)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        for _ in range(self.n_iters):
            slots = self.step(slots, k, v)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots
            slots = self.step(slots, k, v)

        slots = rearrange(slots, "(b t) n d -> b t n d", t=t)

        return slots
