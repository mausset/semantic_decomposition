import lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn

from models.components import SwiGLUFFN, GaussianPrior, GaussianDependent


class SA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=3,
        n_slots=8,
        implicit=False,
        ff_swiglu=False,
        sampler="gaussian",
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.implicit = implicit
        self.eps = eps

        self.scale = input_dim**-0.5

        if sampler == "gaussian":
            self.sampler = GaussianPrior(slot_dim)
        elif sampler == "gaussian_dependent":
            self.sampler = GaussianDependent(slot_dim)
        elif sampler == "embedding":
            self.sampler = nn.Parameter(torch.randn(n_slots, slot_dim))

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        if ff_swiglu:
            self.mlp = SwiGLUFFN(slot_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim * 4),
                nn.ReLU(inplace=True),
                nn.Linear(slot_dim * 4, slot_dim),
            )

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

    def inv_cross_attn(self, q, k, v, mask=None):

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if mask is not None:
            # NOTE: Masking attention is sensitive to the chosen value
            dots.masked_fill_(~mask[:, None, :], -torch.finfo(k.dtype).max)
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        updates = torch.einsum("bjd,bij->bid", v, attn)

        return updates, attn

    def sample(self, x, n_slots, sample=None):
        if sample is not None:
            return sample

        if isinstance(self.sampler, nn.Parameter):
            return repeat(self.sampler, "n d -> b n d", b=x.shape[0])
        return self.sampler(x, n_slots)  # type: ignore

    def step(self, slots, k, v, return_attn=False):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        updates, attn = self.inv_cross_attn(q, k, v)

        slots = rearrange(slots, "b n d -> (b n) d")
        updates = rearrange(updates, "b n d -> (b n) d")

        # NOTE: GRUCell does not support bf16
        if k.device.type == "cuda":
            with torch.autocast(device_type=k.device.type, dtype=torch.float32):
                slots = self.gru(updates, slots)
        else:  # MPS / CPU
            slots = self.gru(updates, slots)

        slots = rearrange(slots, "(b n) d -> b n d", n=n)

        slots = slots + self.mlp(self.norm_pre_ff(slots))

        if return_attn:
            return slots, attn

        return slots

    def forward(self, x, n_slots=8, sample=None):

        x = self.norm_input(x)

        init_slots = self.sample(x, n_slots, sample=sample)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        for _ in range(self.n_iters):
            slots = self.step(slots, k, v)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True)

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map
