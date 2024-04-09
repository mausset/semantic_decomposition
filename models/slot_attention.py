import lightning as pl
from numpy import who
import torch
from torch import nn
from einops import rearrange, repeat

from models.components import CodeBook

from x_transformers import Encoder


class SA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=3,
        implicit=False,
        sample_strategy="prior",
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.implicit = implicit
        self.sample_strategy = sample_strategy
        self.eps = eps

        self.scale = input_dim**-0.5

        if sample_strategy == "prior":
            self.init_mu = nn.Parameter(torch.randn(slot_dim))
            init_log_sigma = torch.empty((1, 1, slot_dim))
            nn.init.xavier_uniform_(init_log_sigma)
            self.init_log_sigma = nn.Parameter(init_log_sigma.squeeze())
        elif sample_strategy == "learned":
            self.dist_encoder = Encoder(
                dim=slot_dim,
                depth=2,
                ff_glu=True,
                ff_swish=True,
            )

            self.mu_projection = nn.Sequential(
                nn.Linear(slot_dim, slot_dim, bias=False)
                # nn.ReLU(inplace=True),
                # nn.Linear(slot_dim, slot_dim),
            )
            self.log_sigma_projection = nn.Sequential(
                nn.Linear(slot_dim, slot_dim, bias=False)
                # nn.ReLU(inplace=True),
                # nn.Linear(slot_dim, slot_dim),
            )

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, slot_dim),
        )

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

    def sample_prior(self, b, n_slots):
        mu = repeat(self.init_mu, "d -> b n d", b=b, n=n_slots)
        log_sigma = repeat(self.init_log_sigma, "d -> b n d", b=b, n=n_slots)

        sample = mu + log_sigma.exp() * torch.randn_like(mu)

        return sample

    def sample_learned(self, x, n_slots):
        x = self.dist_encoder(x).mean(dim=1)
        mu = self.mu_projection(x)
        log_sigma = self.log_sigma_projection(x)

        mu = repeat(mu, "b d -> b n d", n=n_slots)
        log_sigma = repeat(log_sigma, "b d -> b n d", n=n_slots)

        sample = mu + log_sigma.exp() * torch.randn_like(mu)

        return sample

    def step(self, slots, k, v, return_attn=False):
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

        if return_attn:
            return slots, attn

        return slots

    def forward(self, x, n_slots=8):
        b, _, _ = x.shape

        x = self.norm_input(x)

        if self.sample_strategy == "prior":
            init_slots = self.sample_prior(b, n_slots)
        elif self.sample_strategy == "learned":
            init_slots = self.sample_learned(x, n_slots)

        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        for _ in range(self.n_iters):
            slots = self.step(slots, k, v)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots
            slots, attn_map = self.step(slots, k, v, return_attn=True)

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map
