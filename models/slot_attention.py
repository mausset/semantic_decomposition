import lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn
from x_transformers import Encoder

from models.components import SwiGLUFFN


def build_slot_attention(arch: str, args):

    if arch == "sa":
        return SA(**args)
    elif arch == "sat":
        return SAT(**args)
    elif arch == "rsa":
        return RSA(**args)
    else:
        raise ValueError(f"Unknown slot attention architecture: {arch}")


class SA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=3,
        implicit=False,
        sample_strategy="prior",
        ff_swiglu=False,
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

        self.init_mu = nn.Parameter(torch.randn(slot_dim))
        init_log_sigma = torch.empty((1, 1, slot_dim))
        nn.init.xavier_uniform_(init_log_sigma)
        self.init_log_sigma = nn.Parameter(init_log_sigma.squeeze())

        if sample_strategy == "learned":
            self.dec = Encoder(
                dim=slot_dim,
                depth=4,
                ff_glu=True,
                ff_swish=True,
                cross_attend=True,
            )

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

    def sample(self, x, n_slots):
        b, _, _ = x.shape

        mu = repeat(self.init_mu, "d -> b n d", b=b, n=n_slots)
        log_sigma = repeat(self.init_log_sigma, "d -> b n d", b=b, n=n_slots)

        sample = mu + log_sigma.exp() * torch.randn_like(mu)

        if self.sample_strategy == "learned":
            sample = self.dec(sample, context=x)

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

        init_slots = self.sample(x, n_slots)
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


class SAT(nn.Module):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=3,
        implicit=False,
        depth=1,
        gru=False,
        sample_strategy="prior",
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.implicit = implicit
        self.gru = gru
        self.sample_strategy = sample_strategy
        self.eps = eps

        self.scale = input_dim**-0.5

        self.init_mu = nn.Parameter(torch.randn(slot_dim))
        init_log_sigma = torch.empty((1, 1, slot_dim))
        nn.init.xavier_uniform_(init_log_sigma)
        self.init_log_sigma = nn.Parameter(init_log_sigma.squeeze())

        if sample_strategy == "learned":
            self.dec = Encoder(
                dim=slot_dim,
                depth=4,
                ff_glu=True,
                ff_swish=True,
            )

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        if gru:
            self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.t_encoder = Encoder(
            dim=input_dim,
            depth=depth,
            ff_glu=True,
            ff_swish=True,
        )

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_ica = nn.LayerNorm(slot_dim)

    def sample(self, x, n_slots):
        b, _, _ = x.shape

        mu = repeat(self.init_mu, "d -> b n d", b=b, n=n_slots)
        log_sigma = repeat(self.init_log_sigma, "d -> b n d", b=b, n=n_slots)

        sample = mu + log_sigma.exp() * torch.randn_like(mu)

        if self.sample_strategy == "learned":
            sample = self.dec(sample, context=x)

        return sample

    def step(self, slots, k, v, return_attn=False):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps

        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjd,bij->bid", v, attn)

        if self.gru:
            updates = torch.einsum("bjd,bij->bid", v, attn)
            slots = rearrange(slots, "b n d -> (b n) d")
            updates = rearrange(updates, "b n d -> (b n) d")
            slots = self.gru(updates, slots)
            slots = rearrange(slots, "(b n) d -> b n d", n=n)
        else:
            slots = slots + updates

        slots = self.norm_ica(slots)

        slots = self.t_encoder(slots)

        if return_attn:
            return slots, attn

        return slots

    def forward(self, x, n_slots=8):

        x = self.norm_input(x)

        init_slots = self.sample(x, n_slots)
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


class RSA(nn.Module):

    def __init__(
        self,
        input_dim,
        slot_dim,
        depth=4,
        sample_strategy="prior",
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.sample_strategy = sample_strategy
        self.eps = eps

        self.scale = input_dim**-0.5

        if sample_strategy == "prior":
            self.init_mu = nn.Parameter(torch.randn(slot_dim))
            init_log_sigma = torch.empty((1, 1, slot_dim))
            nn.init.xavier_uniform_(init_log_sigma)
            self.init_log_sigma = nn.Parameter(init_log_sigma.squeeze())

        elif sample_strategy == "learned":
            self.init_mu = nn.Parameter(torch.randn(slot_dim))
            init_log_sigma = torch.empty((1, 1, slot_dim))
            nn.init.xavier_uniform_(init_log_sigma)
            self.init_log_sigma = nn.Parameter(init_log_sigma.squeeze())

            self.s_dec = Encoder(
                dim=slot_dim,
                depth=4,
                ff_glu=True,
                ff_swish=True,
                cross_attend=True,
            )

        self.inv_cross_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.inv_cross_q = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.encoder_layers = nn.ModuleList([SwiGLUFFN(slot_dim) for _ in range(depth)])

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_ica = nn.LayerNorm(slot_dim)

    def sample(self, x, n_slots):
        b, _, _ = x.shape

        mu = repeat(self.init_mu, "d -> b n d", b=b, n=n_slots)
        log_sigma = repeat(self.init_log_sigma, "d -> b n d", b=b, n=n_slots)

        sample = mu + log_sigma.exp() * torch.randn_like(mu)

        if self.sample_strategy == "learned":
            sample = self.s_dec(sample, context=x)

        return sample

    def ica(self, slots, k, v):
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
        slots = self.norm_ica(slots)

        return slots, attn

    def forward(self, x, n_slots=8):

        x = self.norm_input(x)

        slots = self.sample(x, n_slots)

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        for layer in self.encoder_layers:
            slots, attn_map = self.ica(slots, k, v)
            slots = layer(slots) + slots

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map
