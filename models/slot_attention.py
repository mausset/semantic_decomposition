import lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn

from models.components import GaussianPrior, GaussianDependent
from sklearn.cluster import AgglomerativeClustering


class SA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        distance_threshold=None,
        implicit=True,
        sampler="gaussian",
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.distance_threshold = distance_threshold
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

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 4, slot_dim),
        )

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

    def sample(self, x, n_slots, sample=None):
        if sample is not None:
            return sample

        if isinstance(self.sampler, nn.Parameter):
            return repeat(self.sampler, "n d -> b n d", b=x.shape[0])
        return self.sampler(x, n_slots)  # type: ignore

    def inv_cross_attn(self, q, k, v, mask=None):

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if mask is not None:
            # NOTE: Masking attention is sensitive to the chosen value
            dots.masked_fill_(~mask[:, None, :], -torch.finfo(k.dtype).max)
        attn = dots.softmax(dim=1)
        attn_vis = attn.clone()

        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjd,bij->bid", v, attn)

        return updates, attn_vis

    def step(self, slots, k, v, return_attn=False, mask=None):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        updates, attn = self.inv_cross_attn(q, k, v, mask=mask)

        slots = rearrange(slots, "b n d -> (b n) d")
        updates = rearrange(updates, "b n d -> (b n) d")

        # NOTE: GRUCell does not support bf16 before PyTorch 2.3
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

    def cluster_slots(self, slots, attn):
        b, n, _ = slots.shape

        cluster_means = torch.zeros_like(slots, device=slots.device)
        cluster_attn = torch.zeros_like(attn, device=slots.device)
        mask = torch.zeros(b, n, device=slots.device, dtype=torch.bool)
        for i in range(b):
            labels = AgglomerativeClustering(
                metric="cosine",
                compute_full_tree=True,  # type: ignore
                linkage="complete",
                n_clusters=None,  # type: ignore
                distance_threshold=self.distance_threshold,
            ).fit_predict(slots[i].cpu().detach().numpy())

            if labels.max() < 2:
                cluster_means[i] = slots[i]
                cluster_attn[i] = attn[i]
                mask[i] = True
                continue

            mask[i, : labels.max() + 1] = True
            for j in range(labels.max() + 1):
                cluster_mask = torch.tensor(labels == j, device=slots.device)
                cluster_means[i, j] = slots[i, cluster_mask].mean(dim=0)
                cluster_attn[i, j] = attn[i, cluster_mask].sum(dim=0)

        return cluster_means, cluster_attn, mask

    def forward(self, x, n_slots=8, mask=None):

        x = self.norm_input(x)

        init_slots = self.sample(x, n_slots)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        for _ in range(self.n_iters):
            slots = self.step(slots, k, v, mask=mask)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True, mask=mask)

        if self.distance_threshold is not None:
            slots, attn_map, mask = self.cluster_slots(slots, attn_map)

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map, mask
