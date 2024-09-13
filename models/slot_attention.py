import math

import torch
from einops import rearrange, repeat
from models.components import GaussianDependent, GaussianPrior
from torch import nn


class SA(nn.Module):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        distance_threshold=None,
        linkage="single",
        implicit=True,
        cluster_pre=False,
        sampler="gaussian",
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.distance_threshold = distance_threshold
        self.cluster_pre = cluster_pre
        self.implicit = implicit
        self.eps = eps

        self.scale = input_dim**-0.5

        if sampler == "gaussian":
            self.sampler = GaussianPrior(slot_dim)
        elif sampler == "gaussian_dependent":
            self.sampler = GaussianDependent(slot_dim)
        elif sampler == "embedding":
            self.sampler = nn.Parameter(torch.randn(n_slots, slot_dim))

        if linkage == "single":
            self.cluster_slots_vectorized = self.cluster_slots_vectorized_single
        elif linkage == "complete":
            self.cluster_slots_vectorized = self.cluster_slots_vectorized_complete

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

    def inv_cross_attn(self, q, k, v, slot_maske, context_mask):

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if context_mask is not None:
            # NOTE: Masking attention is sensitive to the chosen value
            dots.masked_fill_(~context_mask[:, None, :], -torch.finfo(k.dtype).max)
        if slot_maske is not None:
            dots.masked_fill_(~slot_maske[:, :, None], -torch.finfo(k.dtype).max)
        attn = dots.softmax(dim=1)
        attn_vis = attn.clone()

        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjd,bij->bid", v, attn)

        return updates, attn_vis

    def step(self, slots, k, v, return_attn=False, slot_mask=None, context_mask=None):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        updates, attn = self.inv_cross_attn(q, k, v, slot_mask, context_mask)

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

    def cluster_slots_vectorized_complete(self, slots, attn):
        with torch.no_grad():
            norm_slots = torch.nn.functional.normalize(slots, p=2, dim=-1)
            cosine_distance = 1 - torch.einsum("bnd,bmd->bnm", norm_slots, norm_slots)

            M = cosine_distance.clone()

            for k in range(self.n_slots):
                M = torch.minimum(
                    M, torch.maximum(M[:, :, k : k + 1], M[:, k : k + 1, :])
                )

            adjacency_matrix = (M < self.distance_threshold).float()
            for _ in range(math.ceil(math.log2(self.n_slots))):
                adjacency_matrix = torch.clamp(
                    adjacency_matrix @ adjacency_matrix, max=1
                )

            cumsum = adjacency_matrix.cumsum(dim=1)
            cluster_mask = adjacency_matrix * (cumsum <= 1)
            counts = cluster_mask.any(dim=2).sum(dim=1)
            reset_idx = counts < 2
            cluster_mask[reset_idx] = torch.eye(
                self.n_slots,
                device=cluster_mask.device,
                dtype=cluster_mask.dtype,
            )[None]

            cluster_weight = cluster_mask / torch.clamp(
                cluster_mask.sum(dim=2, keepdim=True), min=1
            )

        cluster_means = torch.bmm(cluster_weight, slots)
        cluster_attn = torch.bmm(cluster_mask, attn)
        cluster_mask = cluster_mask.any(dim=2)

        return cluster_means, cluster_attn, cluster_mask

    def cluster_slots_vectorized_single(self, slots, attn):
        with torch.no_grad():
            norm_slots = torch.nn.functional.normalize(slots, p=2, dim=-1)
            cosine_distance = 1 - torch.einsum("bnd,bmd->bnm", norm_slots, norm_slots)

            adjacency_matrix = (cosine_distance < self.distance_threshold).float()
            for _ in range(math.ceil(math.log2(self.n_slots))):
                adjacency_matrix = torch.clamp(
                    adjacency_matrix @ adjacency_matrix, max=1
                )

            cumsum = adjacency_matrix.cumsum(dim=1)
            cluster_mask = adjacency_matrix * (cumsum <= 1)
            counts = cluster_mask.any(dim=2).sum(dim=1)
            reset_idx = counts < 2

            cluster_mask[reset_idx] = torch.eye(
                self.n_slots,
                device=cluster_mask.device,
                dtype=cluster_mask.dtype,
            )[None]

            cluster_weight = cluster_mask / torch.clamp(
                cluster_mask.sum(dim=2, keepdim=True), min=1
            )

        cluster_means = torch.bmm(cluster_weight, slots)
        cluster_attn = torch.bmm(cluster_mask, attn)
        cluster_mask = cluster_mask.any(dim=2)

        return cluster_means, cluster_attn, cluster_mask

    def forward(self, x, n_slots=8, context_mask=None):

        x = self.norm_input(x)

        init_slots = self.sample(x, n_slots)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        for _ in range(self.n_iters):
            slots, attn_map = self.step(
                slots,
                k,
                v,
                context_mask=context_mask,
                return_attn=True,
            )

        slot_mask = torch.ones(x.shape[0], n_slots, device=x.device, dtype=torch.bool)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        if self.cluster_pre and self.distance_threshold is not None:
            slots, _, slot_mask = self.cluster_slots_vectorized(slots, attn_map)

        slots, attn_map = self.step(
            slots,
            k,
            v,
            return_attn=True,
            slot_mask=slot_mask,
            context_mask=context_mask,
        )

        if not self.cluster_pre and self.distance_threshold is not None:
            slots, attn_map, slot_mask = self.cluster_slots_vectorized(slots, attn_map)

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map, slot_mask
