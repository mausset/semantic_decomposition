import lightning as pl
import matplotlib
import torch
from einops import rearrange, repeat
from geomloss import SamplesLoss
from models.components import GaussianDependent, GaussianPrior, SwiGLUFFN
from torch import nn

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

plt.ion()


class SA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_slots=8,
        n_iters=8,
        distance_threshold=None,
        cluster_drop_p=0.1,
        implicit=True,
        sampler="gaussian",
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.distance_threshold = distance_threshold
        self.cluster_drop_p = cluster_drop_p
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

    def inv_cross_attn(self, q, k, v, slot_mask=None, context_mask=None):

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if slot_mask is not None:
            dots.masked_fill_(~slot_mask[:, :, None], -torch.finfo(k.dtype).max)
        if context_mask is not None:
            dots.masked_fill_(~context_mask[:, None, :], -torch.finfo(k.dtype).max)
        attn = dots.softmax(dim=1)
        attn_vis = attn.clone()

        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjd,bij->bid", v, attn)

        return updates, attn_vis

    def step(self, slots, k, v, return_attn=False, slot_mask=None, context_mask=None):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        updates, attn = self.inv_cross_attn(
            q, k, v, slot_mask=slot_mask, context_mask=context_mask
        )

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

            if torch.rand(1) < self.cluster_drop_p or labels.max() < 2:
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

    def forward(self, x, context_mask=None):

        x = self.norm_input(x)

        init_slots = self.sample(x, self.n_slots)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        with torch.no_grad():
            for _ in range(self.n_iters):
                slots = self.step(slots, k, v, context_mask=context_mask)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(
            slots, k, v, return_attn=True, context_mask=context_mask
        )

        ret = {}
        if self.distance_threshold:
            slots, attn_map, mask = self.cluster_slots(slots, attn_map)
            ret["mask"] = mask

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        ret = {"slots": slots, "attn": attn_map}
        return ret


class TSA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        implicit=True,
        sampler="gaussian",
        threshold=0.1,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.implicit = implicit
        self.threshold = threshold
        self.eps = eps

        self.scale = input_dim**-0.5

        if sampler == "gaussian":
            self.sampler = GaussianPrior(slot_dim)
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

    def inv_cross_attn(self, q, k, v, slot_mask=None, context_mask=None):

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if slot_mask is not None:
            dots.masked_fill_(~slot_mask[:, :, None], -torch.finfo(k.dtype).max)
        if context_mask is not None:
            dots.masked_fill_(~context_mask[:, None, :], -torch.finfo(k.dtype).max)
        attn = dots.softmax(dim=1)
        attn_vis = attn.clone()

        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjd,bij->bid", v, attn)

        return updates, attn_vis, dots

    def step(self, slots, k, v, return_attn=False, return_dots=False, slot_mask=None, context_mask=None):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        updates, attn, dots = self.inv_cross_attn(q, k, v, slot_mask=slot_mask, context_mask=context_mask)

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

        if return_dots:
            return slots, dots

        if return_attn:
            return slots, attn

        return slots

    def select_slots(self, slots, dots, mask):
        b = slots.shape[0]

        slots = rearrange(slots, "b n ... -> (b n) ...")
        dots = rearrange(dots, "b n ... -> (b n) ...")
        mask = rearrange(mask, "b n ... -> (b n) ...")

        left = dots.softmax(dim=-1)
        right = rearrange(dots, "... m n -> ... n m").softmax(dim=-1)
        s_to_s = torch.bmm(left, right)

        trace = torch.vmap(torch.trace)(s_to_s) / mask.sum(dim=-1)
        trace = (1 - trace) / torch.sqrt(mask.sum(dim=-1))
        trace = rearrange(trace, "(b n) -> b n", b=b)

        idx = [torch.where(row < self.threshold)[0] for row in trace]
        idx = torch.tensor([row[-1] if row.shape[0] > 0 else 1 for row in idx], device=slots.device)
        torch.clamp_(idx, min=2)

        idx = repeat(
            idx,
            "b -> b 1 m d",
            m=slots.shape[1],  # type: ignore
            d=slots.shape[2],  # type: ignore
        )

        slots = rearrange(slots, "(b n) ... -> b n ...", b=b)
        mask = rearrange(mask, "(b n) ... -> b n ...", b=b)
        slots = torch.gather(slots, 1, idx).squeeze()  # type: ignore
        mask = torch.gather(mask, 1, idx[:, :, :, 0]).squeeze()  # type: ignore

        return slots, mask

    def forward(self, x, context_mask=None):
        x = self.norm_input(x)

        init_slots = self.sample(x, self.n_slots)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        with torch.no_grad():
            slots = repeat(slots, "b ... -> (b n) ...", n=self.n_slots)
            k_batch = repeat(k, "b ... -> (b n) ...", n=self.n_slots)
            v_batch = repeat(v, "b ... -> (b n) ...", n=self.n_slots)
            batch_context_mask = None
            if context_mask is not None:
                batch_context_mask = repeat(context_mask, "b ... -> (b n) ...", n=self.n_slots)
            slot_mask = torch.tril(
                torch.ones(
                    self.n_slots,
                    self.n_slots,
                    device=x.device,
                    dtype=torch.bool,
                )
            )
            slot_mask = repeat(slot_mask, "n ... -> (b n) ...", b=x.shape[0])

            for _ in range(self.n_iters):
                slots, dots = self.step(  # type: ignore
                    slots, k_batch, v_batch, slot_mask=slot_mask, context_mask=batch_context_mask, return_dots=True
                )

            slots = rearrange(slots, "(b n) ... -> b n ...", b=x.shape[0])
            dots = rearrange(dots, "(b n) ... -> b n ...", b=x.shape[0])  # type: ignore
            slot_mask = rearrange(slot_mask, "(b n) ... -> b n ...", b=x.shape[0])
            slots, slot_mask = self.select_slots(slots, dots, slot_mask)  # type: ignore

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True, slot_mask=slot_mask)
        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        ret = {"slots": slots, "attn": attn_map, "mask": slot_mask}
        return ret
