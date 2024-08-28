import lightning as pl
import matplotlib
import torch
from einops import rearrange, repeat
from geomloss import SamplesLoss
from models.components import GaussianDependent, GaussianPrior, SwiGLUFFN
from torch import nn

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.ion()


class SA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        implicit=True,
        ff_swiglu=False,
        sampler="gaussian",
        vis_post_weight_attn=False,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.implicit = implicit
        self.vis_post_weight_attn = vis_post_weight_attn
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
        if self.vis_post_weight_attn:
            attn_vis = attn

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

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map


class PSA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        implicit=True,
        sampler="gaussian",
        convergence_threshold=0.1,
        vis_post_weight_attn=False,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.implicit = implicit
        self.convergence_threshold = convergence_threshold
        self.vis_post_weight_attn = vis_post_weight_attn
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

        self.dist = SamplesLoss("sinkhorn", p=2, blur=0.05)

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
            dots.masked_fill_(~mask[:, :, None], -torch.finfo(k.dtype).max)
        attn = dots.softmax(dim=1)
        attn_vis = attn.clone()

        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        if self.vis_post_weight_attn:
            attn_vis = attn

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

    def select_slots(self, slots, mask):
        b = slots.shape[0]

        weights = mask / mask.sum(dim=1, keepdim=True)
        weights = rearrange(weights, "(b n) m -> b n m", b=b)
        weights_left = rearrange(weights[:, :-1], "b n m -> (b n) m")
        weights_right = rearrange(weights[:, 1:], "b n m -> (b n) m")

        slots_left = rearrange(slots[:, :-1], "b n ... -> (b n) ...")  # type: ignore
        slots_right = rearrange(slots[:, 1:], "b n ... -> (b n) ...")  # type: ignore

        dists = self.dist(weights_left, slots_left, weights_right, slots_right)
        dists = rearrange(dists, "(b n) -> b n", b=b)
        dists_left = rearrange(dists[:, :-1], "b n -> (b n)")
        dists_right = rearrange(dists[:, 1:], "b n -> (b n)")

        rel_imp = torch.abs(dists_right - dists_left) / dists_left
        rel_imp = rearrange(rel_imp, "(b n) -> b n", b=b)
        rel_imp_mask = 1 - (rel_imp > self.convergence_threshold).int()
        rel_imp_mask[:, -1] = 1

        idx = repeat(
            torch.argmax(rel_imp_mask, dim=1) + 2,
            "b -> b 1 m d",
            m=slots.shape[2],  # type: ignore
            d=slots.shape[3],  # type: ignore
        )

        slots = torch.gather(slots, 1, idx).squeeze()  # type: ignore
        mask = rearrange(mask, "(b n) m -> b n m", b=b)
        mask = torch.gather(mask, 1, idx[:, :, :, 0]).squeeze()  # type: ignore

        return slots, mask

    def forward(self, x):
        x = self.norm_input(x)

        init_slots = self.sample(x, self.n_slots)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        with torch.no_grad():
            slots = repeat(slots, "b ... -> (b n) ...", n=self.n_slots)
            k_batch = repeat(k, "b ... -> (b n) ...", n=self.n_slots)
            v_batch = repeat(v, "b ... -> (b n) ...", n=self.n_slots)
            mask = torch.tril(
                torch.ones(
                    self.n_slots,
                    self.n_slots,
                    device=x.device,
                    dtype=torch.bool,
                )
            )
            mask = repeat(mask, "n ... -> (b n) ...", b=x.shape[0])

            for _ in range(self.n_iters):
                slots = self.step(slots, k_batch, v_batch, mask=mask)

            slots = rearrange(slots, "(b n) ... -> b n ...", b=x.shape[0])
            slots, mask = self.select_slots(slots, mask)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True, mask=mask)
        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map, mask


class ESA(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        implicit=True,
        sampler="gaussian",
        convergence_threshold=0.1,
        vis_post_weight_attn=False,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.implicit = implicit
        self.convergence_threshold = convergence_threshold
        self.vis_post_weight_attn = vis_post_weight_attn
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

    def inv_cross_attn(self, q, k, v, mask=None):

        dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if mask is not None:
            # NOTE: Masking attention is sensitive to the chosen value
            dots.masked_fill_(~mask[:, :, None], -torch.finfo(k.dtype).max)
        attn = dots.softmax(dim=1)
        attn_vis = attn.clone()

        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        if self.vis_post_weight_attn:
            attn_vis = attn

        updates = torch.einsum("bjd,bij->bid", v, attn)

        return updates, attn_vis, dots

    def step(self, slots, k, v, return_attn=False, return_dots=False, mask=None):
        _, n, _ = slots.shape

        q = self.inv_cross_q(self.norm_slots(slots))

        updates, attn, dots = self.inv_cross_attn(q, k, v, mask=mask)

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

        dots = dots.masked_fill(~mask[:, :, None], -torch.finfo(dots.dtype).max)
        left = dots.softmax(dim=-1)

        right = rearrange(dots, "... m n -> ... n m").softmax(dim=-1)
        s_to_s = torch.bmm(left, right)

        trace = torch.vmap(torch.trace)(s_to_s) / mask.sum(dim=-1)
        trace = (1 - trace) / torch.sqrt(mask.sum(dim=-1))
        trace = rearrange(trace, "(b n) -> b n", b=b)

        idx = torch.tensor(
            [torch.where(row < self.convergence_threshold)[0][-1] for row in trace],
            device=slots.device,
        )
        torch.clamp_(idx, min=1)

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

    def forward(self, x):
        x = self.norm_input(x)

        init_slots = self.sample(x, self.n_slots)
        slots = init_slots.clone()

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        with torch.no_grad():
            slots = repeat(slots, "b ... -> (b n) ...", n=self.n_slots)
            k_batch = repeat(k, "b ... -> (b n) ...", n=self.n_slots)
            v_batch = repeat(v, "b ... -> (b n) ...", n=self.n_slots)
            mask = torch.tril(
                torch.ones(
                    self.n_slots,
                    self.n_slots,
                    device=x.device,
                    dtype=torch.bool,
                )
            )
            mask = repeat(mask, "n ... -> (b n) ...", b=x.shape[0])

            for _ in range(self.n_iters):
                slots, dots = self.step(  # type: ignore
                    slots, k_batch, v_batch, mask=mask, return_dots=True
                )

            slots = rearrange(slots, "(b n) ... -> b n ...", b=x.shape[0])
            dots = rearrange(dots, "(b n) ... -> b n ...", b=x.shape[0])  # type: ignore
            mask = rearrange(mask, "(b n) ... -> b n ...", b=x.shape[0])
            slots, mask = self.select_slots(slots, dots, mask)  # type: ignore

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True, mask=mask)
        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map, mask
