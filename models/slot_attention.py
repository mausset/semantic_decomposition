import lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn

from models.components import SwiGLUFFN, GaussianPrior, GaussianDependent
from geomloss import SamplesLoss


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
        relative_error=0.1,
        vis_post_weight_attn=False,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.implicit = implicit
        self.relative_error = relative_error
        self.vis_post_weight_attn = vis_post_weight_attn
        self.eps = eps

        self.scale = input_dim**-0.5

        self.sampler = GaussianPrior(slot_dim)

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

    def forward(self, x):

        x = self.norm_input(x)

        init_slots = self.sample(x, self.n_slots)
        slots = init_slots.clone()

        n_slots = 1
        mask = torch.zeros(x.shape[0], self.n_slots, device=x.device, dtype=torch.bool)
        mask[:, :n_slots] = True

        k = self.inv_cross_k(x)
        v = self.inv_cross_v(x)

        with torch.no_grad():
            for _ in range(self.n_iters):
                slots[:, :n_slots] = self.step(slots[:, :n_slots], k, v)  # type: ignore

            prev_slots = slots.clone()  # type: ignore
            n_slots += 1
            mask[:, :n_slots] = True
            for _ in range(self.n_iters):
                slots[:, :n_slots] = self.step(slots[:, :n_slots], k, v)  # type: ignore

            dist = self.dist(slots[:, :n_slots], prev_slots[:, : n_slots - 1])  # type: ignore

            n_slots += 1
            do_update = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
            while torch.any(do_update) and n_slots <= self.n_slots:  # type: ignore
                new_slots = slots.clone()  # type: ignore
                for _ in range(self.n_iters):
                    new_slots[do_update, :n_slots] = self.step(  # type: ignore
                        slots[do_update, :n_slots], k[do_update], v[do_update]
                    )

                idx = torch.arange(x.shape[0], device=x.device)[do_update]
                new_dist = self.dist(
                    new_slots[do_update, :n_slots], slots[do_update, : n_slots - 1]
                )
                rel_error = torch.abs(new_dist - dist[do_update]) / dist[do_update]
                should_update = rel_error > self.relative_error
                do_update[idx[~should_update]] = False
                dist[do_update] = new_dist[should_update]
                slots[do_update] = new_slots[do_update]
                mask[do_update, n_slots - 1] = True
                n_slots += 1

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True, mask=mask)

        attn_map = rearrange(attn_map, "b n m -> b m n")

        print(mask)

        return slots, attn_map, mask


class PSA2(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        implicit=True,
        relative_error=0.1,
        vis_post_weight_attn=False,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.implicit = implicit
        self.relative_error = relative_error
        self.vis_post_weight_attn = vis_post_weight_attn
        self.eps = eps

        self.scale = input_dim**-0.5

        self.sampler = GaussianPrior(slot_dim)

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
                    self.n_slots, self.n_slots, device=x.device, dtype=torch.bool
                )
            )
            mask = repeat(mask, "n ... -> (b n) ...", b=x.shape[0])

            for _ in range(self.n_iters):
                slots = self.step(slots, k_batch, v_batch, mask=mask)

            weights = mask / mask.sum(dim=1, keepdim=True)
            weights = rearrange(weights, "(b n) m -> b n m", b=x.shape[0])
            weights_left = rearrange(weights[:, :-1], "b n m -> (b n) m")
            weights_right = rearrange(weights[:, 1:], "b n m -> (b n) m")

            slots = rearrange(slots, "(b n) ... -> b n ...", b=x.shape[0])
            slots_left = rearrange(slots[:, :-1], "b n ... -> (b n) ...")  # type: ignore
            slots_right = rearrange(slots[:, 1:], "b n ... -> (b n) ...")  # type: ignore

            dists = self.dist(weights_left, slots_left, weights_right, slots_right)
            dists = rearrange(dists, "(b n) -> b n", b=x.shape[0])
            dists_left = rearrange(dists[:, :-1], "b n -> (b n)")
            dists_right = rearrange(dists[:, 1:], "b n -> (b n)")

            rel_imp = torch.abs(dists_right - dists_left) / dists_left
            rel_imp = rearrange(rel_imp, "(b n) -> b n", b=x.shape[0])
            rel_imp_mask = 1 - (rel_imp > self.relative_error).int()
            rel_imp_mask[:, -1] = 1

            idx = repeat(
                torch.argmax(rel_imp_mask, dim=1) + 2,
                "b -> b 1 m d",
                m=slots.shape[2],  # type: ignore
                d=slots.shape[3],  # type: ignore
            )

            slots = torch.gather(slots, 1, idx).squeeze()  # type: ignore
            mask = rearrange(mask, "(b n) m -> b n m", b=x.shape[0])
            mask = torch.gather(mask, 1, idx[:, :, :, 0]).squeeze()  # type: ignore

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True, mask=mask)

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map, mask


class PSA3(pl.LightningModule):

    def __init__(
        self,
        input_dim,
        slot_dim,
        n_iters=8,
        n_slots=8,
        implicit=True,
        relative_error=0.1,
        vis_post_weight_attn=False,
        eps=1e-8,
    ):
        super().__init__()
        self.in_dim = input_dim
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.n_slots = n_slots
        self.implicit = implicit
        self.relative_error = relative_error
        self.vis_post_weight_attn = vis_post_weight_attn
        self.eps = eps

        self.scale = input_dim**-0.5

        self.sampler = GaussianPrior(slot_dim)

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

    def normalized_finite_differences(self, distances):
        h = 1.0

        differences = torch.zeros_like(distances)
        differences[:, 1:-1] = (distances[:, 2:] - distances[:, :-2]) / (2 * h)
        differences[:, 0] = (
            -3 * distances[:, 0] + 4 * distances[:, 1] - distances[:, 2]
        ) / (2 * h)
        differences[:, -1] = (
            3 * distances[:, -1] - 4 * distances[:, -2] + distances[:, -3]
        ) / (2 * h)
        return differences / distances

    def select_slots(self, slots, mask):
        b = slots.shape[0]
        slots = rearrange(slots, "b n m d -> (b n) m d")
        mask = rearrange(mask, "b n m -> (b n) m")

        weights = mask / mask.sum(dim=1, keepdim=True)
        weights = rearrange(weights, "(b n) m -> b n m", b=b)
        weights_left = rearrange(weights[:, :-1], "b n m -> (b n) m")
        weights_right = rearrange(weights[:, 1:], "b n m -> (b n) m")

        slots = rearrange(slots, "(b n) ... -> b n ...", b=b)
        slots_left = rearrange(slots[:, :-1], "b n ... -> (b n) ...")  # type: ignore
        slots_right = rearrange(slots[:, 1:], "b n ... -> (b n) ...")  # type: ignore

        dists = self.dist(weights_left, slots_left, weights_right, slots_right)
        dists = rearrange(dists, "(b n) -> b n", b=b)
        normalized_fd = self.normalized_finite_differences(dists)

        diff_mask = 1 - (normalized_fd < -self.relative_error).int()
        diff_mask[:, -1] = 1

        idx = repeat(
            torch.argmax(diff_mask, dim=1) + 1,
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
            mask = torch.tril(
                torch.ones(
                    self.n_slots, self.n_slots, device=x.device, dtype=torch.bool
                )
            )

            slots_list = []
            mask_list = []
            for i in range(self.n_slots):
                m = repeat(mask[i], "m -> b m", b=x.shape[0])
                for _ in range(self.n_iters):
                    slots = self.step(slots, k, v, mask=m)

                slots_list.append(slots)
                mask_list.append(m)
                slots = torch.cat((slots[:, : i + 1], init_slots[:, i + 1 :]), dim=1)  # type: ignore

            slots = torch.stack(slots_list, dim=1)
            mask = torch.stack(mask_list, dim=1)

            slots, mask = self.select_slots(slots, mask)

        if self.implicit:
            slots = slots.detach() - init_slots.detach() + init_slots  # type: ignore

        slots, attn_map = self.step(slots, k, v, return_attn=True, mask=mask)

        attn_map = rearrange(attn_map, "b n hw -> b hw n")

        return slots, attn_map, mask
