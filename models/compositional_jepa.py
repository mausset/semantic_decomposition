import lightning as pl
import timm
from timm.layers.config import set_fused_attn
import wandb
import torch
from einops import rearrange, repeat
from geomloss import SamplesLoss
from matplotlib import pyplot as plt
from models.decoder import TransformerDecoder
from models.slot_attention import SA
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn
from torch.optim import AdamW
from utils.helpers import block_causal_mask
from utils.plot import plot_attention_interpreter
from x_transformers import Encoder


class CompositionalJEPA(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        slot_attention_args: dict,
        encoder_args: dict,
        predictor_args: dict,
        decoder_args: dict,
        loss_args: dict,
        n_levels: int,
        shrink_factor: int,
        dim: int,
        resolution: tuple[int, int],
        level_schedule: dict = {},
        n_slots: list[int] = [16, 8],
        consistent_noise: bool = False,
        optimizer: str = "adamw",
        optimizer_args: dict = {},
    ):
        super().__init__()

        set_fused_attn(True)
        self.image_encoder = (
            timm.create_model(
                image_encoder_name,
                pretrained=True,
                img_size=resolution,
                num_classes=0,
            )
            .eval()
            .requires_grad_(False)
        )

        self.time_pe = PositionalEncoding1D(dim)

        self.hierarchy = nn.ModuleList([])
        for i in range(n_levels):
            d = {
                "slot_attention": SA(**slot_attention_args, n_slots=n_slots[i]),
                "predictor": Encoder(**predictor_args),
                "decoder": TransformerDecoder(decoder_args),
            }

            if i == 0:
                pass
            else:
                d["encoder"] = Encoder(**encoder_args)

            self.hierarchy.append(nn.ModuleDict(d))

        self.loss_fn = SamplesLoss(**loss_args)

        self.level_schedule = level_schedule
        self.do_decode = [True]
        self.level_stage = 0

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)
        self.patch_size = self.image_encoder.patch_embed.patch_size[0]
        self.resolution = resolution

        self.decode_res = {}
        for idx in range(n_levels):
            if idx == 0:
                self.decode_res[idx] = (
                    resolution[0] // self.patch_size,
                    resolution[1] // self.patch_size,
                )
            else:
                self.decode_res[idx] = (shrink_factor, n_slots[idx])

        self.n_levels = n_levels
        self.shrink_factor = shrink_factor
        self.dim = dim
        self.n_slots = n_slots
        self.consistent_noise = consistent_noise
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def get_time_pe(self, b, t, n, d):
        pe = self.time_pe(torch.zeros((b, t, d), device=self.device))

        return repeat(pe, "b t d -> b t n d", n=n)

    def forward_hierarchy(self, x, stage="train"):
        b, *_ = x.shape

        if self.level_stage + 1 < self.n_levels:
            if (
                self.global_step
                > self.level_schedule[self.level_stage + 1]["start_step"]
            ):
                self.do_decode[self.level_stage] = False
                self.do_decode.append(True)
                self.hierarchy[self.level_stage].requires_grad_(False)
                self.level_stage += 1

        x = x[:, : self.level_schedule[self.level_stage]["t"]]  # Truncate

        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.forward_features(x)
        x = rearrange(x, "(b t) ... -> b t ...", b=b)

        attn_maps = []
        losses = {}
        for idx, (level, decode) in enumerate(zip(self.hierarchy, self.do_decode)):

            x, loss, attn_map = self.forward_level(x, level, idx, decode=decode)

            if loss is not None:
                losses[idx] = loss
            attn_maps.append(attn_map)

            x = x.detach()

            flat = rearrange(x, "b t n d -> (b t n) d")
            self.log(f"{stage}/mean_norm_{idx}", torch.norm(flat, dim=1).mean())
            self.log(f"{stage}/mean_var_{idx}", torch.var(flat, dim=0).mean())

        return losses, attn_maps

    def forward_level(self, x, level, idx, decode=True):
        b, t, *_ = x.shape

        n_slots = self.n_slots[idx]

        context = x
        target = x
        if idx != 0:
            slack = t % self.shrink_factor
            x = x[:, : t - slack]

            x = rearrange(x, "b (t sf) ... -> (b t) sf ...", sf=self.shrink_factor)
            x = x + self.get_time_pe(*x.shape)
            t = t // self.shrink_factor
            x = rearrange(x, "bt sf n ... -> bt (sf n) ...")

            context = level["encoder"](x)
            target = x

            context = rearrange(context, "(b t) ... -> b t ...", b=b)
            target = rearrange(target, "(b t) ... -> b t ...", b=b)

        sample = None
        if self.consistent_noise:
            sample = torch.randn(b, n_slots, self.dim, device=self.device)
            sample = repeat(sample, "b n d -> (b t) n d", t=t)

        context = rearrange(context, "b t ... -> (b t) ...")
        slots, attn_map = level["slot_attention"](
            context, n_slots=n_slots, sample=sample
        )

        slots = rearrange(slots, "(b t) n d -> b t n d", t=t)
        if not decode:
            return slots, None, attn_map

        slots_te = slots[:, :-1] + self.get_time_pe(b, t - 1, n_slots, self.dim)
        slots_te = rearrange(slots_te, "b t n d -> b (t n) d")

        mask = block_causal_mask(t - 1, n_slots, device=self.device)
        predictions = level["predictor"](slots_te, attn_mask=mask)
        predictions = rearrange(predictions, "b (t n) d -> (b t) n d", t=t - 1)

        pred_dec = level["decoder"](predictions, resolution=self.decode_res[idx])

        target = rearrange(target[:, 1:], "b t n d -> (b t) n d")
        loss = self.loss_fn(pred_dec, target.detach()).mean()

        return slots, loss, attn_map

    def training_step(self, x):

        losses, *_ = self.forward_hierarchy(x)

        for k, v in losses.items():
            self.log(f"train/loss_{k}", v.item(), prog_bar=True, sync_dist=True)

        loss = torch.stack(list(losses.values())).sum()

        return loss

    def validation_step(self, x):

        losses, attn_maps = self.forward_hierarchy(x, stage="val")

        for k, v in losses.items():
            self.log(f"val/loss_{k}", v.item(), sync_dist=True)

        attn_plot = self.plot_hierarchy(x, attn_maps) * 255
        if (
            isinstance(self.logger, pl.pytorch.loggers.WandbLogger)  # type: ignore
            and self.trainer.global_rank == 0
        ):
            self.logger.experiment.log(  # type: ignore
                {"attention": wandb.Video(attn_plot.cpu().numpy(), fps=8, format="gif")}
            )

        loss = torch.stack(list(losses.values())).sum()

        return loss

    def plot_hierarchy(self, x, attn_maps):

        attn_hierarchy = []
        for i in range(len(attn_maps)):
            t = min(
                self.shrink_factor ** (self.n_levels - i - 1),
                self.level_schedule[self.level_stage]["t"],
            )
            attn_hierarchy.append(attn_maps[i][:t])
        attn_hierarchy = attn_hierarchy[::-1]

        propagated_attn = []
        for i in range(len(attn_hierarchy)):

            a = attn_hierarchy[i]

            for a_ in attn_hierarchy[i + 1 :]:
                a = rearrange(a, "b (s n) m -> (b s) n m", s=self.shrink_factor)
                a = torch.bmm(a_, a)

            propagated_attn.append(a)

        attn_plots = []
        for p_attn in reversed(propagated_attn):
            t = min(
                self.shrink_factor ** (self.n_levels - 1),
                self.level_schedule[self.level_stage]["t"],
            )
            attn_plots.append(
                plot_attention_interpreter(
                    x[0][:t],
                    p_attn,
                    res=self.resolution[0],
                    patch_size=self.patch_size,
                )
            )

        attn_plots = torch.cat(attn_plots, dim=-1)

        return attn_plots

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), **self.optimizer_args)
        else:
            raise NotImplementedError
