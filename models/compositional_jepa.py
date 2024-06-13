import copy

import lightning as pl
import timm
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from models.slot_attention import SA
from torch import nn
from torch.optim import AdamW
from utils.plot import plot_attention
from positional_encodings.torch_encodings import PositionalEncoding1D
from x_transformers import Encoder
from geomloss import SamplesLoss


class CompositionalJEPA(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        slot_attention_args: dict,
        predictor_args: dict,
        decoder_args: dict,
        dim: int,
        resolution: tuple[int, int],
        n_prototypes: int = 16,
        n_slots=8,
        sacrificial_patches: int = 0,
        sample_same=False,
        alpha: float = 0.996,
        optimizer: str = "adamw",
        optimizer_args: dict = {},
    ):
        super().__init__()

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

        self.positional_encoding = PositionalEncoding1D(dim)
        prototypes = self.positional_encoding(
            torch.empty(1, n_prototypes, dim)
        ).squeeze(0)

        self.student = nn.ModuleDict(
            {
                "slot_attention": SA(**slot_attention_args, n_slots=n_slots),
                "predictor": Encoder(**predictor_args),
                "decoder": Encoder(**decoder_args, cross_attend=True),
                "additional": nn.ParameterDict(
                    {"prototypes": nn.Parameter(prototypes).requires_grad_(False)}
                ),
            }
        )

        self.teacher = copy.deepcopy(self.student).requires_grad_(False)

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)
        self.patch_size = self.image_encoder.patch_embed.patch_size[0]
        self.resolution = resolution
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

        self.dim = dim
        self.alpha = alpha
        self.n_prototypes = n_prototypes
        self.n_slots = n_slots
        self.sacrificial_patches = sacrificial_patches
        self.sample_same = sample_same
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def gen_mask(self, t, n):
        mask = torch.triu(
            torch.ones(t * n, t * n, device=self.device), diagonal=1
        ).bool()
        for i in range(t):
            mask[i * n : (i + 1) * n, i * n : (i + 1) * n] = True

        return mask

    def update_teacher(self):
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(self.alpha).add_(s.detach().data * (1 - self.alpha))

    def get_pe(self, b, t, n, d):
        pe = self.positional_encoding(torch.zeros((b, t, d), device=self.device))

        return repeat(pe, "b t d -> b t n d", n=n)

    def common_step(self, x, stage="train"):
        b, t, *_ = x.shape

        x = rearrange(x, "b t ... -> (b t) ...")

        features = self.forward_features(x)

        sample = None
        if self.sample_same:
            sample = torch.randn(b, self.n_slots, self.dim, device=self.device)
            sample = repeat(sample, "b n d -> (b t) n d", t=t)

        slots, s_attn_map = self.student["slot_attention"](
            features, n_slots=self.n_slots, sample=sample
        )

        context = rearrange(slots, "(b t) n d -> b t n d", b=b)[:, :-1]

        context = context + self.get_pe(b, t - 1, self.n_slots, self.dim)
        context = rearrange(context, "b t n d -> b (t n) d", n=self.n_slots)

        mask = self.gen_mask(t - 1, self.n_slots)
        predictions = self.student["predictor"](context, attn_mask=mask)
        predictions = rearrange(predictions, "b (t n) d -> (b t) n d", n=self.n_slots)

        # s_prototypes = repeat(
        #     self.student["additional"]["prototypes"],
        #     "n d -> b n d",
        #     b=predictions.shape[0],
        # )

        # s_decode = self.student["decoder"](s_prototypes, context=predictions)

        target_slots, _ = self.teacher["slot_attention"](features, n_slots=self.n_slots)
        target = rearrange(target_slots, "(b t) n d -> b t n d", b=b)[:, 1:]
        target = rearrange(target, "b t n d -> (b t) n d")

        # t_prototypes = repeat(
        #     self.teacher["additional"]["prototypes"],
        #     "n d -> b n d",
        #     b=predictions.shape[0],
        # )
        # t_decode = self.teacher["decoder"](t_prototypes, context=target)

        tmp = rearrange(predictions, "bt n d -> (bt n) d")
        self.log(f"{stage}/mean_norm_student", torch.norm(tmp, dim=1).mean())
        self.log(f"{stage}/mean_var_student", torch.var(tmp, dim=0).mean())

        tmp = rearrange(target, "bt n d -> (bt n) d")
        self.log(f"{stage}/mean_norm_teacher", torch.norm(tmp, dim=1).mean())
        self.log(f"{stage}/mean_var_teacher", torch.var(tmp, dim=0).mean())

        loss = self.loss_fn(predictions, target.detach()).mean()

        return loss, s_attn_map

    def training_step(self, x):

        loss, *_ = self.common_step(x)
        self.log("train/loss", loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):

        loss, s_attn_map = self.common_step(x, stage="val")
        self.log("val/loss", loss.item(), prog_bar=True, sync_dist=True)

        s_attention_plot = plot_attention(
            x[0][0],
            s_attn_map[0],
            res=self.resolution[0],
            patch_size=self.patch_size,
        )

        if (
            isinstance(self.logger, pl.pytorch.loggers.WandbLogger)
            and self.trainer.global_rank == 0
        ):
            self.logger.log_image(key="attention_student", images=[s_attention_plot])

        return loss

    def optimizer_step(self, *args, **kwargs):
        self.update_teacher()
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), **self.optimizer_args)
        else:
            raise NotImplementedError
