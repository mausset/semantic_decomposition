import copy

import lightning as pl
import timm
import torch
import torch.nn.functional as F
from einops import rearrange
from geomloss import SamplesLoss
from models.slot_attention import SA
from torch import nn
from torch.optim import AdamW
from utils.plot import plot_attention
from positional_encodings.torch_encodings import PositionalEncoding1D
from x_transformers import Encoder


class CompositionalJEPA(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        slot_attention_args: dict,
        predictor_args: dict,
        dim: int,
        resolution: tuple[int, int],
        weighted: bool = False,
        alpha: float = 0.996,
        n_slots=8,
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

        self.slot_attention = SA(**slot_attention_args, n_slots=n_slots)
        self.teacher = copy.deepcopy(self.slot_attention).requires_grad_(False)

        self.predictor = Encoder(**predictor_args)

        if weighted:
            self.weighter = nn.Linear(dim, 1, bias=False)
            self.teacher_weighter = copy.deepcopy(self.weighter).requires_grad_(False)

        self.positional_encoding = PositionalEncoding1D(dim)

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)
        self.patch_size = self.image_encoder.patch_embed.patch_size[0]
        self.dim = dim
        self.resolution = resolution
        self.feature_resolution = (
            self.resolution[0] // self.patch_size,
            self.resolution[1] // self.patch_size,
        )
        self.loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05)

        self.weighted = weighted
        self.alpha = alpha
        self.n_slots = n_slots
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.log_img = True

    def update_teacher(self):
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.slot_attention.parameters()
        ):
            teacher_param.data = (
                teacher_param.data * self.alpha + student_param.data * (1 - self.alpha)
            )

        if self.weighted:
            for teacher_param, student_param in zip(
                self.teacher_weighter.parameters(), self.weighter.parameters()
            ):
                teacher_param.data = (
                    teacher_param.data * self.alpha
                    + student_param.data * (1 - self.alpha)
                )

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

    def common_step(self, x):
        _, t, *_ = x.shape

        x = rearrange(x, "b t ... -> (b t) ...")

        features = self.forward_features(x)

        slots, attn_map = self.slot_attention(features, n_slots=self.n_slots)
        slots = rearrange(slots, "(b t) n d -> b t n d", t=t)
        slots = slots[:, :-1]
        slots = rearrange(slots, "b t n d -> (b n) t d")

        pe = self.positional_encoding(slots)
        slots = slots + pe
        slots = rearrange(slots, "(b n) t d -> b (t n) d", n=self.n_slots)

        tmp = rearrange(slots, "b tn d -> (b tn) d")
        self.log("train/mean_norm", torch.norm(tmp, dim=1).mean())

        mask = self.gen_mask(t - 1, self.n_slots)

        predictions = self.predictor(slots, attn_mask=mask)
        predictions = rearrange(predictions, "b (t n) d -> (b t) n d", n=self.n_slots)

        targets, _ = self.teacher(features, n_slots=self.n_slots)
        targets = rearrange(targets, "(b t) n d -> b t n d", t=t)
        targets = targets[:, 1:]
        targets = rearrange(targets, "b t n d -> (b t) n d")

        if self.weighted:
            weights_pred = self.weighter(predictions)
            weights_target = self.teacher_weighter(targets)
            loss = self.loss_fn(
                predictions, targets, a=weights_pred, b=weights_target
            ).mean()
        else:
            loss = self.loss_fn(predictions, targets).mean()

        return loss, attn_map

    def training_step(self, x):

        loss, _ = self.common_step(x)

        self.log("train/loss", loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):

        loss, attn_map = self.common_step(x)

        self.log("val/loss", loss.item(), prog_bar=True, sync_dist=True)

        attention_plot = plot_attention(
            x[0][0],
            attn_map[0],
            res=self.resolution[0],
            patch_size=self.patch_size,
        )

        if (
            isinstance(self.logger, pl.pytorch.loggers.WandbLogger)
            and self.trainer.global_rank == 0
        ):
            self.logger.log_image(
                key="attention",
                images=[attention_plot],
            )

        return loss

    def optimizer_step(self, *args, **kwargs):
        self.update_teacher()
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), **self.optimizer_args)
        else:
            raise NotImplementedError
