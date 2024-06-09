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

        self.student = nn.ModuleDict(
            {
                "slot_attention": SA(**slot_attention_args, n_slots=n_slots),
                "predictor": Encoder(**predictor_args),
                "prototypes": nn.ParameterList(  # SO goofy
                    [
                        nn.Parameter(
                            nn.init.kaiming_normal_(torch.empty(n_prototypes, dim))
                        )
                    ]
                ),
                "decoder": Encoder(**decoder_args, cross_attend=True),
            }
        )

        self.teacher = copy.deepcopy(self.student).requires_grad_(False)

        self.positional_encoding = PositionalEncoding1D(dim)

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)
        self.patch_size = self.image_encoder.patch_embed.patch_size[0]
        self.resolution = resolution
        self.loss_fn = nn.MSELoss()

        self.dim = dim
        self.alpha = alpha
        self.n_prototypes = n_prototypes
        self.n_slots = n_slots
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.log_img = True

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
            t.data = t.data * self.alpha + s.data * (1 - self.alpha)

    def get_pe(self, b, t, n, d):
        pe = self.positional_encoding(torch.zeros((b, t, d), device=self.device))

        return repeat(pe, "b t d -> b t n d", n=n)

    def common_step(self, x, stage="train"):
        b, t, *_ = x.shape

        x = rearrange(x, "b t ... -> (b t) ...")

        features = self.forward_features(x)
        s_features = rearrange(features, "(b t) ... -> b t ...", b=b)[:, :-1]
        s_features = rearrange(s_features, "b t ... -> (b t) ...")
        t_features = rearrange(features, "(b t) ... -> b t ...", t=t)[:, 1:]
        t_features = rearrange(t_features, "b t ... -> (b t) ...")

        slots, attn_map = self.student["slot_attention"](
            s_features, n_slots=self.n_slots
        )
        slots = rearrange(slots, "(b t) n d -> b t n d", b=b)
        slots = slots + self.get_pe(b, t - 1, self.n_slots, self.dim)
        slots = rearrange(slots, "b t n d -> b (t n) d", n=self.n_slots)

        mask = self.gen_mask(t - 1, self.n_slots)
        predictions = self.student["predictor"](slots, attn_mask=mask)
        predictions = rearrange(predictions, "b (t n) d -> (b t) n d", n=self.n_slots)

        targets, _ = self.teacher["slot_attention"](t_features, n_slots=self.n_slots)

        s_prototypes = repeat(
            self.student["prototypes"][0], "n d -> b n d", b=predictions.shape[0]
        )
        s_decode = self.student["decoder"](s_prototypes, context=predictions)

        t_prototypes = repeat(
            self.teacher["prototypes"][0], "n d -> b n d", b=targets.shape[0]
        )
        t_decode = self.teacher["decoder"](t_prototypes, context=targets)

        if stage == "train":
            tmp = rearrange(s_decode, "bt n d -> (bt n) d")
            self.log("train/mean_norm_student", torch.norm(tmp, dim=1).mean())
            self.log("train/mean_var_student", torch.var(tmp, dim=0).mean())

            tmp = rearrange(t_decode, "bt n d -> (bt n) d")
            self.log("train/mean_norm_teacher", torch.norm(tmp, dim=1).mean())
            self.log("train/mean_var_teacher", torch.var(tmp, dim=0).mean())

        # Expects predictions to be logits, not probabilities
        loss = self.loss_fn(s_decode, t_decode)

        return loss, attn_map

    def training_step(self, x):

        loss, _ = self.common_step(x)
        self.log("train/loss", loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):

        loss, attn_map = self.common_step(x, stage="val")
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
