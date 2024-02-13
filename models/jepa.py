from abc import ABC, abstractmethod
from copy import deepcopy

import lightning as pl
import torch
from torch import nn
from einops import rearrange, repeat
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
)
from torch.nn import functional as F
from torch.optim import AdamW
from x_transformers import ContinuousTransformerWrapper, Encoder


class JEPA(ABC, pl.LightningModule):

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class SlotJEPA(JEPA):

    def __init__(
        self,
        image_encoder: pl.LightningModule,
        slot_attention: pl.LightningModule,
        dim: int,
        predictor_depth: int,
    ):

        super().__init__()

        self.image_encoder = image_encoder
        self.slot_attention = slot_attention
        self.dim = dim

        self.slot_pos_enc = PositionalEncoding1D(dim)

        self.norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.predictor = ContinuousTransformerWrapper(
            max_seq_len=None,
            attn_layers=Encoder(
                dim=dim,
                depth=predictor_depth,
                ff_glu=True,
                ff_swish=True,
            ),
            use_abs_pos_emb=False,
        )

    def encode(self, images, init_slots=None):
        """
        args:
            images: (b, t, 3, h, w)
            slots: (b, t, n, dim)
        returns:
            slots: (b, t, n, dim)
        """

        _, t, _, _, _ = images.shape

        images = rearrange(images, "b t c h w -> (b t) c h w")
        features = self.image_encoder(images)
        features = rearrange(features, "(b t) c h w -> b t (h w) c", t=t)
        features = self.ffn(self.norm(features))

        slots, init_slots = self.slot_attention(features, init_slots=init_slots)

        return slots, init_slots

    def _create_attn_mask(self, slots):
        _, t, n, _ = slots.shape

        mask = torch.ones(t * n, t * n, device=self.device).triu(1).bool()

        for i in range(t):
            mask[n * i : n * i + n, n * i : n * i + n] = False

        return mask

    def predict(self, slots):
        """
        args:
            slots: (b, t, n, dim)
        returns:
            slots: (b, t, n, dim)
        """
        _, _, n, _ = slots.shape

        attn_mask = self._create_attn_mask(slots)

        pos_enc = self.slot_pos_enc(slots[:, :, 0])
        pos_enc = repeat(pos_enc, "b t dim -> b (t n) dim", n=n)
        slots = rearrange(slots, "b t n dim -> b (t n) dim")
        slots = slots + pos_enc

        slots = self.predictor(slots, attn_mask=attn_mask)
        slots = rearrange(slots, "b (t n) dim -> b t n dim", n=n)

        return slots


class EMAJEPA(pl.LightningModule):

    def __init__(
        self,
        base_model: JEPA,
        dim,
        ema_alpha,
        learning_rate,
        decoder=None,
        decoder_detach=True,
        decoder_start_epoch=0,
    ):
        super().__init__()

        self.context_model = base_model
        self.target_model = deepcopy(self.context_model).requires_grad_(False)

        self.dim = dim
        self.ema_alpha = ema_alpha
        self.learning_rate = learning_rate

        self.decoder = decoder
        self.decoder_detach = decoder_detach
        self.decoder_start_epoch = decoder_start_epoch

        self.last_global_step = 0  # for logging

    def _update_target(self):
        for p, p_targ in zip(
            self.context_model.parameters(), self.target_model.parameters()
        ):
            if not p.requires_grad:
                continue
            p_targ.data.mul_(self.ema_alpha).add_(p.data, alpha=1 - self.ema_alpha)

    def training_step(self, x):

        context, init_slots = self.context_model.encode(x)
        pred = self.context_model.predict(context)
        target, _ = self.target_model.encode(x, init_slots)

        loss = F.mse_loss(pred[:, :-1], target[:, 1:])
        mean_norm = torch.mean(torch.norm(context, dim=-1))
        mean_spread = (torch.var(context, dim=-1) / mean_norm).mean()

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/mean_norm", mean_norm, sync_dist=True)
        self.log("train/mean_spread", mean_spread, sync_dist=True)

        if self.decoder is None or self.current_epoch < self.decoder_start_epoch:
            return loss

        x = rearrange(x, "b t c h w -> (b t) c h w")
        target = rearrange(target, "b t n d -> (b t) n d")

        if self.decoder_detach:
            target = target.detach()
        decoded_image = self.decoder(target)

        reconstruction_loss = F.mse_loss(decoded_image, x)
        self.log(
            "train/reconstruction_loss",
            reconstruction_loss,
            prog_bar=True,
            sync_dist=True,
        )
        sample = torch.cat((x[0], decoded_image[0]), dim=2)
        if (self.global_step + 1) % (
            self.trainer.log_every_n_steps
        ) == 0 and self.global_step != self.last_global_step:
            self.last_global_step = self.global_step
            self.logger.log_image(key="train/sample", images=[sample.clip(0, 1)])

            decoded_components = self.decoder.forward_components(target)[0]
            component_img = rearrange(decoded_components, "t c h w -> c h (t w)")
            self.logger.log_image(
                key="train/decoded_components", images=[component_img.clip(0, 1)]
            )

        loss += reconstruction_loss

        return loss

    def validation_step(self, x):

        context, init_slots = self.context_model.encode(x)
        pred = self.context_model.predict(context)
        target, _ = self.target_model.encode(x, init_slots)

        loss = F.mse_loss(pred[:, :-1], target[:, 1:])
        mean_norm = torch.mean(torch.norm(context, dim=-1))
        mean_spread = (torch.var(context, dim=-1) / mean_norm).mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mean_norm", mean_norm, sync_dist=True)
        self.log("val/mean_spread", mean_spread, sync_dist=True)

        if self.decoder is None or self.current_epoch < self.decoder_start_epoch:
            return

        x = rearrange(x, "b t c h w -> (b t) c h w")
        target = rearrange(target, "b t n d -> (b t) n d")

        if self.decoder_detach:
            target = target.detach()
        decoded_image = self.decoder(target)

        reconstruction_loss = F.mse_loss(decoded_image, x)
        self.log("val/reconstruction_loss", reconstruction_loss, sync_dist=True)

        sample = torch.cat((x[0], decoded_image[0]), dim=2)
        self.logger.log_image(key="val/sample", images=[sample.clip(0, 1)])

        decoded_components = self.decoder.forward_components(target)[0]

        component_img = rearrange(decoded_components, "t c h w -> c h (t w)")

        self.logger.log_image(
            key="val/decoded_components", images=[component_img.clip(0, 1)]
        )

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self._update_target()
