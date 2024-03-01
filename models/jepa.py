from abc import ABC, abstractmethod
from copy import deepcopy

import lightning as pl
import timm
import torch
from einops import rearrange, repeat
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
)
from torch import nn
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
        image_encoder_name,
        slot_attention: pl.LightningModule,
        feature_decoder: pl.LightningModule,
        dim: int,
        predictor_depth: int,
    ):

        super().__init__()

        self.image_encoder = (
            timm.create_model(
                image_encoder_name, pretrained=True, img_size=224, num_classes=0
            )
            .eval()
            .requires_grad_(False)
        )

        feature_dim = self.image_encoder.patch_embed.proj.out_channels

        self.slot_attention = slot_attention
        self.feature_decoder = feature_decoder
        self.dim = dim

        self.slot_pos_enc = PositionalEncoding1D(dim)

        self.norm = nn.LayerNorm(dim)

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

        self.project_down = nn.Linear(feature_dim, dim)
        self.project_up = nn.Linear(dim, feature_dim)

    def encode(self, images):
        """
        args:
            images: (b, t, 3, h, w)
            slots: (b, t, n, dim)
        returns:
            slots: (b, t, n, dim)
            features: (b, t, c, h, w)
        """

        _, t, _, _, _ = images.shape

        images = rearrange(images, "b t c h w -> (b t) c h w")
        features = self.image_encoder.forward_features(images)[:, 1:]
        features_down = self.project_down(features)
        features_flat = rearrange(features_down, "(b t) hw c -> b t hw c", t=t)

        slots = self.slot_attention(features_flat)

        return slots, features

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
        loss_fn,
    ):
        super().__init__()

        self.context_model = base_model
        self.target_model = deepcopy(self.context_model).requires_grad_(False)

        self.dim = dim
        self.ema_alpha = ema_alpha
        self.learning_rate = learning_rate

        self.loss_fn = loss_fn

        self.last_global_step = 0  # for logging

    def _update_target(self):
        for p, p_targ in zip(
            self.context_model.parameters(), self.target_model.parameters()
        ):
            if not p.requires_grad:
                continue
            p_targ.data.mul_(self.ema_alpha).add_(p.data, alpha=1 - self.ema_alpha)

    def training_step(self, x):

        context, _ = self.context_model.encode(x)
        pred = self.context_model.predict(context)
        pred_flat = rearrange(pred, "b t n d -> (b t) n d")
        pred_features = self.context_model.feature_decoder.forward_features(pred_flat)
        pred_features = rearrange(
            pred_features, "(b t) d h w -> b t d h w", b=x.shape[0]
        )
        _, target_features = self.target_model.encode(x)

        loss = self.loss_fn(pred_features[:, :-1], target_features[:, 1:])

        flattened_context = rearrange(context, "b t n d -> (b t n) d")
        mean_norm = torch.mean(torch.norm(flattened_context, dim=-1))
        mean_spread = torch.var(flattened_context, dim=0).mean()

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/mean_norm", mean_norm, sync_dist=True)
        self.log("train/mean_spread", mean_spread, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):
        context, _ = self.context_model.encode(x)
        pred = self.context_model.predict(context)
        pred_flat = rearrange(pred, "b t n d -> (b t) n d")
        pred_features = self.context_model.feature_decoder.forward_features(pred_flat)
        pred_features = rearrange(
            pred_features, "(b t) d h w -> b t d h w", b=x.shape[0]
        )
        _, target_features = self.target_model.encode(x)

        loss = self.loss_fn(pred_features[:, :-1], target_features[:, 1:])

        flattened_context = rearrange(context, "b t n d -> (b t n) d")
        mean_norm = torch.mean(torch.norm(flattened_context, dim=-1))
        mean_spread = torch.var(flattened_context, dim=0).mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mean_norm", mean_norm, sync_dist=True)
        self.log("val/mean_spread", mean_spread, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self._update_target()


class JEPAWrapper(pl.LightningModule):

    def __init__(
        self,
        base_model: JEPA,
        dim,
        learning_rate,
        loss_fn,
    ):
        super().__init__()

        self.context_model = base_model
        self.dim = dim
        self.learning_rate = learning_rate

        self.loss_fn = loss_fn

        self.last_global_step = 0  # for logging

    def training_step(self, x):
        context, features = self.context_model.encode(x)
        pred = self.context_model.predict(context)
        pred_flat = rearrange(pred, "b t n d -> (b t) n d")
        pred_features = self.context_model.feature_decoder(pred_flat)
        pred_features = rearrange(pred_features, "bt h w d -> bt (h w) d")
        pred_features = self.context_model.project_up(pred_features)

        loss = self.loss_fn(pred_features[:, :-1], features[:, 1:])

        flattened_context = rearrange(context, "b t n d -> (b t n) d")
        mean_norm = torch.mean(torch.norm(flattened_context, dim=-1))
        mean_spread = torch.var(flattened_context, dim=0).mean()

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/mean_norm", mean_norm, sync_dist=True)
        self.log("train/mean_spread", mean_spread, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):
        context, features = self.context_model.encode(x)
        pred = self.context_model.predict(context)
        pred_flat = rearrange(pred, "b t n d -> (b t) n d")
        pred_features = self.context_model.feature_decoder(pred_flat)
        pred_features = rearrange(pred_features, "bt h w d -> bt (h w) d")
        pred_features = self.context_model.project_up(pred_features)

        loss = self.loss_fn(pred_features[:, :-1], features[:, 1:])

        flattened_context = rearrange(context, "b t n d -> (b t n) d")
        mean_norm = torch.mean(torch.norm(flattened_context, dim=-1))
        mean_spread = torch.var(flattened_context, dim=0).mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mean_norm", mean_norm, sync_dist=True)
        self.log("val/mean_spread", mean_spread, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
