from abc import ABC, abstractmethod
from copy import deepcopy

import lightning as pl
import torch
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
        backbone,
        dim: int,
        decoder_depth: int,
        predictor_depth: int,
        n_frames: int,
        max_slots: int,
        fixed_point_iterations: int,
    ):

        super().__init__()

        self.backbone = backbone

        self.dim = dim
        self.n_frames = n_frames
        self.max_slots = max_slots
        self.fixed_point_iterations = fixed_point_iterations

        self.slot_pos_enc = PositionalEncoding1D(dim)

        self.encoder = ContinuousTransformerWrapper(
            max_seq_len=max_slots,
            attn_layers=Encoder(
                dim=dim,
                depth=decoder_depth,
                abs_pos_emb=False,
                ff_glu=True,
                ff_swish=True,
                cross_attend=True,
            ),
        )

        self.predictor = ContinuousTransformerWrapper(
            max_seq_len=max_slots * n_frames,
            attn_layers=Encoder(
                dim=dim,
                depth=predictor_depth,
                abs_pos_emb=False,
                ff_glu=True,
                ff_swish=True,
            ),
        )

    def encode(self, images, slots):
        """
        args:
            images: (b, t, 3, h, w)
            slots: (b, t, n, dim)
        returns:
            slots: (b, t, n, dim)
        """

        images = rearrange(images, "b t c h w -> (b t) c h w")
        features = self.backbone(images)
        features = rearrange(features, "bt c h w -> bt (h w) c")

        slots = rearrange(slots, "b t n dim -> (b t) n dim")
        for _ in range(self.fixed_point_iterations - 1):
            slots = self.encoder(slots, context=features)
        self.encoder(slots.detach(), context=features)
        slots = rearrange(slots, "(b t) n dim -> b t n dim", t=self.n_frames)

        return slots

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
        attn_mask = self._create_attn_mask(slots)

        pos_enc = self.slot_pos_enc(slots[:, :, 0])
        pos_enc = repeat(pos_enc, "b t dim -> b (t n) dim", n=slots.shape[-2])
        slots = rearrange(slots, "b t n dim -> b (t n) dim")
        slots = slots + pos_enc

        slots = self.predictor(slots, attn_mask=attn_mask)
        slots = rearrange(slots, "b (t n) dim -> b t n dim", t=self.n_frames)

        return slots


class EMAJEPA(pl.LightningModule):

    def __init__(
        self,
        base_model: JEPA,
        dim,
        n_frames,
        ema_alpha,
        max_slots,
        min_slots,
        learning_rate,
        decoder=None,
        decoder_detach=True,
    ):
        super().__init__()

        self.context_model = base_model
        self.target_model = deepcopy(self.context_model).requires_grad_(False)

        self.dim = dim
        self.n_frames = n_frames
        self.ema_alpha = ema_alpha
        self.max_slots = max_slots
        self.min_slots = min_slots
        self.learning_rate = learning_rate

        self.decoder = decoder
        self.decoder_detach = decoder_detach

        self.last_global_step = 0

    def _update_target(self):
        for p, p_targ in zip(
            self.context_model.parameters(), self.target_model.parameters()
        ):
            if not p.requires_grad:
                continue
            p_targ.data.mul_(self.ema_alpha).add_(p.data, alpha=1 - self.ema_alpha)

    def training_step(self, x):
        n_slots = torch.randint(self.min_slots, self.max_slots, (1,)).item()
        slots = torch.normal(
            mean=0, std=1, size=(x.shape[0], n_slots, self.dim), device=x.device
        )
        slots = repeat(slots, "b n d -> b t n d", t=self.n_frames)

        context = self.context_model.encode(x, slots)
        pred = self.context_model.predict(context)
        target = self.target_model.encode(x, slots)

        loss = F.mse_loss(pred[:, :-1], target[:, 1:])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        if self.decoder is not None:
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
        n_slots = torch.randint(self.min_slots, self.max_slots, (1,)).item()
        slots = torch.normal(
            mean=0, std=1, size=(x.shape[0], n_slots, self.dim), device=x.device
        )
        slots = repeat(slots, "b n d -> b t n d", t=self.n_frames)

        context = self.context_model.encode(x, slots)
        pred = self.context_model.predict(context)
        target = self.target_model.encode(x, slots)

        loss = F.mse_loss(pred[:, :-1], target[:, 1:])
        self.log("val/loss", loss, sync_dist=True)

        if self.decoder is not None:
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
