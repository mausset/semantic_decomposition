import lightning as pl
import timm
import torch
from torch import nn
from torch.optim import AdamW

from utils.plot import plot_attention


class SlotAE(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        slot_attention: pl.LightningModule,
        feature_decoder: pl.LightningModule,
        dim: int,
        learning_rate: float,
        resolution: tuple[int, int],
        loss_fn,
        n_slots: int | tuple[int, int] = 8,
        n_slots_val: int | tuple[int, int] = (3, 9),
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
        self.slot_attention = slot_attention
        self.project_slot = nn.Sequential(
            nn.Linear(slot_attention.slot_dim, dim, bias=False),
            nn.LayerNorm(dim),
        )
        self.feature_decoder = feature_decoder

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)

        self.patch_size = self.image_encoder.patch_embed.patch_size[0]

        self.dim = dim
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.loss_fn = loss_fn
        self.n_slots = n_slots
        self.n_slots_val = n_slots_val

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def common_step(self, x, val=False):
        slot_range = self.n_slots_val if val else self.n_slots
        n_slots = (
            torch.randint(*slot_range, (1,)).item()
            if isinstance(slot_range, tuple)
            else slot_range
        )

        features = self.forward_features(x)
        slots, attn_map_sa = self.slot_attention(features, n_slots)
        slots = self.project_slot(slots)
        decoded_features, attn_map_decoder = self.feature_decoder(slots)
        loss = self.loss_fn(decoded_features, features)

        return loss, attn_map_sa, attn_map_decoder

    def training_step(self, x):
        loss, _, _ = self.common_step(x)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, x):
        loss, attn_map_sa, attn_map_decoder = self.common_step(x, val=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.logger.log_image(
            key="attention",
            images=[
                plot_attention(
                    x[0],
                    [attn_map_sa[0], attn_map_decoder[0]],
                    res=self.resolution[0],
                    patch_size=self.patch_size,
                )
            ],
        )
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
