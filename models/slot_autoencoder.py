import lightning as pl
import timm
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize
from einops import rearrange, repeat
from torch.optim import AdamW


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

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        self.denormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def plot_attention(self, attn_map, x, palette="muted", alpha=0.4):
        attn_map = rearrange(
            attn_map, "(h w) n -> h w n", h=self.resolution[0] // self.patch_size
        )
        max_idx = attn_map.argmax(dim=-1, keepdim=True)

        attn_mask = (
            F.one_hot(max_idx.squeeze(-1), num_classes=attn_map.shape[-1])
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        attn_mask = F.interpolate(
            attn_mask, scale_factor=self.patch_size, mode="nearest"
        ).squeeze(0)

        palette = np.array(sns.color_palette(palette, attn_map.shape[-1]))
        colors = torch.tensor(palette[:, :3], dtype=torch.float32).to(x.device)
        attn_mask = repeat(attn_mask, "n h w -> n 1 h w")
        colors = repeat(colors, "n c -> n c 1 1")
        segmented_img = (colors * attn_mask).sum(dim=0)

        img = self.denormalize(x)
        overlayed_img = img * alpha + segmented_img * (1 - alpha)
        combined_img = torch.cat([img, overlayed_img], dim=2)

        return combined_img

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
        slots, attn_map_slots = self.slot_attention(features, n_slots)
        slots = self.project_slot(slots)
        decoded_features, attn_map_decoder = self.feature_decoder(slots)
        loss = self.loss_fn(decoded_features, features)

        slots = rearrange(slots, "b n d -> (b n) d")
        mean_norm = torch.mean(torch.norm(slots, dim=-1))
        mean_spread = torch.var(slots, dim=0).mean()

        return loss, attn_map_decoder, mean_norm, mean_spread

    def training_step(self, x):
        loss, _, mean_norm, mean_spread = self.common_step(x)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/mean_norm", mean_norm, sync_dist=True)
        self.log("train/mean_spread", mean_spread, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, x):
        loss, attn_map, mean_norm, mean_spread = self.common_step(x, val=True)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mean_norm", mean_norm, sync_dist=True)
        self.log("val/mean_spread", mean_spread, sync_dist=True)
        self.logger.log_image(
            key="attention", images=[self.plot_attention(attn_map[0], x[0])]
        )
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
