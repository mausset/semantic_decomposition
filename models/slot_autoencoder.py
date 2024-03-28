import lightning as pl
import timm
import torch
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
        self.feature_decoder = feature_decoder

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)

        self.patch_size = self.image_encoder.patch_embed.patch_size[0]

        self.dim = dim
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.loss_fn = loss_fn

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        self.denormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def plot_attention(self, attn_map, x):
        max_idx = attn_map.argmax(dim=-1, keepdim=True)
        attn_mask = torch.zeros_like(attn_map).scatter_(-1, max_idx, 1)

        h, w = (
            self.resolution[0] // self.patch_size,
            self.resolution[1] // self.patch_size,
        )
        attn_mask = repeat(
            attn_mask,
            "(h w) n -> c (h hr) (n w wr)",
            h=h,
            hr=self.patch_size,
            w=w,
            wr=self.patch_size,
            c=3,
        )
        img = self.denormalize(x)
        img_repeat = repeat(img, "c h w -> c h (n w)", n=attn_map.shape[-1])
        masked = img_repeat * attn_mask
        img = torch.cat([img, masked], dim=2)
        return img

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def common_step(self, x):
        features = self.forward_features(x)
        slots, attn_map_slots = self.slot_attention(features)
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
        loss, attn_map, mean_norm, mean_spread = self.common_step(x)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mean_norm", mean_norm, sync_dist=True)
        self.log("val/mean_spread", mean_spread, sync_dist=True)
        self.logger.log_image(
            key="attention", images=[self.plot_attention(attn_map[0], x[0])]
        )
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
