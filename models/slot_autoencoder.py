import lightning as pl
import timm
import torch
from torch import nn
from torch.optim import AdamW

from utils.plot import plot_attention, plot_attention_hierarchical


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
        n_slots: int | list[int, int] = [16, 8],
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

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def common_step(self, x):

        slots_collection = []
        attn_maps = []
        losses = []

        features = self.forward_features(x)
        slots, attn_map_sa = self.slot_attention(
            features, self.n_slots[0], init_sample=True
        )
        slots_collection.append(slots)

        slots = self.project_slot(slots)
        decoded_features, attn_map_decoder = self.feature_decoder(slots)

        attn_maps.append((attn_map_sa, attn_map_decoder))
        losses.append(self.loss_fn(decoded_features, features))

        for n_slots in self.n_slots[1:]:
            slots, attn_map_sa = self.slot_attention(
                slots_collection[-1], n_slots, init_sample=True
            )

            slots_collection.append(slots)  # Slots saved before projection

            slots = self.project_slot(slots)
            decoded_features, attn_map_decoder = self.feature_decoder(slots)

            attn_map_sa = attn_maps[-1][0] @ attn_map_sa  # Propagate attention

            attn_maps.append((attn_map_sa, attn_map_decoder))
            losses.append(self.loss_fn(decoded_features, features))

        return losses, attn_maps

    def training_step(self, x):
        losses, _ = self.common_step(x)

        for i, loss in enumerate(losses):
            self.log(f"train/loss_{i}", loss, sync_dist=True)

        loss = torch.stack(losses).mean()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):
        losses, attn_maps = self.common_step(x)

        for i, loss in enumerate(losses):
            self.log(f"val/loss_{i}", loss, sync_dist=True)

        loss = torch.stack(losses).mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        self.logger.log_image(
            key="attention",
            images=[
                plot_attention_hierarchical(
                    x[0],
                    attn_maps,
                    res=self.resolution[0],
                    patch_size=self.patch_size,
                )
            ],
        )
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
