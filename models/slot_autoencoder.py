import lightning as pl
import timm
import torch
from models.decoder import TransformerDecoder
from models.slot_attention import build_slot_attention
from models.components import SwiGLUFFN
from torch import nn
from torch.optim import AdamW
from utils.plot import plot_attention_hierarchical


class SlotAE(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        slot_attention_arch: str,
        slot_attention_args: dict,
        detach_slots: bool,
        feature_decoder_args: dict,
        dim: int,
        learning_rate: float,
        resolution: tuple[int, int],
        loss_fn,
        n_slots: int | list[int, int] = [16, 8],
        single_decoder: bool = False,
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

        self.slot_attention = nn.ModuleList(
            [
                build_slot_attention(slot_attention_arch, slot_attention_args)
                for _ in n_slots
            ]
        )

        self.project_slots = nn.ModuleList(
            [
                nn.Sequential(
                    # nn.Linear(slot_attention_args["slot_dim"], dim, bias=False),
                    SwiGLUFFN(dim),
                    nn.LayerNorm(dim),
                )
                for _ in n_slots
            ]
        )

        self.feature_decoder = nn.ModuleList(
            [
                TransformerDecoder(**feature_decoder_args)
                for _ in range(1 if single_decoder else len(n_slots))
            ]
        )

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)

        self.patch_size = self.image_encoder.patch_embed.patch_size[0]

        self.dim = dim
        self.learning_rate = learning_rate
        self.detach_slots = detach_slots
        self.resolution = resolution
        self.loss_fn = loss_fn
        self.n_slots = n_slots
        self.single_decoder = single_decoder

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def common_step(self, x):

        features = self.forward_features(x)
        slots = features

        losses = []
        attn_maps = []
        for i, (n_slots, slot_attention, project_slots) in enumerate(
            zip(self.n_slots, self.slot_attention, self.project_slots)
        ):

            slots, attn_map = slot_attention(slots, n_slots)

            if attn_maps:
                attn_map = attn_maps[-1][0] @ attn_map  # Propagate attention

            attn_maps.append((attn_map,))

            dec_slots = project_slots(slots)
            if self.single_decoder:
                decoded_features, _ = self.feature_decoder[0](dec_slots)
            else:
                decoded_features, _ = self.feature_decoder[i](dec_slots)

            losses.append(self.loss_fn(decoded_features, features))

            if self.detach_slots:
                slots = slots.detach()

        return losses, attn_maps

    def training_step(self, x):
        losses, _ = self.common_step(x)

        for loss, n_slots in zip(losses, self.n_slots):
            self.log(f"train/loss_{n_slots}", loss, sync_dist=True)

        loss = torch.stack(losses).mean()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):
        losses, attn_maps = self.common_step(x)

        for loss, n_slots in zip(losses, self.n_slots):
            self.log(f"val/loss_{n_slots}", loss, sync_dist=True)

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
