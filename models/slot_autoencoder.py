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
        feature_decoder_args: dict,
        dim: int,
        learning_rate: float,
        resolution: tuple[int, int],
        loss_fn,
        n_slots=[16, 8],
        single_decoder: bool = True,
        project_slots: bool = True,
        random_decoding: bool = True,
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

        self.project_slots = nn.Sequential(
            SwiGLUFFN(dim) if project_slots else nn.Identity(),
            nn.LayerNorm(dim),
        )

        self.feature_decoder = TransformerDecoder(**feature_decoder_args)

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)

        self.patch_size = self.image_encoder.patch_embed.patch_size[0]

        self.dim = dim
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.loss_fn = loss_fn
        self.n_slots = n_slots
        self.single_decoder = single_decoder
        self.random_decoding = random_decoding

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def common_step(self, x):

        features = self.forward_features(x)
        slots = features

        losses = {}
        attn_maps = []
        slots_dict = {}
        for n_slots, slot_attention in zip(self.n_slots, self.slot_attention):

            # Check if n_slots is a list
            if isinstance(n_slots, list):
                for n_slot in n_slots:
                    slots, attn_map = slot_attention(slots, n_slot)
                    if attn_maps:
                        attn_map = attn_maps[-1][0] @ attn_map
                    attn_maps.append((attn_map,))
                    slots_dict[n_slot] = slots
            else:
                slots, attn_map = slot_attention(slots, n_slots)
                if attn_maps:
                    attn_map = attn_maps[-1][0] @ attn_map  # Propagate attention
                attn_maps.append((attn_map,))
                slots_dict[n_slots] = slots

        if self.random_decoding:
            # Sample from
            slot_keys = list(slots_dict.keys())
            sampled_key = slot_keys[torch.randint(len(slot_keys), (1,))]
            slots = slots_dict[sampled_key]
            dec_slots = self.project_slots(slots)
            decoded_features, _ = self.feature_decoder(dec_slots)

            losses[sampled_key] = self.loss_fn(decoded_features, features)
        else:
            raise NotImplementedError

        return losses, attn_maps

    def training_step(self, x):
        losses, _ = self.common_step(x)

        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, sync_dist=True)

        loss = torch.stack(list(losses.values())).mean()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):
        losses, attn_maps = self.common_step(x)

        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, sync_dist=True)

        loss = torch.stack(list(losses.values())).mean()
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
