import lightning as pl
import timm
import torch
from models.decoder import TransformerDecoder
from models.slot_attention import build_slot_attention
from torch import nn
from torch.optim import AdamW
from utils.plot import plot_attention_hierarchical
from utils.helpers import compute_combined_attn, pad_batched_slots

from x_transformers import Encoder


class SlotAE(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        slot_attention_arch: str,
        slot_attention_args: dict,
        feature_decoder_args: dict,
        dim: int,
        resolution: tuple[int, int],
        loss_fn,
        n_slots=[16, 8],
        ignore_decode_slots=[],
        slot_encoder=4,
        decode_strategy: str = "random",
        mode: str = "hierarchical",
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

        self.project_features = nn.Sequential(
            nn.Linear(dim, slot_attention_args["input_dim"]),
            nn.LayerNorm(slot_attention_args["input_dim"]),
        )

        self.slot_attention = nn.ModuleList(
            [
                build_slot_attention(slot_attention_arch, slot_attention_args)
                for _ in n_slots
            ]
        )

        self.slot_encoder = (
            Encoder(
                dim=slot_attention_args["slot_dim"],
                depth=slot_encoder,
                ff_glu=True,
                ff_swish=True,
            )
            if slot_encoder
            else nn.Identity()
        )

        self.project_slots = nn.Sequential(
            nn.Linear(slot_attention_args["slot_dim"], dim),
            nn.LayerNorm(dim),
        )

        self.feature_decoder = TransformerDecoder(**feature_decoder_args)

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)

        self.patch_size = self.image_encoder.patch_embed.patch_size[0]

        self.dim = dim
        self.resolution = resolution
        self.feature_resolution = (
            self.resolution[0] // self.patch_size,
            self.resolution[1] // self.patch_size,
        )
        self.loss_fn = loss_fn
        self.n_slots = n_slots
        self.ignore_decode_slots = ignore_decode_slots
        self.decode_strategy = decode_strategy
        self.mode = mode
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def common_step(self, x):

        features = self.forward_features(x)
        slots = self.project_features(features)

        attn_list = []
        slots_list = [features]
        slots_dict = {}
        for n_slots, slot_attention in zip(self.n_slots, self.slot_attention):
            if isinstance(n_slots, int):
                n_slots = [n_slots]

            for n in n_slots:

                match self.mode:
                    case "hierarchical":
                        slots, attn_map = slot_attention(slots, n_slots=n)
                        attn_map = attn_map[0]
                        if attn_list:
                            attn_map = attn_list[-1] @ attn_map
                    case "multi_scale":
                        slots, attn_maps = slot_attention(slots_list, n_slots=n)
                        slots_list.append(slots)
                        attn_map = compute_combined_attn(attn_list, attn_maps)
                    case "propagate":
                        prev_attn = attn_list[-1] if attn_list else None
                        slots, attn_map = slot_attention(
                            features, slots, prev_attn, n_slots=n
                        )
                    case "flat":
                        slots = features
                        slots, attn_map = slot_attention(slots, n_slots=n)
                        attn_map = attn_map[0]

                slots = self.slot_encoder(slots)
                if n not in self.ignore_decode_slots:
                    slots_dict[n] = self.project_slots(slots)
                attn_list.append(attn_map)

        losses = {}
        match self.decode_strategy:
            case "random":
                slot_keys = list(slots_dict.keys())
                sampled_key = slot_keys[torch.randint(len(slot_keys), (1,))]
                slots = slots_dict[sampled_key]
                decoded_features, _ = self.feature_decoder(
                    slots, self.feature_resolution
                )

                losses[sampled_key] = self.loss_fn(decoded_features, features)

            case "all":
                padded_slots, padded_mask = pad_batched_slots(slots_dict)

                decoded_features, _ = self.feature_decoder(
                    padded_slots, self.feature_resolution, context_mask=padded_mask
                )

                chunked_decoded_features = torch.chunk(
                    decoded_features, len(slots_dict), dim=0
                )

                for slot_key, chunk in zip(slots_dict.keys(), chunked_decoded_features):
                    losses[slot_key] = self.loss_fn(chunk, features)
            case _:
                raise NotImplementedError

        return losses, attn_list

    def training_step(self, x):
        losses, _ = self.common_step(x)

        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, sync_dist=True)

        loss = torch.stack(list(losses.values())).mean()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):
        losses, attn_list = self.common_step(x)

        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, sync_dist=True)

        loss = torch.stack(list(losses.values())).mean()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        attention_plot = plot_attention_hierarchical(
            x[0],
            attn_list,
            res=self.resolution[0],
            patch_size=self.patch_size,
        )

        if isinstance(self.logger, pl.pytorch.loggers.WandbLogger):
            self.logger.log_image(
                key="attention",
                images=[attention_plot],
            )

        return loss

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), **self.optimizer_args)
        else:
            raise NotImplementedError
