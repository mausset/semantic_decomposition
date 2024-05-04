import lightning as pl
import timm
import torch
from models.decoder import TransformerDecoder
from models.slot_attention import build_slot_attention
from torch import nn
from torch.optim import AdamW
from utils.plot import plot_attention_hierarchical

# from x_transformers import Encoder


class Interpreter(pl.LightningModule):

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
        optimizer: str = "adamw",
        optimizer_args: dict = {},
        weight_share: bool = False,
        detach_slots: bool = False,
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

        if weight_share:
            self.slot_attention = [
                build_slot_attention(slot_attention_arch, slot_attention_args)
            ] * len(n_slots)
            self.slot_decoder = [TransformerDecoder(**feature_decoder_args)] * len(
                n_slots
            )
        else:
            self.slot_attention = nn.ModuleList(
                [
                    build_slot_attention(slot_attention_arch, slot_attention_args)
                    for _ in n_slots
                ]
            )

            self.slot_decoder = nn.ModuleList(
                [TransformerDecoder(**feature_decoder_args) for _ in n_slots]
            )

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
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.weight_share = weight_share
        self.detach_slots = detach_slots

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def common_step(self, x):

        features = self.forward_features(x)
        slots = self.project_features(features)

        attn_list = []
        slots_list = [features]
        decoded_slots_list = []
        res = self.feature_resolution

        for n, slot_attention, slot_decoder in zip(
            self.n_slots, self.slot_attention, self.slot_decoder
        ):
            if n != self.n_slots[0]:
                slots = slots + slot_decoder.make_scaffold(slots, res)

            slots, attn_map = slot_attention(slots, n_slots=n)
            if attn_list:
                attn_map = attn_list[-1] @ attn_map
            attn_list.append(attn_map)

            decoded_slots, _ = slot_decoder(slots, res)
            if self.detach_slots:
                slots = slots.detach()
            slots_list.append(slots)
            decoded_slots_list.append(decoded_slots)

            res = (1, attn_map.shape[2])

        losses = {}

        for slot, decoded_slot, n in zip(slots_list, decoded_slots_list, self.n_slots):
            loss = self.loss_fn(decoded_slot, slot)
            losses[n] = loss

        return losses, attn_list

    def training_step(self, x):
        losses, _ = self.common_step(x)

        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, prog_bar=True, sync_dist=True)

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
