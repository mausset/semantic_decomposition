import lightning as pl
import timm
import torch
from models.components import SwiGLUFFN
from models.decoder import TransformerDecoder
from models.slot_attention import build_slot_attention
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import RobertaModel, RobertaTokenizer
from utils.plot import plot_alignment

from x_transformers import Encoder


class Align(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name: str,
        slot_attention_arch: str,
        slot_attention_args: dict,
        feature_decoder_args: dict,
        dim: int,
        resolution: tuple[int, int],
        n_slots=7,
        n_img_slots=32,
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

        self.image_slot_attention = build_slot_attention(
            slot_attention_arch, slot_attention_args
        )

        self.text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.text_encoder = (
            RobertaModel.from_pretrained("roberta-base").eval().requires_grad_(False)
        )

        self.project_features_img = nn.Sequential(
            SwiGLUFFN(dim),
            nn.LayerNorm(dim),
        )

        self.project_features_txt = nn.Sequential(
            SwiGLUFFN(dim),
            nn.LayerNorm(dim),
        )

        self.slot_attention = build_slot_attention(
            slot_attention_arch, slot_attention_args
        )

        self.slot_encoder = Encoder(
            dim=dim,
            depth=4,
            ff_glu=True,
            ff_swish=True,
        )

        self.project_slots_img = nn.Sequential(
            SwiGLUFFN(dim),
            nn.LayerNorm(dim),
        )

        self.project_slots_txt = nn.Sequential(
            SwiGLUFFN(dim),
            nn.LayerNorm(dim),
        )

        self.feature_decoder_img = TransformerDecoder(**feature_decoder_args)
        self.feature_decoder_txt = TransformerDecoder(**feature_decoder_args)

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)
        self.patch_size = self.image_encoder.patch_embed.patch_size[0]

        self.dim = dim
        self.resolution = resolution
        self.img_feature_resolution = (
            self.resolution[0] // self.patch_size,
            self.resolution[1] // self.patch_size,
        )
        self.loss_fn = F.mse_loss
        self.n_slots = n_slots
        self.n_img_slots = n_img_slots
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    # Shorthand
    def forward_features_img(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def forward_features_txt(self, x):
        tokens = self.text_tokenizer(
            x, return_tensors="pt", padding=True, return_attention_mask=True
        )
        for k in tokens:
            tokens[k] = tokens[k].to(self.device)
        return (
            self.text_encoder(**tokens).last_hidden_state[:, 1:],
            tokens.attention_mask[:, 1:].bool(),
            tokens.input_ids[:, 1:],
        )

    def common_step(self, x):
        img, txt = x

        features_img = self.forward_features_img(img)

        slots_img, attn_map_img = self.image_slot_attention(
            features_img, n_slots=self.n_img_slots
        )

        mask_img = torch.ones_like(slots_img[:, :, 0]).bool()

        features_proj_img = self.project_features_img(slots_img)

        features_txt, mask_txt, token_ids = self.forward_features_txt(txt)
        features_proj_txt = self.project_features_txt(features_txt)

        features = torch.cat([features_proj_img, features_proj_txt], dim=1)
        mask = torch.cat([mask_img, mask_txt], dim=1)

        encoded_features = self.slot_encoder(features, mask=mask)

        slots, attn_map = self.slot_attention(
            encoded_features, n_slots=self.n_slots, mask=mask
        )

        slots_img = self.project_slots_img(slots)
        slots_txt = self.project_slots_txt(slots)

        recon_features_img, _ = self.feature_decoder_img(
            slots_img,
            self.img_feature_resolution,
        )

        text_resolution = (1, features_txt.size(1))
        recon_features_txt, _ = self.feature_decoder_txt(
            slots_txt, text_resolution, mask=mask_txt
        )

        recon_features_txt = recon_features_txt * mask_txt.unsqueeze(-1)
        features_txt = features_txt * mask_txt.unsqueeze(-1)

        loss_txt = (
            self.loss_fn(recon_features_txt, features_txt, reduction="none")
            .mean(-1)
            .sum(dim=-1)
            / mask_txt.sum(-1)
        ).mean()

        losses = {
            "img": self.loss_fn(recon_features_img, features_img),
            "txt": loss_txt,
        }

        attn_map_img = attn_map_img @ attn_map[:, : features_proj_img.size(1)]
        attn_map_txt = attn_map[:, features_proj_img.size(1) :]

        tokens_txt = [
            self.text_tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in token_ids
        ]

        tokens_txt = [
            [t.replace("Ġ", "") for t in tokens if t != "<pad>"]
            for tokens in tokens_txt
        ]

        info = [
            {"img": i, "img_attn": attn_img, "txt": t, "txt_attn": attn_txt[m]}
            for i, attn_img, t, attn_txt, m in zip(
                img, attn_map_img, tokens_txt, attn_map_txt, mask_txt
            )
        ]

        return losses, info

    def training_step(self, x):
        if self.optimizer == "adamw_schedulefree":
            self.optimizers().train()

        losses, _ = self.common_step(x)

        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, prog_bar=True, sync_dist=True)

        loss = torch.stack(list(losses.values())).sum()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, x):
        if self.optimizer == "adamw_schedulefree":
            self.optimizers().eval()

        losses, info = self.common_step(x)

        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, sync_dist=True)

        loss = torch.stack(list(losses.values())).sum()
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        fig = plot_alignment(
            info[0]["img"],
            info[0]["img_attn"],
            info[0]["txt"],
            info[0]["txt_attn"],
        )

        self.logger.log_image(
            key="attention",
            images=[fig],
        )
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), **self.optimizer_args)
        elif self.optimizer == "adamw_schedulefree":
            return AdamWScheduleFree(self.parameters(), **self.optimizer_args)
