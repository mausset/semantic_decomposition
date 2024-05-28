import lightning as pl
import timm
import torch
from einops import rearrange
from models.decoder import TransformerDecoder
from models.slot_attention import build_slot_attention
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from utils.helpers import pad_batched_slots
from utils.metrics import ARIMetric, UnsupervisedMaskIoUMetric
from utils.plot import plot_attention_hierarchical

from matplotlib import pyplot as plt


class Interpreter(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        slot_attention_arch: str,
        slot_attention_args: dict,
        feature_decoder_args: dict,
        dim: int,
        resolution: tuple[int, int],
        loss_strategy: str,
        decode_strategy: str,
        shared_weights: tuple[bool, bool] = [True, True],
        n_slots=[16, 8],
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

        if shared_weights[0]:
            self.slot_attention = build_slot_attention(
                slot_attention_arch, slot_attention_args
            )
        else:
            self.slot_attention = nn.ModuleList(
                [
                    build_slot_attention(slot_attention_arch, slot_attention_args)
                    for _ in n_slots
                ]
            )

        if shared_weights[1]:
            self.decoder = TransformerDecoder(**feature_decoder_args)
        else:
            self.decoder = nn.ModuleList(
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
        self.loss_fn = torch.nn.MSELoss()
        self.loss_strategy = loss_strategy
        self.decode_strategy = decode_strategy

        self.shared_weights = shared_weights
        self.n_slots = n_slots
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        self.mbo_i_metric = UnsupervisedMaskIoUMetric(
            matching="best_overlap", ignore_background=True, ignore_overlaps=True
        ).to(self.device)
        self.mbo_c_metric = UnsupervisedMaskIoUMetric(
            matching="best_overlap", ignore_background=True, ignore_overlaps=True
        ).to(self.device)

        self.miou_metric = UnsupervisedMaskIoUMetric(
            matching="hungarian", ignore_background=True, ignore_overlaps=True
        ).to(self.device)

        self.ari_metric = ARIMetric(foreground=True, ignore_overlaps=True).to(
            self.device
        )

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def sample_coordinates(self, x, n_slots):
        return torch.rand(x.shape[0], n_slots, 2, device=x.device)

    def encode(self, x):
        slots = x

        up = {slots.shape[1]: slots}
        attn_list = []
        slots_list = []

        slot_attention_list = self.slot_attention
        if self.shared_weights[0]:
            slot_attention_list = [self.slot_attention] * len(self.n_slots)

        for n, sa in zip(self.n_slots, slot_attention_list):
            slots, attn_map = sa(slots, n_slots=n)
            if attn_list:
                attn_map = attn_list[-1] @ attn_map
            attn_list.append(attn_map)
            slots_list.append(slots)
            up[n] = slots

        return up, slots_list, attn_list

    def decode(self, slots_list):

        down = {}

        decoder_list = self.decoder
        if self.shared_weights[1]:
            decoder_list = [self.decoder] * len(self.n_slots)

        if self.decode_strategy == "flat":
            slots_dict = {n: slots for n, slots in zip(self.n_slots, slots_list)}

            slots, mask = pad_batched_slots(slots_dict)

            decoded, _ = self.decoder(
                slots,
                resolution=self.feature_resolution,
                context_mask=mask,
            )

            down[self.n_slots[0]] = decoded

        if self.decode_strategy == "hierarchical":
            for i in range(len(self.n_slots) - 1, 0, -1):
                slots = slots_list[i]
                res = (1, slots_list[i - 1].shape[1])
                decoded, _ = decoder_list[i](
                    slots,
                    resolution=res,
                )
                down[self.n_slots[i]] = decoded
                if self.loss_strategy == "anchored":
                    slots_list[i - 1] = torch.cat([slots_list[i - 1], decoded], dim=0)

            decoded, _ = decoder_list[0](slots_list[0], self.feature_resolution)
            down[self.n_slots[0]] = decoded

        if self.decode_strategy == "concat":
            slots = torch.cat(slots_list, dim=1)

            decoded, _ = self.decoder(
                slots,
                resolution=self.feature_resolution,
            )

            down[self.n_slots[0]] = decoded

        return down

    def calculate_loss(self, up, down):
        """
        args:
            up: dict, up features
            down: dict, down features

        returns:
            loss, dict
        """

        losses = {}
        if self.loss_strategy == "flat":
            decoded_features = down[self.n_slots[0]]
            features = up[decoded_features.shape[1]]
            decoded_chunked = torch.chunk(decoded_features, len(self.n_slots), dim=0)

            for n, chunk in zip(self.n_slots, decoded_chunked):
                loss = self.loss_fn(chunk, features)
                losses[n] = loss

        if self.loss_strategy == "concat":
            decoded_features = down[self.n_slots[0]]
            features = up[decoded_features.shape[1]]
            loss = self.loss_fn(decoded_features, features)
            losses["concat"] = loss

        return losses

    def common_step(self, x):

        features = self.forward_features(x)

        up, slots_list, attn_list = self.encode(features)

        down = self.decode(slots_list)
        losses = self.calculate_loss(up, down)

        return losses, attn_list

    def training_step(self, x):
        losses, _ = self.common_step(x)

        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, prog_bar=True, sync_dist=True)

        loss = torch.stack(list(losses.values())).mean()
        # loss = torch.stack(list(losses.values())).sum()
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def calc_metrics(self, masks, attn_list):

        i_mask, c_mask, ignore_mask = masks

        attn_list = [
            rearrange(a, "b (h w) n -> b n h w", h=self.feature_resolution[0])
            for a in attn_list
        ]

        attn_list = [
            F.interpolate(a, size=self.resolution, mode="bilinear").unsqueeze(2)
            for a in attn_list
        ]

        pred_mask = [a.argmax(1).squeeze(1) for a in attn_list]

        true_mask_i = F.one_hot(i_mask).to(torch.float32).permute(0, 3, 1, 2)
        true_mask_c = F.one_hot(c_mask).to(torch.float32).permute(0, 3, 1, 2)
        pred_mask = [
            F.one_hot(m).to(torch.float32).permute(0, 3, 1, 2) for m in pred_mask
        ]

        self.mbo_i_metric.update(pred_mask[-1], true_mask_i, ignore_mask)
        self.mbo_c_metric.update(pred_mask[-1], true_mask_c, ignore_mask)
        self.miou_metric.update(pred_mask[-1], true_mask_i, ignore_mask)
        self.ari_metric.update(pred_mask[-1], true_mask_i, ignore_mask)

        mbo_i = self.mbo_i_metric.compute()
        mbo_c = self.mbo_c_metric.compute()
        miou = self.miou_metric.compute()
        ari = self.ari_metric.compute()

        return mbo_i, mbo_c, miou, ari

    def validation_step(self, x):
        x, *masks = x

        losses, attn_list = self.common_step(x)

        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, sync_dist=True)

        loss = torch.stack(list(losses.values())).mean()
        # loss = torch.stack(list(losses.values())).sum()
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

        mbo_i, mbo_c, miou, ari = self.calc_metrics(masks, attn_list)

        self.log("val/mbo_i", mbo_i, sync_dist=True, prog_bar=True)
        self.log("val/mbo_c", mbo_c, sync_dist=True, prog_bar=True)
        self.log("val/miou", miou, sync_dist=True, prog_bar=True)
        self.log("val/ari", ari, sync_dist=True, prog_bar=True)

        return loss

    def on_validation_end(self):
        self.mbo_i_metric.reset()
        self.mbo_c_metric.reset()
        self.miou_metric.reset()
        self.ari_metric.reset()

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), **self.optimizer_args)
        else:
            raise NotImplementedError
