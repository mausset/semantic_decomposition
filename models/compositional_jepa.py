import lightning as pl
import timm
import torch
import wandb
from einops import rearrange, repeat
from geomloss import SamplesLoss

from torchmetrics.aggregation import MeanMetric

# from matplotlib import pyplot as plt
from models.decoder import TransformerDecoder
from models.slot_attention import SA
from positional_encodings.torch_encodings import PositionalEncoding1D
from timm.layers.config import set_fused_attn
from torch import nn
from torch.optim import AdamW
from utils.helpers import block_causal_mask
from utils.plot import plot_attention_interpreter_hierarchical
from x_transformers import Encoder


class CJEPALayer(nn.Module):

    def __init__(
        self,
        encoder_args: dict | None,
        slot_attention_args: dict,
        predictor_args: dict,
        decoder_args: dict,
        shrink_factor: int,
        dim: int,
        decode_res: tuple[int, int],
        n_slots: int,
        max_seq_len: int = 32,
    ):
        super().__init__()

        self.time_pe = PositionalEncoding1D(dim)

        self.encoder = Encoder(**encoder_args) if encoder_args else None
        self.slot_attention = SA(**slot_attention_args, n_slots=n_slots)
        self.predictor = Encoder(**predictor_args)
        self.decoder = TransformerDecoder(decoder_args)

        self.shrink_factor = shrink_factor
        self.dim = dim
        self.decode_res = decode_res
        self.n_slots = n_slots
        self.max_seq_len = max_seq_len + 1

    @property
    def device(self):
        return next(self.parameters()).device

    def _time_pe(self, b, t, n, d):
        pe = self.time_pe(torch.zeros((b, t, d), device=self.device))
        return repeat(pe, "b t d -> b t n d", n=n)

    def shrink_time(self, x, add_pe=True):
        if self.shrink_factor == 1:
            return x

        b, t, *_ = x.shape
        slack = t % self.shrink_factor
        x = x[:, : t - slack]
        x = rearrange(x, "b (t sf) ... -> (b t) sf ...", sf=self.shrink_factor)
        if add_pe:
            x = x + self._time_pe(*x.shape)
        x = rearrange(x, "(b t) sf n ... -> b t (sf n) ...", b=b)

        return x

    def forward(self, x):
        b, t, *_ = x.shape

        x = self.shrink_time(x)
        if self.encoder:
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.encoder(x)
            x = rearrange(x, "(b t) ... -> b t ...", b=b)

        b, t, *_ = x.shape
        x = rearrange(x, "b t ... -> (b t) ...")
        slots, attn_map = self.slot_attention(x, n_slots=self.n_slots)
        slots = rearrange(slots, "(b t) n d -> b t n d", t=t)

        return slots, attn_map

    def predict_decode(self, slots):

        slots = slots[:, : min(self.max_seq_len, slots.shape[1]) - 1]

        b, t, *_ = slots.shape

        slots_te = slots + self._time_pe(b, t, self.n_slots, self.dim)
        slots_te = rearrange(slots_te, "b t n d -> b (t n) d")

        mask = block_causal_mask(t, self.n_slots, device=self.device)
        predictions = self.predictor(slots_te, attn_mask=mask)
        predictions = rearrange(predictions, "b (t n) d -> (b t) n d", t=t)

        pred_dec = self.decoder(predictions, resolution=self.decode_res)
        pred_dec = rearrange(pred_dec, "(b t) n d -> b t n d", b=b)
        pred_dec = rearrange(
            pred_dec, "b t (sf n) ... -> b (t sf) n ...", sf=self.shrink_factor
        )

        return pred_dec

    def make_target(self, x):
        x = self.shrink_time(x, add_pe=False)
        x = x[:, 1 : min(self.max_seq_len, x.shape[1])]
        x = rearrange(x, "b t (sf n) ... -> b (t sf) n ...", sf=self.shrink_factor)

        return x


class CompositionalJEPA(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        encoder_args: dict,
        slot_attention_args: dict,
        predictor_args: dict,
        decoder_args: dict,
        loss_args: dict,
        shrink_factors: list[int],
        dim: int,
        resolution: tuple[int, int],
        schedule: list = [],
        n_slots: list[int] = [16, 8],
        max_seq_lens: list[int] = [32, 32],
        optimizer: str = "adamw",
        optimizer_args: dict = {},
    ):
        super().__init__()

        self.dim = dim
        self.n_slots = n_slots
        self.shrink_factors = shrink_factors
        self.resolution = resolution

        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        self.schedule = schedule
        self.current_config = schedule[0]

        set_fused_attn(True)
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

        self.discard_tokens = 1 + (4 if "reg4" in image_encoder_name else 0)
        self.patch_size = self.image_encoder.patch_embed.patch_size[0]

        self.time_pe = PositionalEncoding1D(dim)

        base_res = (resolution[0] // self.patch_size, resolution[1] // self.patch_size)
        self.layers = nn.ModuleList(
            [
                CJEPALayer(
                    encoder_args=encoder_args if i > 0 else None,
                    slot_attention_args=slot_attention_args,
                    predictor_args=predictor_args,
                    decoder_args=decoder_args,
                    shrink_factor=shrink_factors[i],
                    dim=dim,
                    decode_res=(shrink_factors[i], n_slots[i]) if i > 0 else base_res,
                    n_slots=n_slots[i],
                    max_seq_len=max_seq_lens[i],
                )
                for i in range(len(shrink_factors))
            ]
        )

        self.loss_fn = SamplesLoss(**loss_args)

        self.metric_loggers = nn.ModuleList([])
        for _ in range(len(self.layers)):
            self.metric_loggers.append(
                nn.ModuleDict(
                    {
                        "mean_norm": MeanMetric(),
                        "mean_var": MeanMetric(),
                        "loss": MeanMetric(),
                    }
                )
            )

    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def update_config(self, step):
        for config in self.schedule:
            if step > config["start_step"]:
                self.current_config = config

        for train_config, layer in zip(self.current_config["train"], self.layers):
            layer.requires_grad_(train_config)

    def forward_hierarchy(self, x, stage="train"):
        b, *_ = x.shape

        x = x[:, : self.current_config["t_max"]]
        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.forward_features(x)
        x = rearrange(x, "(b t) ... -> b t ...", b=b)

        attn_maps = []
        losses = {}
        for idx, layer in enumerate(self.layers):
            if self.current_config["skip"][idx]:
                break

            slots, attn_map = layer(x)
            attn_maps.append(attn_map)

            if self.current_config["train"][idx]:
                pred_dec = layer.predict_decode(slots)
                target = layer.make_target(x)
                pred_dec = rearrange(pred_dec, "b t n d -> (b t) n d")
                target = rearrange(target, "b t n d -> (b t) n d")
                loss = self.loss_fn(pred_dec, target.detach()).mean()
                losses[idx] = loss

            x = slots.detach()

            flat = rearrange(x, "b t n d -> (b t n) d")
            self.metric_loggers[idx]["mean_norm"](torch.norm(flat, dim=1).mean())  # type: ignore
            self.metric_loggers[idx]["mean_var"](torch.var(flat, dim=0).mean())  # type: ignore

        return losses, attn_maps

    def log_metrics(self, batch_idx, stage="train"):
        if stage == "train" and batch_idx % self.trainer.accumulate_grad_batches != 0:
            return

        for idx, logger in enumerate(self.metric_loggers):
            for k, v in logger.items():
                value = v.compute()
                if torch.isnan(value):
                    continue
                self.log(f"{stage}/{k}_{idx}", value, prog_bar=True, sync_dist=True)
                v.reset()

    def training_step(self, x, batch_idx: int):
        self.update_config(self.global_step)

        losses, _ = self.forward_hierarchy(x)

        for k, v in losses.items():
            self.metric_loggers[k]["loss"](v)  # type: ignore

        self.log_metrics(batch_idx)

        loss = torch.stack(list(losses.values())).sum()

        return loss

    def validation_step(self, x):

        losses, attn_maps = self.forward_hierarchy(x, stage="val")

        for k, v in losses.items():
            self.log(f"val/loss_{k}", v.item(), sync_dist=True)

        attn_plots = plot_attention_interpreter_hierarchical(
            x=x,
            attn_maps=attn_maps,
            shrink_factors=self.shrink_factors,
            t_max=self.current_config["t_max"],
            res=self.resolution,
            patch_size=self.patch_size,
        )
        if (
            isinstance(self.logger, pl.pytorch.loggers.WandbLogger)  # type: ignore
            and self.trainer.global_rank == 0
        ):
            log_dict = {}
            for idx, attn in enumerate(attn_plots):
                log_dict[f"attention_{idx}"] = wandb.Video(
                    attn.cpu().numpy() * 255, fps=8, format="gif"
                )

            self.logger.experiment.log(log_dict)  # type: ignore

        loss = torch.stack(list(losses.values())).sum()

        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), **self.optimizer_args)
        else:
            raise NotImplementedError
