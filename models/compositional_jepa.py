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


class CJEPABlock(nn.Module):

    def __init__(
        self,
        config: dict,
        prev_n_slots: int | tuple[int, int],
    ):
        super().__init__()
        self.dim = config["dim"]
        self.slot_dim = config["slot_dim"]
        self.n_slots = config["n_slots"]
        self.shrink_factor = config["shrink_factor"]
        self.context_len = config["context_len"]
        if isinstance(prev_n_slots, tuple):
            self.decode_res = prev_n_slots
        else:
            # self.decode_res = (config["shrink_factor"], prev_n_slots)
            self.decode_res = next(
                (i, prev_n_slots // i)
                for i in range(int(prev_n_slots**0.5), 0, -1)
                if prev_n_slots % i == 0
            )

        print(f"Decode res: {self.decode_res}")

        self.time_pe = PositionalEncoding1D(self.slot_dim)

        self.encoder = (
            Encoder(
                dim=self.dim,
                depth=config["enc_depth"],
                ff_glu=True,
                ff_swish=True,
                attn_flash=True,
            )
            if "enc_depth" in config
            else None
        )
        self.slot_attention = SA(
            input_dim=self.dim,
            slot_dim=self.slot_dim,
            n_slots=self.n_slots,
            n_iters=8,
        )
        self.predictor = Encoder(
            dim=self.slot_dim,
            depth=config["pred_depth"],
            ff_glu=True,
            ff_swish=True,
            attn_flash=True,
        )
        self.decoder = TransformerDecoder(
            dim=self.dim, dim_context=self.slot_dim, depth=config["dec_depth"]
        )

        if config["loss"] == "mse":
            self.loss_fn = nn.MSELoss()
        elif config["loss"] == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5)

    @property
    def device(self):
        return next(self.parameters()).device

    def _time_pe(self, b, t, n, d):
        pe = self.time_pe(torch.zeros((b, t, d), device=self.device))
        return repeat(pe, "b t d -> b t n d", n=n)

    def shrink_time(self, x, add_pe=False):
        if self.shrink_factor == 1:
            return x

        b, t, *_ = x.shape
        if t > self.shrink_factor:
            slack = t % self.shrink_factor
            x = x[:, : t - slack]

        x = rearrange(x, "b (t sf) ... -> (b t) sf ...", sf=self.shrink_factor)
        if add_pe:
            x = x + self._time_pe(*x.shape)
        x = rearrange(x, "(b t) sf n ... -> b t (sf n) ...", b=b)

        return x

    def forward(self, x):
        b, t, *_ = x.shape

        x = self.shrink_time(x, add_pe=True)
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

        slots = slots[:, :-1]
        context_len = min(self.context_len, slots.shape[1])
        if slots.shape[1] > self.context_len:
            slack = slots.shape[1] % self.context_len
            slots = slots[:, : slots.shape[1] - slack]

        slots = rearrange(slots, "b (t s) n d -> (b t) s n d", s=context_len)

        b, t, *_ = slots.shape

        slots_te = slots + self._time_pe(b, t, self.n_slots, self.slot_dim)
        slots_te = rearrange(slots_te, "b t n d -> b (t n) d")

        mask = block_causal_mask(t, self.n_slots, device=self.device)
        predictions = self.predictor(slots_te, attn_mask=mask)
        predictions = rearrange(predictions, "b (t n) d -> (b t) n d", t=t)

        pred_dec = self.decoder(predictions, resolution=self.decode_res)
        pred_dec = rearrange(
            pred_dec, "(b t) (sf n) d -> b (t sf) n d", b=b, sf=self.shrink_factor
        )

        return pred_dec

    def make_target(self, x):
        x = self.shrink_time(x)
        x = x[:, 1:]
        context_len = min(self.context_len, x.shape[1])
        if x.shape[1] > self.context_len:
            slack = x.shape[1] % self.context_len
            x = x[:, : x.shape[1] - slack]
        x = rearrange(x, "b (t s) n d -> (b t) s n d", s=context_len)
        x = rearrange(x, "b t (sf n) ... -> b (t sf) n ...", sf=self.shrink_factor)

        return x

    def calc_loss(self, x, slots):
        pred_dec = self.predict_decode(slots)
        target = self.make_target(x)

        assert pred_dec.shape == target.shape

        pred_dec = rearrange(pred_dec, "b t n d -> (b t) n d")
        target = rearrange(target, "b t n d -> (b t) n d")

        return self.loss_fn(pred_dec, target.detach()).mean()


class CompositionalJEPA(pl.LightningModule):

    def __init__(
        self,
        image_encoder_name,
        block_configs: list[dict],
        resolution: tuple[int, int],
        schedule: list = [],
        optimizer_config: dict = {},
        lr_warmup_steps: int = 0,
    ):
        super().__init__()

        self.resolution = resolution
        self.shrink_factors = [config["shrink_factor"] for config in block_configs]

        self.optimizer_config = optimizer_config
        self.lr_warmup_steps = lr_warmup_steps

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

        base_res = (resolution[0] // self.patch_size, resolution[1] // self.patch_size)
        self.layers = nn.ModuleList([])
        for i in range(len(block_configs)):
            self.layers.append(
                CJEPABlock(
                    block_configs[i],
                    prev_n_slots=(
                        block_configs[i - 1]["n_slots"] if i > 0 else base_res
                    ),
                )
            )

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

        if "t_max" in self.current_config:
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
                loss = layer.calc_loss(x, slots)
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
                    attn.cpu().numpy() * 255, fps=3, format="gif"
                )

            self.logger.experiment.log(log_dict)  # type: ignore

        loss = torch.stack(list(losses.values())).sum()

        return loss

    def dynamic_warmup(self, step):
        lr_factor = min(
            (step - self.current_config["start_step"]) / self.lr_warmup_steps, 1.0
        )
        return lr_factor

    def configure_optimizers(self):  # type: ignore
        optimizer = AdamW(self.parameters(), **self.optimizer_config)
        if self.lr_warmup_steps == 0:
            return optimizer

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: self.dynamic_warmup(step),
        )

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

        return config
