import lightning as pl
import torch
import wandb
from einops import rearrange, repeat
from geomloss import SamplesLoss
from models.decoder import TransformerDecoder
from models.jepa import JEPA
from models.slot_attention import SA
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn
from torch.optim import AdamW
from torchmetrics.aggregation import MeanMetric
from utils.helpers import block_causal_mask
from utils.plot import denormalize_imagenet, plot_attention_interpreter_hierarchical
from x_transformers import Encoder


class InterpreterBlock(nn.Module):

    def __init__(
        self,
        config: dict,
        prev_n_slots: int,
    ):
        super().__init__()
        self.dim = config["dim"]
        self.slot_dim = config["slot_dim"]
        self.n_slots = config["n_slots"]
        self.time_shrink = config["time_shrink"]
        self.context_len = config["context_len"]

        self.decode_res = next(
            (i, prev_n_slots // i)
            for i in range(int(prev_n_slots**0.5), 0, -1)
            if prev_n_slots % i == 0
        )

        print(f"Decode res: {self.decode_res}")

        self.time_pe = PositionalEncoding1D(self.slot_dim)

        self.encoder = Encoder(
            dim=self.dim,
            depth=config["enc_depth"],
            ff_glu=True,
            ff_swish=True,
            attn_flash=True,
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
            dim=self.dim,
            dim_context=self.slot_dim,
            depth=config["dec_depth"],
            resolution=self.decode_res,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def update_target(self):
        for p, target_p in zip(self.encoder.parameters(), self.target.parameters()):
            target_p.data.mul_(self.alpha).add_(p.data.detach(), alpha=1 - self.alpha)

    def _time_pe(self, b, t, n, d):
        pe = self.time_pe(torch.zeros((b, t, d), device=self.device))
        return repeat(pe, "b t d -> b t n d", n=n)

    def shrink_time(self, x, add_pe=False):
        if self.time_shrink == 1:
            return x

        b, t, *_ = x.shape
        if t > self.time_shrink:
            slack = t % self.time_shrink
            x = x[:, : t - slack]

        x = rearrange(x, "b (t sf) ... -> (b t) sf ...", sf=self.time_shrink)
        if add_pe:
            x = x + self._time_pe(*x.shape)
        x = rearrange(x, "(b t) sf n ... -> b t (sf n) ...", b=b)

        return x

    def forward(self, x):
        b, *_ = x.shape

        x = self.shrink_time(x, add_pe=True)
        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.encoder(x)
        x = rearrange(x, "(b t) ... -> b t ...", b=b)

        if self.slot_attention is None:
            return x, None
        x = rearrange(x, "b t ... -> (b t) ...")
        slots, attn_map = self.slot_attention(x, n_slots=self.n_slots)
        slots = rearrange(slots, "(b t) ... -> b t ...", b=b)

        return slots, attn_map

    def predict_decode(self, slots):
        slots = slots[:, :-1]
        context_len = min(self.context_len, slots.shape[1])
        if slots.shape[1] > self.context_len:
            slack = slots.shape[1] % self.context_len
            slots = slots[:, : slots.shape[1] - slack]

        slots = rearrange(slots, "b (t s) n d -> (b t) s n d", s=context_len)

        b, t, n, *_ = slots.shape

        slots = slots + self._time_pe(b, t, n, self.slot_dim)
        slots = rearrange(slots, "b t n d -> b (t n) d")

        mask = block_causal_mask(t, n, device=self.device)
        predictions = self.predictor(slots, attn_mask=mask)
        predictions = rearrange(predictions, "b (t n) d -> (b t) n d", t=t)

        pred_dec = self.decoder(predictions)

        pred_dec = rearrange(
            pred_dec,
            "(b t) (sf n) d -> b (t sf) n d",
            b=b,
            sf=self.time_shrink,
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
        x = rearrange(x, "b t (sf n) ... -> b (t sf) n ...", sf=self.time_shrink)

        return x


class Interpreter(nn.Module):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
    ):
        super().__init__()

        self.base = JEPA(base_config)

        base_resolution = self.base.feature_map_resolution
        n_slots = base_resolution[0] * base_resolution[1]
        self.blocks = nn.ModuleList([])
        for i in range(len(block_configs)):
            self.blocks.append(InterpreterBlock(block_configs[i], prev_n_slots=n_slots))
            n_slots = block_configs[i]["n_slots"]

    def forward(self, x):

        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.base.teacher(x)
        x = rearrange(x, "(b t) ... -> b t ...")

        attn_maps = []
        features = [x]
        for block in self.blocks:  # type: ignore
            x, attn_map = block(x)
            features.append(x)
            attn_maps.append(attn_map)

        return x, attn_maps

    # Assumes that only the last block is to be trained
    def forward_train(self, x):
        attn_maps = []
        with torch.no_grad():
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.base.teacher(x)
            x = rearrange(x, "(b t) ... -> b t ...")

            for block in self.blocks[:-1]:  # type: ignore
                x, attn_map = block(x)
                attn_maps.append(attn_map)

        target = self.blocks[-1].make_target(x)

        x, attn_map = self.blocks[-1](x)
        attn_maps.append(attn_map)
        pred = self.blocks[-1].predict_decode(x)

        return pred, target, attn_maps


class InterpreterTrainer(pl.LightningModule):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
        optimizer_config: dict = {},
    ):
        super().__init__()

        self.resolution = base_config["resolution"]
        self.patch_size = base_config["patch_size"]
        self.shrink_factors = [config["shrink_factor"] for config in block_configs]

        self.optimizer_config = optimizer_config

        self.interpreter = Interpreter(base_config, block_configs)

        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5)

        self.metric_loggers = nn.ModuleDict(
            {
                "norm": MeanMetric(),
                "var": MeanMetric(),
                "loss": MeanMetric(),
            }
        )

    def forward(self, x, stage="train"):

        pred, target, attn_maps = self.interpreter.forward_train(x)

        pred = rearrange(pred, "b t ... -> (b t) ...")
        target = rearrange(target, "b t ... -> (b t) ...")

        loss = self.loss_fn(pred, target.detach()).mean()

        flat = rearrange(pred, "bt n d -> (bt n) d")
        self.metric_loggers["norm"](torch.norm(flat.detach(), dim=1).mean())  # type: ignore
        self.metric_loggers["var"](torch.var(flat.detach(), dim=0).mean())  # type: ignore

        return loss, attn_maps

    def log_metrics(self, batch_idx):
        if batch_idx % self.trainer.accumulate_grad_batches != 0:
            return

        for k, v in self.metric_loggers.items():
            value = v.compute()
            if torch.isnan(value):
                continue
            self.log(
                f"train/{k}_{len(self.blocks)}", value, prog_bar=True, sync_dist=True
            )
            v.reset()

    def training_step(self, x, batch_idx: int):

        loss, *_ = self.forward_hierarchy(x)

        self.log_metrics(batch_idx)

        return loss

    def validation_step(self, x):

        loss, attn_maps = self.forward_hierarchy(x, stage="val")

        self.log(f"val/loss_{len(self.blocks)}", loss.item(), sync_dist=True)

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
                    attn.cpu().numpy() * 255, fps=5, format="gif"
                )

            self.logger.experiment.log(log_dict)  # type: ignore

        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.blocks[0].update_target()

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
