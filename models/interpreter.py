import lightning as pl

# from models.jepa import JEPA
import timm
import torch
import wandb
from einops import rearrange, repeat
from geomloss import SamplesLoss
from models.decoder import TransformerDecoder
from models.slot_attention import SA
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn
from torch.optim import AdamW
from torchmetrics.aggregation import MeanMetric
from utils.helpers import block_causal_mask
from utils.schedulers import WarmupCosineSchedule
from utils.plot import (
    plot_attention_interpreter_hierarchical,
    plot_attention_simple,
)
from x_transformers import Encoder


class TimmWrapper(nn.Module):

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 0,
        in_chans: int = 3,
        resolution: int = 224,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            img_size=resolution,
        )

        self.discard = 1 + 4 if "reg" in model_name else 1

        self.patch_size = self.model.patch_embed.patch_size[0]

    def forward(self, x):
        return self.model.forward_features(x)[:, self.discard :]


class InterpreterBlock(nn.Module):

    def __init__(
        self,
        config: dict,
        n_decode: int,
    ):
        super().__init__()
        self.dim = config["dim"]
        self.slot_dim = config["slot_dim"]
        self.n_slots = config["n_slots"]
        self.time_shrink = config.get("time_shrink", 1)
        self.context_len = config["context_len"]

        self.decode_res = next(
            (i, n_decode // i)
            for i in range(int(n_decode**0.5), 0, -1)
            if n_decode % i == 0
        )

        print(f"Decode res: {self.decode_res}")

        self.time_pe = PositionalEncoding1D(self.slot_dim)

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

        slots, attn_map = self.slot_attention(x, n_slots=self.n_slots)
        slots = rearrange(slots, "(b t) ... -> b t ...", b=b)

        return slots, attn_map

    def predict(self, x):
        x = x[:, :-1]
        context_len = min(self.context_len, x.shape[1])
        if x.shape[1] > self.context_len:
            slack = x.shape[1] % self.context_len
            x = x[:, : x.shape[1] - slack]

        x = rearrange(x, "b (t s) n d -> (b t) s n d", s=context_len)

        b, t, n, *_ = x.shape
        x = x + self._time_pe(b, t, n, self.slot_dim)
        x = rearrange(x, "b t n d -> b (t n) d")

        mask = block_causal_mask(t, n, device=self.device)
        pred = self.predictor(x, attn_mask=mask)
        pred = rearrange(pred, "b (t n) d -> b t n d", t=t)

        return pred

    def make_pred_target(self, x):
        x = x[:, 1:]
        context_len = min(self.context_len, x.shape[1])
        if x.shape[1] > self.context_len:
            slack = x.shape[1] % self.context_len
            x = x[:, : x.shape[1] - slack]

        return rearrange(x, "b (t s) n d -> (b t) s n d", s=context_len)

    def decode(self, x):
        b = x.shape[0]

        x = rearrange(x, "b t n d -> (b t) n d")
        dec = self.decoder(x)

        dec = rearrange(
            dec,
            "(b t) (sf n) d -> b (t sf) n d",
            b=b,
            sf=self.time_shrink,
        )

        return dec

    def make_dec_target(self, x):
        x = self.shrink_time(x)
        return rearrange(x, "b t (sf n) ... -> b (t sf) n ...", sf=self.time_shrink)


class Interpreter(nn.Module):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
        active_block: int | None = None,
    ):
        super().__init__()

        if active_block is not None:
            block_configs = block_configs[: active_block + 1]

        self.base = TimmWrapper(**base_config).eval().requires_grad_(False)

        base_resolution = base_config["resolution"][0] // self.base.patch_size
        n_decode = base_resolution**2
        self.blocks = nn.ModuleList([])
        for i in range(len(block_configs)):
            self.blocks.append(InterpreterBlock(block_configs[i], n_decode=n_decode))
            n_decode = block_configs[i]["n_slots"]

        for block in self.blocks[:-1]:  # type: ignore
            block.eval().requires_grad_(False)

    def forward(self, x):

        b = x.shape[0]
        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.base(x)
        x = rearrange(x, "(b t) ... -> b t ...", b=b)

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
            b = x.shape[0]
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.base(x)
            x = rearrange(x, "(b t) ... -> b t ...", b=b)

            for block in self.blocks[:-1]:  # type: ignore
                x, attn_map = block(x)
                attn_maps.append(attn_map)

        dec_target = self.blocks[-1].make_dec_target(x)

        x, attn_map = self.blocks[-1](x)
        attn_maps.append(attn_map)

        pred_target = self.blocks[-1].make_pred_target(x)
        pred = self.blocks[-1].predict(x)

        dec = self.blocks[-1].decode(x)

        return dec, dec_target, pred, pred_target, attn_maps


class InterpreterTrainer(pl.LightningModule):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
        active_block: int | None = None,
        optimizer_config: dict = {},
        log_config: dict = {},
        checkpoint_path: str | None = None,
    ):
        super().__init__()

        self.interpreter = torch.compile(
            Interpreter(base_config, block_configs, active_block)
        )

        if checkpoint_path is not None:
            self.load_state_dict(
                torch.load(checkpoint_path)["state_dict"], strict=False
            )

        self.patch_size = self.interpreter.base.patch_size  # type: ignore
        self.resolution = base_config["resolution"]
        self.time_shrink = [block.time_shrink for block in self.interpreter.blocks]  # type: ignore
        self.n_blocks = len(self.interpreter.blocks)  # type: ignore
        self.optimizer_config = optimizer_config
        self.log_config = log_config

        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5)

        self.metric_loggers = nn.ModuleDict(
            {
                "norm": MeanMetric(),
                "var": MeanMetric(),
                "loss_decode": MeanMetric(),
                "loss_prediction": MeanMetric(),
            }
        )

        self.gif = None

    def forward(self, x, stage="train"):

        dec, dec_target, pred, pred_target, attn_maps = self.interpreter.forward_train(x)  # type: ignore

        dec = rearrange(dec, "b t ... -> (b t) ...")
        dec_target = rearrange(dec_target, "b t ... -> (b t) ...")

        pred = rearrange(pred, "b t ... -> (b t) ...")
        pred_target = rearrange(pred_target, "b t ... -> (b t) ...")

        loss_decode = self.loss_fn(dec, dec_target).mean()
        loss_pred = self.loss_fn(pred, pred_target).mean()

        flat = rearrange(pred, "bt n d -> (bt n) d")
        self.metric_loggers["norm"](torch.norm(flat.detach(), dim=1).mean())  # type: ignore
        self.metric_loggers["var"](torch.var(flat.detach(), dim=0).mean())  # type: ignore
        self.metric_loggers["loss_decode"](loss_decode.detach())  # type: ignore
        self.metric_loggers["loss_prediction"](loss_pred.detach())  # type: ignore

        loss = loss_decode + loss_pred

        return loss, attn_maps

    def log_metrics(self, batch_idx):
        if batch_idx % self.trainer.accumulate_grad_batches != 0:
            return

        for k, v in self.metric_loggers.items():
            value = v.compute()
            if torch.isnan(value):
                continue
            self.log(
                f"train/{k}_{self.n_blocks-1}",
                value,
                prog_bar=True,
                sync_dist=True,
            )
            v.reset()

    def on_train_epoch_end(self):
        if self.gif is not None:
            self.gif = None

    def training_step(self, x, batch_idx: int):

        loss, attn_maps = self.forward(x)

        self.log_metrics(batch_idx)

        if (
            isinstance(self.logger, pl.pytorch.loggers.WandbLogger)  # type: ignore
            and self.trainer.is_global_zero
            and self.gif is None
        ):
            attn_plot = plot_attention_simple(
                imgs=x.detach(),
                attn_maps=[attn.detach() for attn in attn_maps],
                res=self.resolution,
                patch_size=self.patch_size,
                n_frames=self.log_config["n_frames"],
            )
            log_dict = {
                f"attention_{self.n_blocks-1}": wandb.Video(
                    (attn_plot.cpu().numpy() * 255).astype("uint8"),
                    fps=self.log_config["fps"],
                    format="gif",
                )
            }
            self.logger.experiment.log(log_dict)  # type: ignore

            # self.logger.experiment.log(log_dict)  # type: ignore
            self.gif = log_dict

        return loss

    def validation_step(self, x):
        raise NotImplementedError

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):  # type: ignore
        optimizer = AdamW(
            self.parameters(), weight_decay=self.optimizer_config["weight_decay"]
        )

        steps_per_epoch = self.trainer.estimated_stepping_batches / self.trainer.max_epochs  # type: ignore

        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=self.optimizer_config["warmup_epochs"] * steps_per_epoch,
            start_lr=self.optimizer_config["start_lr"],
            ref_lr=self.optimizer_config["ref_lr"],
            final_lr=self.optimizer_config["final_lr"],
            T_max=self.trainer.max_epochs * steps_per_epoch,  # type: ignore
        )

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

        return config
