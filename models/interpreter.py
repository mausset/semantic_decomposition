import lightning as pl

import timm
import torch
import wandb
from einops import rearrange, repeat
from geomloss import SamplesLoss
from models.decoder import TransformerDecoder, TransformerDecoderOld
from models.slot_attention import SA
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn
from torch.optim.adamw import AdamW
from torchmetrics.aggregation import MeanMetric
from utils.schedulers import WarmupCosineSchedule
from utils.plot import (
    plot_attention_interpreter_hierarchical,
    plot_attention_hierarchical,
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
        n_decode: int | tuple[int, int],
    ):
        super().__init__()
        self.dim = config["dim"]
        self.slot_dim = config["slot_dim"]
        self.n_slots = config["n_slots"]
        self.time_shrink = config.get("time_shrink", 1)

        if isinstance(n_decode, int):
            self.decode_res = next(
                (i, n_decode // i)
                for i in range(int(n_decode**0.5), 0, -1)
                if n_decode % i == 0
            )
        else:  # Tuple
            self.decode_res = n_decode
        if self.time_shrink > 1:
            self.decode_res = (
                self.time_shrink,
                self.decode_res[0] * self.decode_res[1],
            )

        print(f"Decode res: {self.decode_res}")

        self.time_pe = PositionalEncoding1D(self.slot_dim)

        self.slot_attention = SA(
            input_dim=self.dim,
            slot_dim=self.slot_dim,
            n_slots=self.n_slots,
            sampler=config.get("sampler", "gaussian"),
            n_iters=8,
        )

        if "enc_depth" in config:
            self.encoder = Encoder(
                dim=self.dim,
                depth=config["enc_depth"],
                ff_glu=True,
                attn_flash=True,
            )
        else:
            self.encoder = None

        self.decoder = TransformerDecoder(
            dim=self.dim,
            depth=config["dec_depth"],
            resolution=self.decode_res,
            sincos=True,
        )

    @property
    def device(self):
        return next(self.parameters()).device

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
        if self.encoder is not None:
            x = self.encoder(x)

        slots, attn_map = self.slot_attention(x, n_slots=self.n_slots)
        slots = rearrange(slots, "(b t) ... -> b t ...", b=b)

        return slots, attn_map

    def decode(self, x):
        b, *_ = x.shape

        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.decoder(x)
        x = rearrange(
            x,
            "(b t) (s n) d -> b (t s) n d",  # Ablate this
            b=b,
            s=self.time_shrink,
        )

        return x


class Interpreter(nn.Module):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
    ):
        super().__init__()

        self.base = TimmWrapper(**base_config).eval().requires_grad_(False)

        base_resolution = (
            base_config["resolution"][0] // self.base.patch_size,
            base_config["resolution"][1] // self.base.patch_size,
        )
        n_decode = base_resolution
        self.blocks = nn.ModuleList([])
        for i in range(len(block_configs)):
            self.blocks.append(InterpreterBlock(block_configs[i], n_decode=n_decode))
            n_decode = block_configs[i]["n_slots"]

    def setup_only_last(self):
        for block in self.blocks[:-1]:  # type: ignore
            block.eval().requires_grad_(False)

    def train_only_last(self, x):
        b = x.shape[0]

        attn_maps = []
        with torch.no_grad():
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.base(x)
            x = rearrange(x, "(b t) ... -> b t ...", b=b)
            for block in self.blocks[:-1]:  # type: ignore
                x, attn_map = block(x)
                attn_maps.append(attn_map)
                x = x.detach()

        features = [x]
        x, attn_map = self.blocks[-1](x)
        attn_maps.append(attn_map)
        decoded = [self.blocks[-1].decode(x)]

        return decoded, features, attn_maps

    def forward(self, x, current_block=None):
        b = x.shape[0]

        with torch.no_grad():
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.base(x)
            x = rearrange(x, "(b t) ... -> b t ...", b=b)

        blocks = self.blocks
        if current_block is not None:
            blocks = blocks[: current_block + 1]

        attn_maps = []
        features = [x]
        decoded = []
        for block in blocks:  # type: ignore
            x, attn_map = block(x)
            features.append(x)
            attn_maps.append(attn_map)
            decoded.append(block.decode(x))
            x = x.detach()

        return decoded, features, attn_maps


class InterpreterTrainer(pl.LightningModule):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
        optimizer_config: dict = {},
        log_config: dict = {},
        layer_schedule: dict | None = {},
        only_last: bool = False,
        checkpoint_path: str | None = None,
    ):
        super().__init__()

        self.model = Interpreter(base_config, block_configs)

        if checkpoint_path is not None:
            self.load_state_dict(
                torch.load(checkpoint_path)["state_dict"], strict=False
            )

        self.patch_size = self.model.base.patch_size  # type: ignore
        self.resolution = base_config["resolution"]
        self.time_shrink = [block.time_shrink for block in self.model.blocks]  # type: ignore
        self.n_blocks = len(self.model.blocks)  # type: ignore
        self.optimizer_config = optimizer_config
        self.log_config = log_config
        self.layer_schedule = layer_schedule
        self.only_last = only_last

        if self.only_last:
            self.model.setup_only_last()

        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5)

        self.metric_loggers = nn.ModuleDict(
            {f"loss_{i}": MeanMetric() for i in range(self.n_blocks)}
        )

        self.gif = None

    @property
    def current_block(self):
        if self.layer_schedule is None:
            return self.n_blocks

        schedule = sorted(list(self.layer_schedule.keys()))

        current = 0
        for k in schedule:
            if self.current_epoch >= k:
                current = self.layer_schedule[k]

        return current

    def forward(self, x, stage="train"):

        if self.only_last:
            decoded, target, attn_maps = self.model.train_only_last(x)
        else:
            decoded, target, attn_maps = self.model.forward(x, current_block=self.current_block)  # type: ignore

        loss = 0
        start = len(self.model.blocks) - 1 if self.only_last else 0
        for i, (d, t) in enumerate(zip(decoded, target), start=start):
            d = rearrange(d, "b t ... -> (b t) ...")
            t = rearrange(t, "b t ... -> (b t) ...")
            local_loss = self.loss_fn(d, t.detach()).mean()

            self.metric_loggers[f"loss_{i}"](local_loss.detach())
            loss += local_loss

        return loss, attn_maps

    def log_metrics(self):

        for k, v in self.metric_loggers.items():
            value = v.compute()
            if torch.isnan(value):
                continue
            self.log(
                f"train/{k}",
                value,
                prog_bar=True,
                sync_dist=True,
            )
            v.reset()

    def on_train_epoch_end(self):
        if self.gif is not None:
            self.gif = None

    def training_step(self, x, batch_idx: int):  # type: ignore
        # if training with images
        if len(x.shape) == 4:
            x = rearrange(x, "b c h w -> b 1 c h w")

        loss, attn_maps = self.forward(x)

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.log_metrics()

        if (
            isinstance(self.logger, pl.pytorch.loggers.WandbLogger)  # type: ignore
            and self.trainer.is_global_zero
            and self.gif is None
        ):
            with torch.no_grad():
                attn_plots = plot_attention_hierarchical(
                    imgs=x,
                    attn_maps=attn_maps,
                    res=self.resolution,
                    patch_size=self.patch_size,
                )

            log_dict = {}
            for i, attn_plot in enumerate(attn_plots):
                log_dict[f"attention_{i}"] = wandb.Video(
                    (attn_plot.cpu().numpy() * 255).astype("uint8"),
                    fps=self.log_config["fps"],
                    format="gif",
                )
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
