from io import BytesIO

import lightning as pl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import torch
import wandb
from einops import rearrange, repeat
from geomloss import SamplesLoss
from models.decoder import TransformerDecoderV1
from models.slot_attention import SA
from PIL import Image
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn
from torch.optim.adamw import AdamW
from torchmetrics.aggregation import MeanMetric
from utils.plot import plot_attention_hierarchical
from utils.schedulers import LinearSchedule, WarmupCosineSchedule
from x_transformers import Encoder

matplotlib.use("Agg")


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
            **config.get("sa_kwargs", {}),
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

        self.decoder = TransformerDecoderV1(
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

    def forward(self, x, seq_mask=None):
        b, t, n, _ = x.shape

        assert t % self.time_shrink == 0, "Time shrink must divide sequence length"

        mask = None
        if seq_mask is not None:
            mask = repeat(seq_mask, "b t -> b t n", n=n)
            mask = self.shrink_time(mask)
            mask = rearrange(mask, "b t n -> (b t) n")

        x = self.shrink_time(x, add_pe=True)
        x = rearrange(x, "b t ... -> (b t) ...")
        if self.encoder is not None:
            x = self.encoder(x, mask=mask)

        ret = self.slot_attention(x, context_mask=mask)
        ret["slots"] = rearrange(ret["slots"], "(b t) ... -> b t ...", b=b)
        ret["attn_map"] = rearrange(ret["attn_map"], "(b t) ... -> b t ...", b=b)

        return ret

    def decode(self, x, context_mask=None):
        b, *_ = x.shape

        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.decoder(x, context_mask=context_mask)
        x = rearrange(
            x,
            "(b t) (s n) d -> b (t s) n d",
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

    def train_only_last(self, x, seq_mask=None):
        b = x.shape[0]

        attn_maps = []
        extras = []
        with torch.no_grad():
            x = rearrange(x, "b t ... -> (b t) ...")
            x = self.base(x)
            x = rearrange(x, "(b t) ... -> b t ...", b=b)
            for block in self.blocks[:-1]:  # type: ignore
                ret = block(x, seq_mask=seq_mask)
                attn_maps.append(ret["attn_map"])
                extras.append(ret)
                x = ret["slots"].detach()

        features = [x]
        ret = self.blocks[-1](x, seq_mask=seq_mask)

        attn_maps.append(ret["attn_map"])
        extras.append(ret)
        decoded = [self.blocks[-1].decode(x, context_mask=ret.get("mask"))]

        return decoded, features, attn_maps, extras

    def forward(self, x, seq_mask=None, current_block=None):
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
        extras = []
        for block in blocks:  # type: ignore
            ret = block(x, seq_mask=seq_mask)
            features.append(ret["slots"])
            attn_maps.append(ret["attn_map"])
            extras.append(ret)
            decoded.append(block.decode(features[-1], context_mask=ret.get("mask")))
            x = ret["slots"].detach()

        return decoded, features, attn_maps, extras


class InterpreterTrainer(pl.LightningModule):

    def __init__(
        self,
        base_config: dict,
        block_configs: list[dict],
        optimizer_config: dict = {},
        log_config: dict = {},
        layer_schedule: dict | None = {},
        only_last: bool = False,
        loss: str = "sinkhorn",
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

        if loss == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.5)
        elif loss == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss: {loss}")

        self.metric_loggers = nn.ModuleDict(
            {f"loss_{i}": MeanMetric() for i in range(self.n_blocks)}
        )

        self.gif = None
        self.slot_counts = torch.zeros(17).to("cuda")

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

    def update_slot_counts(self, attn):
        max_idx = attn.argmax(dim=-1)
        active = [len(set(row.tolist())) for row in max_idx]
        self.slot_counts += torch.bincount(
            torch.tensor(active, device="cuda"), minlength=17
        ).detach()

    def forward(self, x, seq_mask=None, stage="train"):

        if self.only_last:
            decoded, target, attn_maps, extras = self.model.train_only_last(x)
        else:
            decoded, target, attn_maps, extras = self.model.forward(x, seq_mask=seq_mask, current_block=self.current_block)  # type: ignore

        loss = 0
        start = len(self.model.blocks) - 1 if self.only_last else 0
        for i, (d, t, a, _) in enumerate(
            zip(decoded, target, attn_maps, extras), start=start
        ):
            d = rearrange(d, "b t ... -> (b t) ...")
            t = rearrange(t, "b t ... -> (b t) ...")

            if seq_mask is not None:
                mask = rearrange(seq_mask, "b t -> (b t)")
                d = d[mask]
                t = t[mask]

            local_loss = self.loss_fn(d, t.detach()).mean()
            self.metric_loggers[f"loss_{i}"](local_loss.detach())

            self.update_slot_counts(a)
            print(self.slot_counts / self.slot_counts.sum().tolist())

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

        if isinstance(self.logger, pl.pytorch.loggers.WandbLogger) and self.slot_counts.sum() > 0:  # type: ignore
            plt.figure(figsize=(10, 6))
            self.slot_counts = self.slot_counts / self.slot_counts.sum()
            sns.barplot(
                x=list(range(17)),
                y=self.slot_counts.cpu().numpy(),
            )
            plt.yscale("log")
            plt.ylim(1e-4, 1)

            plt.xlabel("# Slots")
            plt.ylabel("Count")
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)

            img = Image.open(buf)
            self.logger.experiment.log(  # type: ignore
                {
                    "slot_counts": wandb.Image(img),
                }
            )

        self.slot_counts = torch.zeros(17, device="cuda")
        plt.close()

    def training_step(self, x, batch_idx: int):  # type: ignore
        # if training with images

        seq_mask = x["sequence_mask"]
        x = x["frames"]

        loss, attn_maps = self.forward(x, seq_mask)

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
