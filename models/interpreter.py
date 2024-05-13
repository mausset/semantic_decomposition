import lightning as pl
import timm
import torch
from models.decoder import TransformerDecoder
from models.positional_encoding import FourierScaffold
from models.slot_attention import build_slot_attention
from torch.optim import AdamW
from utils.plot import plot_attention_hierarchical
from utils.helpers import pad_batched_slots

from einops import repeat


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
        share_pos_enc=(False, True),
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
        self.pos_enc = FourierScaffold(in_dim=2, out_dim=dim)

        if share_pos_enc[0]:
            slot_attention_args["sampler"] = self.pos_enc
        if share_pos_enc[1]:
            feature_decoder_args["pos_enc"] = self.pos_enc

        self.slot_attention = build_slot_attention(
            slot_attention_arch, slot_attention_args
        )

        self.decoder = TransformerDecoder(**feature_decoder_args)

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

        self.share_pos_enc = share_pos_enc
        self.n_slots = n_slots
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    # Shorthand
    def forward_features(self, x):
        return self.image_encoder.forward_features(x)[:, self.discard_tokens :]

    def sample_coordinates(self, x, n_slots):
        return torch.rand(x.shape[0], n_slots, 2, device=x.device)

    def calculate_loss(self, up, down):
        """
        args:
            up: dict, up features
            down: dict, down features

        returns:
            loss, dict
        """

        losses = {}
        if self.loss_strategy == "anchored":
            decoded_features = down[self.n_slots[0]]
            features = up[decoded_features.shape[1]]
            decoded_chunked = torch.chunk(decoded_features, len(self.n_slots), dim=0)

            for n, chunk in zip(self.n_slots, decoded_chunked):
                loss = self.loss_fn(chunk, features)
                losses[n] = loss

        if self.loss_strategy == "local":
            for k, v in down.items():
                loss = self.loss_fn(v, up[v.shape[1]])
                losses[k] = loss

        return losses

    def decode(self, slots_list):

        down = {}

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
                decoded, _ = self.decoder(
                    slots,
                    resolution=res,
                )
                down[self.n_slots[i]] = decoded
                if self.loss_strategy == "anchored":
                    slots_list[i - 1] = torch.cat([slots_list[i - 1], decoded], dim=0)

            decoded, _ = self.decoder(slots_list[0], self.feature_resolution)
            down[self.n_slots[0]] = decoded

        return down

    def common_step(self, x):

        features = self.forward_features(x)
        slots = features

        up = {slots.shape[1]: slots}

        attn_list = []
        slots_list = []
        for n in self.n_slots:
            slots, attn_map = self.slot_attention(slots, n_slots=n)
            if attn_list:
                attn_map = attn_list[-1] @ attn_map
            attn_list.append(attn_map)
            slots_list.append(slots)
            up[n] = slots

        down = self.decode(slots_list)
        losses = self.calculate_loss(up, down)

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
