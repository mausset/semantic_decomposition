import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from models.positional_encoding import FourierScaffold
from x_transformers import Encoder
from models.components import PE2D


class TransformerDecoderV1(pl.LightningModule):

    def __init__(self, dim, depth, pos_enc=None) -> None:
        super().__init__()

        self.dim = dim
        self.depth = depth

        if pos_enc is None:
            pos_enc = FourierScaffold(in_dim=2, out_dim=dim)
        self.pos_enc = pos_enc

        self.transformer = Encoder(
            dim=dim,
            depth=depth,
            cross_attend=True,
            ff_glu=True,
            ff_swish=True,
        )

    def forward(self, x, resolution, sample=None, mask=None, context_mask=None):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
            (B, HW, N), cross-attention map
        """

        target = self.pos_enc(x, resolution, sample=sample)

        result, hiddens = self.transformer(
            target,
            mask=mask,
            context=x,
            context_mask=context_mask,
            return_hiddens=True,
        )

        # Last cross-attention map, mean across heads
        attn_map = hiddens.attn_intermediates[-1].post_softmax_attn.mean(dim=1)

        return result, attn_map


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_args) -> None:
        super().__init__()

        self.pe = PE2D(decoder_args["dim"])
        self.transformer = Encoder(**decoder_args)

    def forward(self, x, resolution, mask=None, context_mask=None):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
        """

        target = self.pe(x, resolution)

        # target = torch.randn(
        #     x.shape[0], resolution[0] * resolution[1], x.shape[-1], device=x.device
        # )

        result = self.transformer(
            target,
            mask=mask,
            context=x,
            context_mask=context_mask,
        )

        return result
