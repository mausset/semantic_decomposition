import lightning as pl
from einops import repeat
from torch import nn
from torch.nn import functional as F
from models.positional_encoding import FourierScaffold
from x_transformers import Encoder
from models.components import GaussianPrior


class TransformerDecoder(pl.LightningModule):

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


class TransformerDecoderIterative(pl.LightningModule):

    def __init__(self, dim, depth, n_iters=5) -> None:
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.n_iters = n_iters

        self.prior = GaussianPrior(dim)

        self.transformer = Encoder(
            dim=dim,
            depth=1,
            cross_attend=True,
            ff_glu=True,
            ff_swish=True,
        )

    def forward(self, x, resolution, mask=None, context_mask=None):
        """
        args:
            x: (B, N, D), extracted object representations
        returns:
            (B, HW, D), decoded features
            (B, HW, N), cross-attention map
        """

        sample = self.prior(x, resolution[0] * resolution[1])
        target = sample

        for _ in range(self.n_iters):
            result, hiddens = self.transformer(
                target,
                mask=mask,
                context=x,
                context_mask=context_mask,
                return_hiddens=True,
            )
            target = result

        target = target.detach() - sample.detach() + sample
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
