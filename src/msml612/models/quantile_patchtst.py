"""PatchTST with a multi-quantile head.

Adapted from the official PatchTST implementation (Nie et al., ICLR 2023,
https://github.com/yuqinie98/PatchTST, Apache-2.0). The backbone files in ./patchtst are
copied unmodified except for relative imports. Our changes live here:
  * the flatten head predicts H x Q values instead of H (one per quantile level);
  * outputs are made non-crossing: the median is predicted directly and the other
    quantiles as cumulative softplus offsets above / below it;
  * RevIN (non-affine) is applied here rather than inside the backbone, so the
    monotone transform happens in normalised space and de-normalisation, a positive
    affine map, preserves quantile order.
"""
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .patchtst.PatchTST_backbone import PatchTST_backbone
from .patchtst.RevIN import RevIN


class QuantilePatchTST(nn.Module):
    def __init__(self, c_in: int, seq_len: int, pred_len: int, quantiles: Sequence[float],
                 patch_len: int = 16, stride: int = 8, n_layers: int = 3, d_model: int = 128,
                 n_heads: int = 16, d_ff: int = 256, dropout: float = 0.2, fc_dropout: float = 0.2,
                 head_dropout: float = 0.0, target_ch: int = -1, revin: bool = True):
        super().__init__()
        qs = sorted(quantiles)
        assert 0.5 in qs, "quantile grid must include the median"
        self.register_buffer("quantiles", torch.tensor(qs))
        self.mid = qs.index(0.5)
        self.pred_len, self.n_q, self.target_ch = pred_len, len(qs), target_ch
        self.backbone = PatchTST_backbone(
            c_in=c_in, context_window=seq_len, target_window=pred_len * len(qs),
            patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout,
            head_dropout=head_dropout, padding_patch="end", revin=False)
        self.revin = RevIN(c_in, affine=False) if revin else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] -> quantile forecasts of the target channel, [B, H, Q]."""
        if self.revin is not None:
            x = self.revin(x, "norm")
        z = self.backbone(x.permute(0, 2, 1))                    # [B, C, H*Q]
        z = z.reshape(z.shape[0], z.shape[1], self.pred_len, self.n_q)
        z = self.monotone(z, self.mid)                           # [B, C, H, Q], non-crossing
        if self.revin is not None:
            B, C = z.shape[:2]
            z = self.revin(z.permute(0, 2, 3, 1).reshape(B, -1, C), "denorm")
            z = z.reshape(B, self.pred_len, self.n_q, C).permute(0, 3, 1, 2)
        return z[:, self.target_ch]                              # [B, H, Q]

    @staticmethod
    def monotone(raw: torch.Tensor, mid: int) -> torch.Tensor:
        med = raw[..., mid:mid + 1]
        up = med + torch.cumsum(F.softplus(raw[..., mid + 1:]), -1)
        down = med - torch.cumsum(F.softplus(raw[..., :mid]).flip(-1), -1).flip(-1)
        return torch.cat([down, med, up], -1)
