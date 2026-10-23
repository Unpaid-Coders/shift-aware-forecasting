"""Losses and evaluation metrics for quantile forecasts."""
import numpy as np
import torch


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """pred [B, H, Q], y [B, H], quantiles [Q] -> mean pinball (quantile) loss."""
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()


def interval_metrics(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, mask=None) -> dict:
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    if mask is not None:
        covered, width = covered[mask], width[mask]
    return {"coverage": float(covered.mean()), "width": float(width.mean())}


def point_metrics(yhat: np.ndarray, y: np.ndarray, mask=None) -> dict:
    err = yhat - y
    if mask is not None:
        err = err[mask]
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def rolling_coverage(covered: np.ndarray, window: int) -> np.ndarray:
    """covered [..., T] bool -> trailing-window coverage rate along the last axis."""
    c = np.cumsum(covered.astype(float), -1)
    out = np.full(covered.shape, np.nan)
    out[..., window - 1:] = c[..., window - 1:] - np.concatenate(
        [np.zeros(covered.shape[:-1] + (1,)), c[..., :-window]], -1)
    return out / window
