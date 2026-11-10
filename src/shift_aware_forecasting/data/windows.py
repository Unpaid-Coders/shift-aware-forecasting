"""Sliding-window datasets over the dense Azure CPU array built by scripts/build_dataset.py.

Splits are chronological and shared by all VMs (a global model is trained across VMs):
a window belongs to the split that contains its first *target* step, and its input may
reach back into the previous split (standard practice for forecasting evaluation).
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

STEPS_PER_DAY = 288  # 5-minute readings
CHANNEL_INDEX = {"min_cpu": 0, "max_cpu": 1, "avg_cpu": 2}


@dataclass
class SplitSpec:
    train_days: float = 18
    val_days: float = 3  # the rest is test

    def bounds(self, T: int) -> dict:
        a = int(self.train_days * STEPS_PER_DAY)
        b = int((self.train_days + self.val_days) * STEPS_PER_DAY)
        return {"train": (0, a), "val": (a, b), "test": (b, T)}


def load_npz(path: str, channels=("avg_cpu",), vm_limit: int = 0, seed: int = 0) -> dict:
    d = np.load(path, allow_pickle=False)
    idx = np.arange(d["cpu"].shape[0])
    if vm_limit and vm_limit < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, vm_limit, replace=False))
    ch = [CHANNEL_INDEX[c] for c in channels]
    return {
        "x": d["cpu"][idx][..., ch].astype(np.float32) / 100.0,  # CPU % -> [0, 1]
        "observed": d["observed"][idx],
        "vm_ids": d["vm_ids"][idx],
        "category": d["category"][idx],
    }


class WindowDataset(Dataset):
    """Windows (input [L, C], target [H]) where the target is channel `target_ch`."""

    def __init__(self, x: np.ndarray, split: tuple, seq_len: int, pred_len: int,
                 stride: int = 1, target_ch: int = -1):
        self.x = torch.from_numpy(x)
        self.L, self.H, self.target_ch = seq_len, pred_len, target_ch
        lo, hi = split
        first_target = np.arange(max(lo, seq_len), hi - pred_len + 1, stride)
        n_vm = x.shape[0]
        self.index = np.stack(np.meshgrid(np.arange(n_vm), first_target, indexing="ij"), -1).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        v, t = self.index[i]
        inp = self.x[v, t - self.L:t]
        tgt = self.x[v, t:t + self.H, self.target_ch]
        return inp, tgt, torch.tensor([v, t])
