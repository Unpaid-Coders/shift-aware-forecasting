"""Online conformal methods on conformity scores, vectorised over many sequences.

Adapted from aangelopoulos/conformal-time-series, core/methods.py at commit b729c3f
(https://github.com/aangelopoulos/conformal-time-series), MIT License,
Copyright (c) 2023 Anastasios Angelopoulos. Methods:

  trail    trailing-window split conformal (the reference's `trailing_window`; here ACI with lr=0)
  aci      adaptive conformal inference, Gibbs & Candes (NeurIPS 2021); the reference's `aci_clipped`
  qt       quantile tracker, P control (the reference's `quantile`)
  pid      quantile tracker + saturated log integrator, i.e. conformal PID control without the
           scorecaster, Angelopoulos, Candes & Tibshirani (NeurIPS 2023)
           (the reference's `quantile_integrator_log`)

All methods return an offset q_t per step. With CQR scores (Romano et al., NeurIPS 2019)
s_t = max(lo_t - y_t, y_t - hi_t), the calibrated interval is [lo_t - q_t, hi_t + q_t] and it
covers y_t exactly when s_t <= q_t.

Changes from the reference implementation:
  * One call runs a batch of S sequences (one per VM) in a single time loop, instead of a Python
    loop per sequence, and runs every method/config in the same pass.
  * Prediction k may use only scores up to index k - ahead (an h-step forecast has ahead = h).
    The reference scales the quantile-tracker step size with scores up to index k; here the
    scale uses only scores observed by prediction time as well.
  * Before any score is observed (and, for ACI/trail, before ceil(1/alpha) scores), q = 0, i.e. the
    base model's own interval. The reference uses an infinite interval or a burn-in period.
  * ACI and trail use a trailing window of `window` scores (the reference default is unbounded).
  * The step size of qt/pid and the integrator gain of pid are multiplied by the range of the
    trailing window of scores (the reference does this for the step size only), so one set of
    hyperparameters works across VMs whose scores differ in scale by orders of magnitude.
  * Reported offsets are clipped to [lower, max of trailing window]: `lower` = -(hi - lo)/2 keeps
    the interval from crossing, and the upper clip replaces infinite quantiles (the reference's
    `aci_clipped`; for pid it bounds the tan() blow-up). The internal state is not clipped.
    Feedback (covered or not) uses the reported, clipped interval.
  * np.infty (removed in NumPy 2) is not used.
"""
from dataclasses import dataclass

import numpy as np

METHOD_LABELS = {"base": "Base quantiles", "trail": "Trailing split CP", "aci": "ACI",
                 "qt": "Quantile tracker", "pid": "Conformal PID"}


@dataclass(frozen=True)
class MethodConfig:
    method: str            # trail | aci | qt | pid
    lr: float = 0.0        # ACI: step on alpha_t; qt/pid: step on q as a fraction of the score range
    ki: float = 0.0        # pid integrator gain, as a fraction of the score range
    csat: float = 1.0      # pid saturation constant (reference default)

    @property
    def name(self) -> str:
        if self.method == "trail":
            return "trail"
        if self.method == "pid":
            return f"pid_lr{self.lr:g}_ki{self.ki:g}"
        return f"{self.method}_lr{self.lr:g}"


def online_offsets(scores: np.ndarray, alpha: float, ahead: int, configs: list,
                   window: int = 288, lower: np.ndarray | None = None) -> dict:
    """scores [S, T] in time order -> {config.name: q [S, T] float32}.

    q[:, k] is the offset for prediction k and depends only on scores[:, :k - ahead + 1].
    lower [S, T] (optional) is the smallest allowed offset per step.
    """
    S, T = scores.shape
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(S)
    n_min = int(np.ceil(1 / alpha))
    out = {c.name: np.zeros((S, T), np.float32) for c in configs}
    alpha_t = {c.name: np.full(S, alpha) for c in configs if c.method in ("aci", "trail")}
    qt = {c.name: np.zeros(S) for c in configs if c.method in ("qt", "pid")}
    n_err = {c.name: np.zeros(S) for c in configs if c.method == "pid"}

    for k in range(T):
        p = k - ahead  # newest score observed when prediction k is made
        if p < 0:
            continue   # q stays 0 (base interval)
        s_p = scores[:, p]
        lo_w = max(0, p - window + 1)
        srt = np.sort(scores[:, lo_w:p + 1], axis=1)
        n = p + 1 - lo_w
        smax = srt[:, -1]
        scale = smax - srt[:, 0]                      # trailing score range
        lb = lower[:, k] if lower is not None else -np.inf
        for c in configs:
            q_out = out[c.name]
            err = (s_p > q_out[:, p]).astype(np.float64)  # 1 = prediction p missed
            if c.method in ("aci", "trail"):
                a = alpha_t[c.name]
                if c.method == "aci":
                    a += c.lr * (alpha - err)
                if n < n_min:
                    continue
                # 'higher' empirical quantile at level 1 - alpha_t; alpha_t <= 0 gives the max (clipped ACI)
                idx = np.ceil((n - 1) * np.clip(1 - a, 0, 1)).astype(int)
                q = srt[rows, idx]
            else:
                x = qt[c.name]
                x += c.lr * scale * (err - alpha)
                q = x.copy()
                if c.method == "pid":
                    e = n_err[c.name]
                    e += err
                    m = p + 1  # number of observed errors
                    arg = (e - m * alpha) * np.log(m + 1) / (c.csat * (m + 1))
                    q += c.ki * scale * np.tan(np.clip(arg, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6))
            q_out[:, k] = np.maximum(np.minimum(q, smax), lb)  # lower bound wins
    return out


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Conformalized quantile regression score; <= 0 when y is inside [lo, hi]."""
    return np.maximum(lo - y, y - hi)
