# Baseline results log

## 2026-09-30: small CPU baseline (`configs/baseline_cpu_small.yaml`)

200 random VMs of the 1,909-VM subset, 1 day of input, 1 hour ahead (12 steps), quantile
PatchTST with d_model 64, 2 layers (208k params), 8 epochs x 60k windows, ~70 s/epoch on
12 CPU threads. Test = days 21-30, windows every hour (stride 12), gap-filled targets skipped.

| Metric (CPU %) | Quantile PatchTST (median) | Persistence | Seasonal naive (1 day) |
|---|---|---|---|
| MAE | **1.93** | 2.83 | 3.26 |
| RMSE | **6.48** | 7.63 | 9.76 |

| Interval | Target | Empirical coverage | Mean width |
|---|---|---|---|
| 5-95% | 90% | 88.2% | 7.9 |
| 25-75% | 50% | 48.2% | 2.4 |

The baseline is slightly under-covered on the test period even without a targeted shift, and
coverage at h=1 swings between ~81% and ~94% over time (`outputs/baseline_cpu_small/coverage_over_time_test_s12.png`).
That is the gap the online conformal wrapper should close.

Notes for the conformal phase:
- RevIN re-normalises every input window, so pure *level* shifts are partly absorbed by the
  baseline itself (seen on synthetic data: coverage dips at the shift and recovers within
  about a day). Variance shifts and spikes are where fixed intervals should fail hardest;
  the injected-shift experiments should include both kinds.
- The venv has NumPy 2.x: when vendoring `conformal-time-series/core/methods.py`, replace
  `np.infty` with `np.inf`.
- Run `evaluate --stride 1` to get rolling forecasts at every step before replaying conformal methods.
