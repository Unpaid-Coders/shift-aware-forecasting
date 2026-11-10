# Baseline results log

## 2026-11-10: full baseline on Zaratan (`configs/baseline_zaratan.yaml`)

All 1,909 VMs, 2 days of input, 1 hour ahead (12 steps), paper-sized quantile PatchTST
(1.18M params), 30 epochs on one A100 (51 min; validation loss was still improving at the
last epoch). Test = days 21-30, windows every hour (stride 12; stride 1 gives nearly
identical numbers), gap-filled targets skipped.

| Metric (CPU %) | Quantile PatchTST (median) | Persistence | Seasonal naive (1 day) |
|---|---|---|---|
| MAE | **1.44** | 2.61 | 2.82 |
| RMSE | **5.17** | 6.97 | 8.31 |

| Interval | Target | Empirical coverage | Mean width |
|---|---|---|---|
| 5-95% | 90% | 85.0% | 5.29 |
| 10-90% | 80% | 74.0% | 3.90 |
| 25-75% | 50% | 44.3% | 1.97 |

Coverage is below target at every level and decreases with the horizon (90% interval:
86.3% at h=1, about 84% at h=12). Averaged over all VMs, h=1 coverage stays at 84-88%
throughout the test days, so the gap is steady rather than a visible shift-driven dip;
shift effects are analysed per VM in the conformal evaluation (`shift_aware_forecasting.conformal.run`).
Figures: `docs/figures/baseline_coverage_over_time_test_s1.png` and `_s12.png`.

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
- The conformal methods replay the rolling forecasts from `evaluate --stride 1`
  (`predictions_test_s1.npz`).
