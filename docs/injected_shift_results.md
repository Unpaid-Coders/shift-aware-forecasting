# Injected-shift results

Zaratan job 23995554 (`slurm/injected_shifts.sbatch`, 2026-11-11, 32 min on one A100). Every one of the
1,909 VMs gets one shift at a random 5-minute step in test days 23-26 (`data/inject.py`): **level**
(+3 training std), **scale** (deviations from the pre-shift day mean x3) or **level_scale** (both).
The baseline checkpoint is unchanged. The conformal settings are the ones selected on the clean data
(`outputs/conformal_zaratan/selected.json`: ACI lr 0.001, QT lr 0.005, PID lr 0.01 / ki 0.05), so
nothing is tuned on the shifted data. Metrics use days 22-30 (day 21 is the conformal burn-in).
Full numbers: `python scripts/shift_report.py` (saved on Zaratan as `logs/shift_report.txt`).

## 1. Base forecaster: level shifts are absorbed, variance shifts are not

| Data | Median MAE | RMSE | 90% cov. / width | 80% cov. | 50% cov. |
|---|---|---|---|---|---|
| clean | 1.44 | 5.16 | 0.850 / 5.30 | 0.742 | 0.444 |
| level | 1.33 | 4.44 | 0.856 / 4.99 | 0.748 | 0.448 |
| scale | 2.57 | 8.29 | 0.823 / 8.79 | 0.711 | 0.422 |
| level + scale | 2.47 | 7.25 | 0.851 / 8.95 | 0.743 | 0.449 |

RevIN re-normalises each 2-day input window, so a level shift barely affects overall accuracy or
coverage. A variance shift nearly doubles the error and is the only kind that lowers coverage over
the whole test period. (Level MAE is slightly *below* clean, probably because shifted values are
clipped at 100% CPU and spread over fewer distinct values. This is worth a footnote, not a claim.)

## 2. Coverage around the shift (90% interval, horizon 1, all 1,909 VMs)

Hours since the shift, coverage / mean width:

**Variance shift (scale)**

| method | day before | 0-1 h | 1-3 h | 3-6 h | 6-12 h | 12-24 h | 24-48 h | 48-72 h |
|---|---|---|---|---|---|---|---|---|
| base | 0.863 / 4.3 | 0.363 | 0.489 | 0.558 | 0.631 | 0.715 | 0.832 | 0.864 / 10.3 |
| trail | 0.897 | 0.391 | 0.557 | 0.717 | 0.855 | 0.918 | 0.929 | 0.905 |
| ACI | 0.898 | 0.396 | 0.587 | 0.776 | 0.884 | 0.931 / 14.2 | 0.936 | 0.909 |
| QT | 0.899 | 0.449 | 0.679 | 0.801 | 0.878 | 0.908 / 12.4 | 0.915 | 0.904 |
| PID | 0.899 / 4.8 | **0.503** | **0.752** | **0.857** | **0.901** | 0.911 / 12.8 | 0.908 | 0.901 / 11.5 |

- The base intervals take about two days, the length of the input window, to return to their usual
  coverage. For the first 12 hours they cover 36-63% instead of 90%.
- PID is back at target 6-12 h after the shift. QT is close behind. ACI and the trailing window are
  slower in the first 6 h and then **overshoot** (93-94% for the next day and a half, with wider
  intervals than PID), because their one-day score window remembers the transient.
- The same ordering holds at 80% and 50%. At 50%, h=1, 6-12 h after the shift: base 0.27, trail
  0.33, ACI 0.35, QT 0.50, PID 0.51.
- In the day after the shift (all horizons), 90% coverage is: base 0.630, trail 0.817, ACI 0.844,
  QT 0.842, PID 0.866. Widths are 9.7 / 13.6 / 15.1 / 13.5 / 14.7, so QT matches ACI's coverage with
  about 10% less width, and PID gets the most coverage for less width than ACI.

**Level shift**: every method drops to about 0.74-0.77 in the first hour (h=1). Coverage is then
*above* target for 1-6 h, because the jump inside the input window inflates RevIN's std and widens
the base quantiles (width 13 at 0-1 h, against 4.3 before). After that, coverage is near normal.
Conformal makes little difference to recovery here. At h=12, trail and ACI under-cover 24-48 h after
the shift (80% interval: 0.72 and 0.74), after learning from the over-covered stretch. QT and PID
stay at 0.80.

**Level + variance**: this sits in between. Base h=1 coverage is 0.62 in the first hour and 0.82-0.84
for hours 1-12. PID is at 0.88-0.91 from 1 h on, QT at 0.87-0.90, and ACI and trail at 0.85-0.89.

**Unavoidable dip at long horizons**: forecasts whose target falls in the first hour after the
shift were issued before it. At h=12, 0-1 h coverage is 0.03-0.10 for *every* method, including the
base, for level shifts. No online method can fix this, so the paper should measure recovery from
the first forecast issued after the shift, or report h=1 only.

## 3. Whole test period (90% interval)

| Data | base | trail | ACI | QT | PID |
|---|---|---|---|---|---|
| clean, coverage / width | 0.851 / 5.28 | 0.897 / 5.86 | 0.900 / 5.97 | 0.900 / 5.95 | 0.900 / 6.19 |
| clean, local cov. error | 0.062 | 0.020 | 0.019 | 0.013 | **0.011** |
| scale, coverage | 0.820 | 0.892 | 0.899 | 0.896 | 0.897 |
| scale, local cov. error | 0.090 | 0.030 | 0.028 | 0.020 | **0.015** |
| scale, VM-days >10 pp under | 33.5% | 5.0% | 1.9% | 1.8% | **0.3%** |

All four methods hit the target on average. The differences show up in *local* (per VM, per day)
coverage, where PID is best everywhere and QT is second.

## 4. Natural shifts (70 Azure VMs, clean run)

Base coverage barely moves across these shifts (day before 0.857, day after 0.854), and the hourly
traces are noisy with no clear dip. The natural shifts in the trace are mild *for this model*, so
they show the calibration fix (0.86 -> 0.90, local error 0.06 -> 0.01) but not shift recovery.

## What this means for the paper

1. **The recovery story is variance shifts.** The base model handles level shifts by design
   (RevIN), so a variance shift is where fixed quantile intervals fail for about two days and
   online conformal recovers in hours. That is the headline figure:
   `outputs/conformal_shift_scale/fig_shift_aligned_90_h1.png`.
2. **Level shifts are the honest control.** The architecture already absorbs them, so conformal
   adds little there. Reporting this explains *why* the model needs help with variance and not with level.
3. **Natural shifts support calibration, not recovery.** On the real trace, the claim is that
   conformal removes the steady under-coverage and cuts local coverage error by 3-6x, with no
   loss in shift windows.
4. **Method ranking: PID > QT > ACI > trail.** PID has the fastest recovery, no overshoot and
   the lowest local error. The cost is about 4% more average width than QT/ACI on clean data.
5. **Caveat on ACI:** selection on the clean data picked the smallest step size (0.001), which is
   the slowest to react. A fair comparison should show ACI's sweep on the shifted data (already in
   `outputs/conformal_shift_*/table_90.csv`), so a reviewer can't say ACI was set up to lose.
6. Shift size is a single setting (3x). A smaller variance shift (for example 1.5x) would show
   whether the gap holds when the shift is less extreme.
