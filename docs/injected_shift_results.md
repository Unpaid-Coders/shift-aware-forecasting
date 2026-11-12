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

**Variance shift (scale)**, with each method at its *selected* setting (ACI lr 0.001, QT lr 0.005,
PID lr 0.01 / ki 0.05). These step sizes differ, so this table compares settings, not algorithms.
Section 3 compares the methods at matched step size and width.

| method | day before | 0-1 h | 1-3 h | 3-6 h | 6-12 h | 12-24 h | 24-48 h | 48-72 h |
|---|---|---|---|---|---|---|---|---|
| base | 0.863 / 4.3 | 0.363 | 0.489 | 0.558 | 0.631 | 0.715 | 0.832 | 0.864 / 10.3 |
| trail | 0.897 | 0.391 | 0.557 | 0.717 | 0.855 | 0.918 | 0.929 | 0.905 |
| ACI | 0.898 | 0.396 | 0.587 | 0.776 | 0.884 | 0.931 / 14.2 | 0.936 | 0.909 |
| QT | 0.899 | 0.449 | 0.679 | 0.801 | 0.878 | 0.908 / 12.4 | 0.915 | 0.904 |
| PID | 0.899 / 4.8 | **0.503** | **0.752** | **0.857** | **0.901** | 0.911 / 12.8 | 0.908 | 0.901 / 11.5 |

- The base intervals take about two days, the length of the input window, to return to their usual
  coverage. For the first 12 hours they cover 36-63% instead of 90%.
- At these settings, PID is back at target 6-12 h after the shift, and QT is close behind. ACI
  (at its slowest step size) and the trailing window are slower in the first 6 h, then
  **overshoot** (93-94% for the next day and a half), because their one-day score window remembers
  the transient.
- The same ordering holds at 80% and 50%. At 50%, h=1, 6-12 h after the shift: base 0.27, trail
  0.33, ACI 0.35, QT 0.50, PID 0.51.
- In the day after the shift (all horizons), 90% coverage is: base 0.630, trail 0.817, ACI 0.844,
  QT 0.842, PID 0.866, with widths 9.7 / 13.6 / 15.1 / 13.5 / 14.7. Section 3 shows that larger ACI
  step sizes change this.

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

## 3. Whole hyperparameter sweep: no single winner

The selected settings above came from the clean data (smallest width within 1 pp of target), which
favours slow step sizes. The full sweep (`outputs/conformal_*/table_<level>.csv` and
`shift_table_<level>.csv`) compares the methods more fairly. Here is the 90% interval. "Width" is the
average over the whole test period, "local err" is the mean |daily per-VM coverage - 0.9|, and
"after 1 d" is coverage in the day after the shift.

| setting | clean width | clean local err | scale width | scale local err | scale after 1 d | scale width after 1 d |
|---|---|---|---|---|---|---|
| base | 5.28 | 0.062 | 9.20 | 0.090 | 0.630 | 9.7 |
| trail | 5.86 | 0.020 | 10.83 | 0.030 | 0.817 | 13.6 |
| ACI lr 0.001 | 5.97 | 0.019 | 11.32 | 0.028 | 0.844 | 15.1 |
| ACI lr 0.005 | 6.19 | 0.016 | 11.88 | 0.022 | 0.891 | 19.1 |
| ACI lr 0.01 | 6.50 | 0.013 | 12.40 | 0.017 | **0.904** | 21.3 |
| ACI lr 0.05 | 9.36 | 0.008 | 17.22 | 0.008 | 0.905 | 25.6 |
| QT lr 0.005 | 5.95 | 0.013 | 10.91 | 0.020 | 0.842 | 13.5 |
| QT lr 0.01 | 6.18 | **0.011** | 11.30 | **0.015** | 0.865 | 14.6 |
| QT lr 0.05 | 8.48 | 0.009 | 15.39 | 0.009 | 0.889 | 19.0 |
| QT lr 0.1 | 11.37 | 0.009 | 20.77 | 0.009 | 0.893 | 24.2 |
| PID lr 0.01, ki 0.05 | 6.19 | **0.011** | 11.33 | **0.015** | 0.866 | 14.7 |
| PID lr 0.01, ki 1 | 6.54 | 0.010 | 11.91 | 0.013 | 0.881 | 16.0 |
| PID lr 0.05, ki 0.05 | 8.50 | 0.009 | 15.42 | 0.009 | 0.889 | 19.0 |

What the sweep shows:

- **QT and PID are almost the same algorithm here.** At equal step size, they match within 0.001
  in coverage and within 1% in width, on clean and shifted data alike. The integrator only matters
  at ki = 1, where it buys about 1.5 pp more after-shift coverage for 5% more width. The earlier
  "PID beats QT" came from the selection picking QT at a smaller step size.
- **For even calibration, QT/PID beat ACI.** At matched average width, QT/PID have lower local
  error. On clean data at width about 6.2, local error is 0.011 for QT/PID and 0.016 for ACI. On the
  variance shift at width about 11.9, it is 0.013 for PID with ki 1 and 0.022 for ACI lr 0.005.
  To match QT/PID's local error, ACI needs about 10% more width.
- **For recovery right after a shift, ACI beats QT/PID.** ACI with lr 0.005-0.01 is back to
  about 0.90 coverage in the day after a variance shift. QT/PID never get above 0.893 in that day,
  even at step size 0.1 with nearly twice the width. At matched *overall* width (about 11.9), ACI
  lr 0.005 reaches 0.891 after the shift, against 0.881 for PID with ki 1. ACI gets there by
  widening sharply after the shift (19-21 against 15-16), because it moves the quantile *level*
  of a trailing score window. QT/PID move the offset in score units, with a step scaled by the
  previous day's score range, so their steps are too small right after a variance jump.
- The same pattern holds at 80% and for level + variance shifts. For level + variance at 90%,
  ACI lr 0.005 reaches 0.903 after the shift at width 11.43, against 0.886 for QT lr 0.01 at 10.99.
- **The trailing window is dominated by everything else.**

Figure: `outputs/frontier/frontier_90.png` and `frontier_80.png`, from
`python -m shift_aware_forecasting.conformal.frontier` (see its docstring for the command).

So there is a real trade-off. QT/PID spread their width evenly and give the most even coverage.
ACI with a larger step size spends width on the hours after a shift and recovers faster.
"Which method is best" depends on whether the goal is even calibration or fast recovery.

## 4. Natural shifts (70 Azure VMs, clean run)

Base coverage barely moves across these shifts (day before 0.857, day after 0.854), and the hourly
traces are noisy with no clear dip. The natural shifts in the trace are mild *for this model*, so
they show the calibration fix (0.86 -> 0.90, local error 0.06 -> 0.01) but not shift recovery.

## What this means for the paper

1. **The recovery story is variance shifts.** The base model handles level shifts by design
   (RevIN). A variance shift is where fixed quantile intervals fail for about two days and every
   online method recovers within hours to a day. That is the headline figure:
   `outputs/conformal_shift_scale/fig_shift_aligned_90_h1.png`.
2. **Level shifts are the honest control.** The architecture already absorbs them, so conformal
   adds little there.
3. **Natural shifts support calibration, not recovery.** On the real trace, conformal removes
   the steady under-coverage (0.85 -> 0.90 at 90%) and cuts local coverage error by 3-6x.
4. **Don't claim one best method. Present the trade-off.** QT ≈ PID give the most even
   coverage for a given width. ACI with a larger step size recovers fastest after a variance shift.
   The trailing window is dominated. The fair figure is a width-vs-error frontier over the
   whole sweep (local error against width, and after-shift coverage against width), not a table
   of one setting per method.
5. **The integrator adds little.** PID ≈ QT at equal step size, so the paper should not credit
   PID's integrator for the gains.
6. **Selection protocol matters.** Selecting on clean data picks slow settings. A cleaner
   protocol would select on a shift injected into the validation days (days 18-20) and then
   evaluate on the test days. That rule rewards recovery without tuning on the test shifts.
7. Shift size is a single setting (3x). A milder variance shift (for example 1.5x) would show
   whether the trade-off holds when the shift is less extreme.
