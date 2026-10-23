# Open-source implementations to adapt

Reviewed 2026-09-30 by cloning each repository. BibTeX entries are in `../CITATIONS.bib`.

## Decision

| Role | Use | Why |
|---|---|---|
| Forecaster backbone | **PatchTST, official repo** (`yuqinie98/PatchTST`, Apache-2.0) | Strong, simple, widely cited transformer baseline; channel-independent patching suits many univariate VM series; small self-contained backbone (3 files, ~560 lines) that vendors cleanly. |
| Online conformal methods | **`aangelopoulos/conformal-time-series`** (MIT) | Reference implementations of ACI, quantile tracking (P control), and Conformal PID (the NeurIPS 2023 paper), all written as functions over a *score sequence*, so any forecaster plugs in. |
| Cross-check only | MAPIE `TimeSeriesRegressor` (BSD-3) | Has EnbPI and ACI, but wraps scikit-learn estimators and refits ensembles, which is awkward for a trained torch model. Useful to sanity-check our ACI numbers on a small case. |
| Secondary baseline (optional) | Time-Series-Library (`thuml`, MIT) vanilla `Transformer` / `iTransformer` | If reviewers want a non-patch transformer, TSLib's models share the same `[B, L, C] -> [B, H, C]` interface and can take the same quantile head. |
| Reference only | `isgibbs/AdaptiveConformal` (R, no license file) | Original ACI experiments; read for algorithm details, do not copy code (no license). |

## What we adapted and how (already in the code)

**PatchTST** (`src/msml612/models/patchtst/`): `PatchTST_backbone.py`, `PatchTST_layers.py`, `RevIN.py` are copied verbatim from `PatchTST_supervised/layers/` at commit `204c21e`, with only the two import lines made relative. Apache-2.0 `LICENSE` copied alongside. Our changes are all in `src/msml612/models/quantile_patchtst.py`:

- the flatten head outputs `pred_len x n_quantiles` instead of `pred_len`;
- quantiles are made non-crossing (median + cumulative softplus offsets);
- RevIN (Kim et al., ICLR 2022) is applied outside the backbone in non-affine mode, so the monotone transform happens in normalised space and de-normalisation keeps quantile order;
- trained with pinball (quantile) loss instead of MSE.

**Conformal** (next phase, not yet written): the plan is to wrap the baseline's own 5%/95% quantiles with conformalized quantile regression scores (Romano et al., NeurIPS 2019), `s_t = max(lo_t - y_t, y_t - hi_t)`, and update the offset online with the methods in `conformal-time-series/core/methods.py`: `aci` (Gibbs & Candès 2021), `quantile` (quantile tracker / P control), and `quantile_integrator_log` (PID without scorecaster). Those functions consume a 1-D score array per series and return `q_t`; the interval is `[lo_t - q_t, hi_t + q_t]`. We would vendor them the same way (MIT, keep header), fixing `np.infty` which NumPy 2 removed (10 uses in `methods.py`), and run one sequence per VM per horizon, using `ahead=h` for h-step forecasts. `evaluate.py` already saves the rolling test forecasts (`predictions_test_s1.npz`) in exactly that form.

## Candidates considered and not chosen

- **Informer / Autoformer / FEDformer** (in TSLib and the PatchTST repo's `Formers/`): heavier encoder-decoder models that PatchTST beats on standard benchmarks; no benefit for this question.
- **Temporal Fusion Transformer** (pytorch-forecasting): has native quantile output, but brings a large framework (Lightning, TimeSeriesDataSet) that is harder to adapt and cite cleanly.
- **Chronos / TimesFM / Moirai foundation models**: interesting as a later extension ("does conformal help a foundation model too?") but would change the baseline story and the compute budget.
- **EnbPI (Xu & Xie 2021)**: needs bootstrap ensembles of the forecaster; expensive for a transformer. Worth one comparison row if time allows.
- **DtACI (Gibbs & Candès, JMLR 2024)**: removes ACI's step-size choice; not in the chosen repo but a short algorithm we could implement if ACI's step size turns out to matter.
