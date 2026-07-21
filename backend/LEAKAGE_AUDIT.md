# Leakage Audit — AI4CM Forecast Pipelines

**Date**: 2026-04-01  
**Auditor**: Automated + manual code trace  
**Scope**: All four pipeline families (A stat, B ML, C DL, E quantile)  
**Verdict**: **PASS** — no data leakage found. One low-severity warning addressed.

---

## Methodology

For each pipeline, we trace the data flow from input to prediction and verify:
1. Train/test temporal separation is strict (no overlap)
2. Features are backward-looking only (no future data)
3. Models fit only on training data
4. Output semantics (origin_date, target_date, y_true) are correct
5. Prediction intervals use only training/calibration data

---

## A - Statistical Pipeline (run_a_stat.py)

| Check | Result | Lines | Details |
|-------|--------|-------|---------|
| Fold non-overlap | PASS | 97-112, 278-280 | Yearly folds: train ends Dec 31 of Y-1, test covers year Y. Zero overlap. |
| Training end date | PASS | 103-105 | `tr_end = idx[idx <= Timestamp(f"{Y-1}-12-31")][-1]` |
| Model fit on training only | PASS | 129-212 | All models (NAIVE, ETS, SARIMAX, STL_ARIMA, THETA) receive only `y_tr`. |
| Native PIs from training | PASS | 170-172, 187-188 | ETS/SARIMAX PIs via `get_forecast(n)` on training-fitted model. |
| origin_date = train_end | PASS | 299 | `origin_date: tr_end` — last training date. |
| y_true from test actuals | PASS | 289, 305 | `y_true = y_all.reindex(idx_te).values` — not available to model. |

---

## B - ML Pipeline (b_ml_pipeline.py)

| Check | Result | Lines | Details |
|-------|--------|-------|---------|
| Training boundary | PASS | 573-574 | `s.loc[s.index <= train_end]` — strict cutoff. |
| Target construction | PASS | 603 | `s_train_full.shift(-h)` only accesses within training series, last h rows become NaN. |
| Feature matrix (prediction) | PASS | 621-625 | Pre-computed `feat_all_origins` uses backward-looking ops only. Row at origin O depends only on data <= O. |
| Delta modeling | PASS | 606-607 | Both operands over same training index. Correct subtraction. |
| Conformal PI split | PASS | 631-639 | Val = last 20% of training (temporal, before test). |
| Multivariate exog ffill | PASS | 554 | Reindex+ffill on full series, but subsetted to training/origin indices at use-time. No leakage. |
| Fit-once-per-fold | PASS | 582-615 | Training data identical across all origins within fold (proven in comments 588-591). |
| Alignment verification | PASS | 751-791 | Post-hoc check on full index — verifying, not training. |

---

## C - DL Pipeline (c_dl_pipeline.py)

| Check | Result | Lines | Details |
|-------|--------|-------|---------|
| Sequence boundaries | PASS | 464-475 | Input `[end_i - seq_len + 1 : end_i + 1]`, label at `end_i + horizon`. No future peek. |
| Target masking at origin | PASS | 466-469, 740 | For stock targets, target column zeroed in last sequence row. Prevents lag-0 shortcut. |
| Calendar features | PASS | 131-164 | Date-derived only (dow, month, is_holiday). No windowing. |
| Exogenous features | PASS | 408-420 | Resampled with `.last()`/`.sum()`, no negative shifts. |
| Fold integrity | PASS | 747-762 | Masks built on label dates (`ld_all`). Train labels < test labels. |
| Conformal calibration | PASS | 777-780 | Tail of training data (20%), before test. No overlap. |
| Feature scaler | PASS | 792-796 | Fitted on training (`X_fit`), applied to val/cal/test. |
| Inference isolation | PASS | 828-832 | Model receives only input sequences; no future injection. |

---

## E - Quantile Pipeline (e_quantile_daily_pipeline.py)

| Check | Result | Lines | Details |
|-------|--------|-------|---------|
| Fold non-overlap | PASS | 45-74, 266-267 | Expanding window: `train=[0:tr_end)`, `test=[tr_end:te_end)`. |
| Lag features backward | PASS | 103-104 | `y.shift(l)` with l >= 1. Row i gets y[i-l]. |
| Rolling windows trailing | PASS | 106-107 | `rolling(w).mean().shift(1)` — trailing with 1-step lag. |
| Exog shifted by 1 | PASS | 116 | `exog.ffill().bfill().shift(1)` — previous-period values. |
| Target h-step-ahead | PASS | 128-129 | `y_target[i] = y_vals[i + h]` — positional, not calendar-day. |
| Models fit on training | PASS | 175-178, 195-197 | GBQuantile and ResidualRF receive only X_tr, y_tr. |
| Residual quantiles | PASS | 196-202 | Computed from training residuals only. |
| Output semantics | PASS | 298-311 | origin_date, target_date, origin_value all correctly distinguished. |

---

## Summary

All four pipelines **PASS** the leakage audit. No data from the test period is used
during model training, feature construction, or prediction interval estimation.
Features are strictly backward-looking across all families.
