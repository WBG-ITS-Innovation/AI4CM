# AI4CM Backtest Report — NOT AVAILABLE

**Status:** cannot be generated yet. This file exists so that its absence is legible rather than silent.

## Why

The trust pack asks for a report "whose 'one shared baseline' sentence is now true". It cannot be made
true today. The generated report's own explanatory text claims skill is "measured on one shared baseline
for all families". Measured (review §4.3), that sentence is false:

| Family | Persistence MAE (pre-Phase-1 data) | Window | Horizon unit |
|---|---:|---|---|
| A_STAT | 60,976,736.58 | 2025-01-01 … 2025-08-06 | 5 business days |
| B_ML | 60,976,736.58 | 2025-01-01 … 2025-08-06 | 5 business days |
| E_QUANTILE | 65,888,163.54 | 2025-01-10 … 2025-08-06 | **5 calendar days** |
| C_DL | 52,957,744.22 | **2019-01-01** … 2025-08-06 | 5 business days |

Two of four agree. E_QUANTILE is the only family not reindexed to `freq="B"`, so its `h=5` is a shorter
forecast than everyone else's. C_DL folds over every available year, so its reported skill is an average
over 2019–2025 rather than the holdout — it reported +10.84% while being −5.19% on the 2025 window.

Regenerating this report before that is fixed would restate the exact falsehood the review caught.

## What must happen first

1. **Unified yardstick** (Phase-1 Step 2, not started): reindex E_QUANTILE to `freq="B"`, pin C_DL to
   `TEST_START`, and verify all four families report literally one persistence number.
2. **Re-run on the regenerated data.** The Phase-1 data fix moved the h=5 persistence baseline from
   60,976,736.58 to **83,534,152.85**, so every skill figure in the review is superseded and will fall.
3. **Per-family evaluation windows stated in the report**, as the trust pack requires.

Tracked in `docs/EXECUTION_PLAN.md` Step 2 and `CHANGELOG.md` under "Known gaps".
