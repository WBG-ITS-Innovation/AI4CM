# Roadmap

Where the project is, in order. Each item lists what it changes and what it unblocks, so a
reader can see why the sequence is what it is.

**Status as of 2026-08-05:** three target champions exist as *candidates*, validated on 2024.
Nothing is tuned, nothing is approved, and the 2025 holdout has never been evaluated against.

---

## Now — what exists

| | |
|---|---|
| Coverage | 3 of 41 Treasury lines, 5 working days ahead |
| Validated on | 2024 (DEV). 2025 sealed, **zero** evaluations |
| Accuracy vs benchmark | 29–41% lower error than "assume today repeats in five working days" |
| Genuine event signal | **1 of 3 targets.** State budget balance yes; revenues and expenditure no |
| Tuning | none — all principled defaults |
| Approvals | none — no approval workflow exists |

The signal result is the governing fact for everything below: three workstreams and a
three-instrument robustness study have failed to find event-level signal in the two flow
targets. See `reports/ws5_multivariate.md` §6.

---

## 1 · Workstream 4 — target scaling

Compare **raw** vs **asinh** vs **ratio-to-trailing-level**, statistics fit per fold. The
stock target keeps its delta path. `log1p` is unavailable while Treasury question **T1**
(negative revenues) is open, since it is undefined below −1.

*Care required:* this threads a transform through the prediction path, where `origin_value`
and `y_true` are written — the columns the shared persistence benchmark is computed from. An
error there fails silently. Verify the benchmark is byte-identical before and after.

**Unblocks:** WS2, which should tune once, on the final recipe.

## 2 · Workstream 2 — hyperparameter tuning

LightGBM quantile port (P10/P50/P90, crossing-safe), Optuna ≈100 trials with horizon-gapped
early stopping. **CatBoost joins the pool here** (installed 2026-08-05; `CatBoost_L1` and
`CatBoost_Quantile` register but are unablated, so they are not promotable until they have
been through the same TRAIN-folds-then-one-DEV-confirmation protocol as workstream 1). Build on the absolute-error objective — pinball loss at the median *is*
absolute error, so the objectives are already consistent.

Deliberately sequenced after WS3/WS4/WS5, all of which changed the feature set or target
representation. Tuning before them would have tuned a recipe that no longer exists.

## 3 · Workstreams 6 & 7 — selection, intervals, reporting

- **Ensembling** (WS6).
- **Per-target selection** (WS7) — must keep squared-error candidates in the pool: on
  Expenditure the best 2024 model is a squared-error one.
- **Conformalised quantile regression** — the fix for the known interval defect: on the
  largest third of days the nominal 80% range currently captures ~50% of outcomes.
- **Dual-probe signal reporting** — **decided 2026-08-05: adopt.** Report both the linear
  (ridge) and nonlinear (tree) signal readings side by side. The 1.50 threshold stays with
  ridge and continues to gate. The tree reading is **reported only** until a null
  distribution is derived for it, which is WS7 work — a threshold calibrated for one
  instrument does not transfer to another, and the same feature set reads 1.167 under ridge
  and 1.421 under a tree.
- **Model-agreement disagreement** — **decided 2026-08-05: report, do not gate.** The
  independent point and interval models disagree by up to 35.6% on Expenditure and 21.0% on
  Revenues against 1.5% on the stock target. It goes in artifacts and the narrative as a
  readable confidence signal. Gating on it needs its own null distribution first: we do not
  know what disagreement looks like when a model is behaving well, so any threshold today
  would be invented rather than derived.
- **The Expenditure tie-break** — **decided 2026-08-05.** The champion stays `LightGBM_L1`
  on five-TRAIN-fold evidence, with `not_the_dev_best` disclosed. WS7 owns the formal
  tie-break rule and **must keep squared-error candidates in the pool**.

## 4 · Operations — daily automation

Missing/late/partial input handling, idempotent re-runs, drift detection, and the artifact
**validator** that fails loudly before anything is published. Plus the deferred Phase-1
cleanup (untrack `.venv` and `.env`).

**Unblocks:** running unattended.

## 5 · Registry approvals

An actual approval workflow, so `approved_by` can stop being null: who signs off, against
what evidence, and how a recipe is retired. Until this exists, no model can honestly be
called production-ready.

## 6 · The single TEST read + trust pack

**Once.** Evaluate the final per-target candidates on 2025 through the gated harness, then
publish the trust pack. Any model failing a gate ships as WITHHELD with the reason — gates
are not relaxed to make a model pass.

This is deliberately last among the modelling items: it is a one-shot check and spending it
early destroys its value as an independent estimate.

## 7 · Front-end — FastAPI + React

Replace the Streamlit demo with a service and a real front-end. Streamlit has been the right
tool for building and showing the analysis; it is not the right tool for a ministry-facing
daily product.

## 8 · Global model & foundation-model benchmark

One model across many Treasury lines (cross-learning), and a benchmark against
time-series foundation models. Both are genuinely promising and both are *after* the
groundwork, because neither can be evaluated honestly without the harness above.

---

## Outside the modelling queue — and possibly worth more than all of it

**Treasury's forward domestic debt auction and redemption calendar.**

The days the flow models cannot anticipate are overwhelmingly debt-operation days: on the 72
business days when revenues print negative, the debt lines are negative on 64 and 65 of them
respectively, with correlations of 0.97. Those days are not on a fixed date in the month, so
no feature engineering reaches them. We already hold the *historical* debt figures and tested
them — they do not help, because knowing what happened yesterday does not tell you what is
scheduled next week.

A forward schedule is the only untried input that is knowable in advance and targets exactly
the days that are unpredictable. `docs/FISCAL_CALENDAR_SOURCES.md` is the document prepared
for Treasury to confirm; also outstanding there are public-sector salary and pension payment
dates, which no public source could supply.

**Also open:** question **T1** — whether negative revenue figures are a genuine netting
convention or a recording issue. It changes which target transforms are available in WS4.

---

## The honest alternative to items 1–3

If WS4, WS2 and WS6/7 also fail to move the flow signal, the correct conclusion is that a
five-working-day-ahead forecast of daily Treasury *flows* does not exist in this data beyond
the level of central tendency — and the deliverable becomes reporting them that way, with
calibrated ranges and an explicit statement of what the system cannot do.

That would be a legitimate outcome, not a failed project. Three negative results and a
robustness study are evidence for it. The stock target already forecasts genuinely, and a
system that reliably tells you which of its own outputs to trust is worth more than one that
claims to forecast everything.
