# The signal finding

**What we established, how, and what would change it.** Every figure traces to a row in
`experiments/log.csv`; none is quoted from a summary table.

**Date:** 2026-08-05 · **Data SHA-256:** `0b009fd0…5361f1` · **Calendar version:** `4b480eae9c8f`
**TEST (2025) reads: 0** — the finding rests entirely on TRAIN folds and the 2024 DEV window.

---

## In one paragraph

Five separate accuracy levers each improved how closely the models track daily Treasury flows,
and **not one of them moved the flow targets' signal test above its threshold** — under three
independent statistical probes. On `Revenues` and `Expenditure` the system therefore publishes a
**calibrated central-tendency guide**, not an event forecast: the numbers are useful for "what
does a normal day look like", and must not be relied on to anticipate an unusual one. The stock
target, `State budget balance`, is different — it has clear signal and forecasts genuinely.

---

## What the signal test is, and why the threshold is 1.50

Shuffle the historical answers, refit the model on the same features, and compare held-out
error. If the features genuinely inform the target, destroying the feature-answer pairing should
hurt badly. The ratio of shuffled error to real error is the reading.

The threshold is **1.50** — the error must get at least half again worse. A reading of 1.00 means
shuffling changed nothing at all, so the margin above 1.00 exists so that noise cannot pass.

The test deliberately answers *"is there signal?"* and **not** *"is there leakage?"* Those were
once conflated in the opposite direction: leakage makes real-target error implausibly *small* and
so makes this ratio *large*. A low reading means the features never carried much information.

---

## The five levers, and what each did

Every row is measured on DEV (2024) against the single shared h-step persistence benchmark.
Sentinel is the ridge probe, the one that gates.

| # | Lever | Best accuracy effect on a flow target | Flow sentinel after |
|---|---|---|---|
| 1 | **Absolute-error objectives** (WS1) | L1 beat its squared-error twin in **17 of 18** paired comparisons | unmoved |
| 2 | **Georgian fiscal calendar** (WS3) | DEV error **−6.32%** on Revenues | 1.138 → 1.226 |
| 3 | **Multivariate / exogenous lines** (WS5) | **every** block made both flows worse | 1.226 → 1.173 (down) |
| 4 | **Target scaling** (WS4) | DEV error **−25.73%** on Revenues — the largest single gain in the project | unmoved |
| 5 | **Hyperparameter tuning** (WS2, 100 trials × 3) | both flows got **worse** on DEV (−3.13%, −0.17%) | 1.226 → 1.167 |

Two of the five made the flows *worse*. The two that helped most — the fiscal calendar and target
scaling — moved accuracy substantially and the sentinel barely at all.

### Across every logged reading

| Target | Logged readings | Range | Highest ever | Threshold |
|---|---:|---|---:|---:|
| Revenues | 36 | 1.0000 – 1.2255 | **1.2255** | 1.50 |
| Expenditure | 35 | 1.0710 – 1.1977 | **1.1977** | 1.50 |

Not one reading, across five workstreams and 71 logged flow runs, reached the threshold.

---

## Three probes, not one

The obvious objection is that the instrument is too blunt: the probe is a **ridge** regression,
so it can only register signal a *linear* model can use. That was tested rather than assumed
(`reports/sentinel_probe_study.md`).

**The blind spot is real.** On a synthetic control carrying a pure two-way interaction — strong
signal, zero linear signal — ridge reads 1.295 and misses it, while a decision tree reads 5.137
and a random forest 4.933.

**But it is not the blind spot that matters here.** On the real flow targets:

| Target | ridge | tree | forest | Verdict |
|---|---:|---:|---:|---|
| Revenues | 1.167 | **1.421** | 1.260 | no signal |
| Expenditure | 1.071 | **1.397** | 1.155 | no signal |
| State budget balance | 3.992 | 3.182 | 3.580 | signal |

The tree probe does read consistently higher on the flows — about 0.25–0.33 — so ridge *does*
mildly understate. It is nowhere near enough: the highest flow reading under **any** of three
instruments is **1.4212**.

One hypothesis of ours was falsified in the process. We had suggested the fiscal calendar's
rare-categorical features were what a linear probe could not exploit. On exactly that shape all
three probes agree to within 0.04 — a rare binary indicator is a single coefficient, which ridge
handles perfectly well. That explanation was wrong.

---

## What this means for what we publish

**`Revenues` and `Expenditure` publish as calibrated central-tendency guides.** Their errors
genuinely are 29–56% below the benchmark of "assume the value from five working days ago
repeats". That is real and it is useful. But with no detectable feature signal, what the model is
doing is regressing toward a central level against a spiky benchmark — so a large improvement
over a weak benchmark is **still not a forecast of individual days**.

Both carry the verdict `withheld_as_forecast` in `registry/recipes.json`, with the reason in
plain language and the named fix. The numbers are shown; the *claim* is withheld.

**`State budget balance` publishes as a forecast.** Sentinel 3.99–7.01 depending on the recipe,
comfortably above threshold.

The two numbers must always be quoted together. `LightGBM_L1` on Revenues shows **55.92% skill**
beside a sentinel of **1.2255**. Either alone misleads: the first oversells, the second hides a
genuine accuracy gain.

---

## What would change this — two inputs, neither a modelling change

### 1 · Treasury's forward auction and redemption calendar

The days the flow models cannot anticipate are overwhelmingly **debt-operation days**. On the 72
business days when `Revenues` prints negative, `Increase in liabilities` is also negative on
**64** of them (correlation **0.971**) and `Domestic` on **65** (**0.969**) —
`docs/DATA_SEMANTICS.md` §1.

Those days are not on a fixed calendar date, so no `day-of-month` feature reaches them. And
critically, **the historical debt figures do not substitute for a schedule.** We hold them and we
tested them: WS5's `debt_ops` block made every target worse. That 0.971 is a *contemporaneous*
correlation — on a day when debt operations net negative, revenues do too. It does not say
yesterday's operations predict next week's, and no amount of lagging converts a same-day
accounting identity into a forecast. Knowing what happened is not knowing what is scheduled.

A forward schedule is the only untried input that is knowable in advance and targets exactly the
days that are unpredictable.

### 2 · A current data file

The canonical dataset **ends 2025-08-06**. Every forward forecast therefore predicts dates that
are beyond the data and already in the past, and no published forecast can be scored until the
input moves forward. Refreshing it does two things at once: it makes the forward output
operationally meaningful, and it starts the published-forecast track record accumulating real
out-of-sample evidence — which is how accuracy gets demonstrated without spending the sealed
2025 holdout.

---

## The alternative that deserves stating

If the auction calendar arrives and the flow sentinel still does not move, the correct conclusion
is that **a five-working-day-ahead forecast of daily Treasury flows does not exist in this data
beyond the level of central tendency** — and the deliverable becomes reporting them exactly that
way, with calibrated ranges and an explicit statement of what the system cannot do.

That would be a legitimate finding, not a failed project. Five negative results and a
three-instrument robustness study are evidence *for* it. The stock target already forecasts
genuinely, and a system that reliably tells you which of its own outputs to trust is worth more
than one that claims to forecast everything.

---

## Sources

| Claim | Where |
|---|---|
| Per-lever effects and sentinels | `experiments/log.csv`; `reports/ws1_objectives.md`, `ws3_fiscal_calendar.md`, `ws4_target_scaling.md`, `ws5_multivariate.md`, `ws2_tuning.md` |
| Three-probe study | `reports/sentinel_probe_study.md` |
| Negative-`Revenues` / debt-operation measurement | `docs/DATA_SEMANTICS.md` §1 |
| Drift-dependence of the scaling gain | `reports/ws4_robustness.md` |
| Published verdicts and their reasons | `registry/recipes.json` |
