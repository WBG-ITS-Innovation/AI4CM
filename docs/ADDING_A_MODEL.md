# Adding a model

**Fifteen minutes, one file, no permission required.** This guide is written for somebody
who wants to try a model on Treasury data and does not want to read the rest of the
codebase first.

The short version: add one entry to `backend/model_catalog.py`. Write a small function
that builds the model if it is not one line. That is the whole of it. Everything that
makes a number here trustworthy is inherited, and none of it needs touching.

---

## What you get for free

The moment your entry exists, your model is run through exactly the same machinery as the
champion, with no work on your part:

| What it inherits | What that means |
|---|---|
| Rolling-origin evaluation | It is only ever asked to predict days it was not trained on, with a gap between the two so it cannot read the answer. |
| The naive rule | Its error is compared to repeating the same weekday last week. A model that cannot beat that has no business being published. |
| The shared benchmark | Its error is also compared to carrying the value from five working days ago forward. That is the one yardstick every model family here shares, which is what makes their numbers comparable. |
| The Treasury's current method | On the flow lines, its error is compared to the planning construction in use today. |
| The signal self-test | The historical answers are shuffled and the model is refitted. If its error does not get meaningfully worse, it was tracking a typical level rather than anticipating anything, and it is labelled accordingly. |
| The leakage and mimicry checks | Whether any input carries information from after the forecast origin, and whether the "forecast" is a delayed copy of a recent actual. |
| The overfitting cap | Whether it is far better on days it trained on than on days it did not. |
| The experiment ledger | Every run gets a row in `experiments/log.csv` with a run id, a data fingerprint and a code fingerprint, and a sidecar JSON with the full configuration. |
| The publication checks | Applied to whatever it measured, with a plain-language reason attached to every verdict. |

You do not wire any of that up. Every one of them works from the estimator your entry
returns, rather than from a branch that names your model, and there is a test asserting
that no such branch exists.

---

## Step 1: add the entry

Open `backend/model_catalog.py` and add one `ModelSpec` to `MODELS`:

```python
ModelSpec(
    "TheilSen", FAMILY_ML, factory=_theil_sen, added="2026-09-01",
    summary="A straight-line fit computed from the median of many small fits, so a "
            "handful of extreme days cannot move it at all. Slower than the other "
            "linear models and almost unmovable by outliers.",
),
```

Five fields, and only three of them need thought.

- **`name`** is how it appears everywhere: the Lab's model list, the Models page, the
  ledger, the leaderboard. Use the same spelling throughout.
- **`summary`** is one plain sentence, shown to somebody choosing from a list. Say what
  the model *does* and when it might help. Do not name the algorithm and stop, and do not
  claim it is good, because at this point nobody has measured it.
- **`added`** is today's date. It is what makes a new arrival visibly new rather than
  merely unmeasured.
- **`requires`** names any package that must be installed, for example
  `requires=("catboost",)`. Leave it out if the model needs nothing beyond what is
  already here. If your model does need a new package, say so before installing it: a new
  dependency is a decision for a person, not a side effect of adding a model.

## Step 2: add the factory, if you need one

If the estimator is not already constructible from what the catalogue imports, add a
zero-argument function above `MODELS`:

```python
def _theil_sen():
    from sklearn.linear_model import TheilSenRegressor

    return _linear_pipeline(TheilSenRegressor(random_state=0))
```

Two conventions, both load-bearing:

1. **Import inside the function.** The catalogue is read by the Streamlit interpreter,
   which has pandas and nothing else. A module-level `import sklearn` would make the
   Models page unable to list your model's name without the whole modelling stack being
   installed alongside the web app.
2. **Set the seed.** Every model here is constructed with `random_state=0` or its
   equivalent, so a rerun reproduces a number rather than approximating it.

`_linear_pipeline` imputes missing values with the median and standardises the inputs. Use
it for anything sensitive to scale. Tree models do not need it and are returned bare.

## Step 3: run it

```bash
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from b_ml_pipeline import available_models;print('TheilSen' in available_models())"
```

Then run it from the Lab, or from the comparison in each target's Models panel on the
Forecast page. Either way the run is exploratory: it is measured on train and dev data
only, it is never published, and it never touches the sealed window.

---

## What "UNTESTED" means, and why your model has it

Your model appears immediately, badged **UNTESTED**, with no numbers beside it.

That badge is not a judgement. It means no measured result has been recorded for it in
`experiments/log.csv`, so there is no figure anybody can quote, trace to a run, or put in
front of the publication checks. The status is derived from that ledger every time it is
asked, never declared in a field, because a declared status is a claim that drifts the
moment somebody adds a model and does not run it.

It was worth building because the project was already in exactly that state. On the day
the catalogue was written, **seven of the thirteen models the Lab offered had no recorded
result on any target**, and the Lab's default selection was one of them. They sat in the
list beside the champion with nothing distinguishing them.

While a model is untested:

- it can be run as an experiment from anywhere in the app;
- it shows no measured numbers, because there are none;
- it cannot become the model behind an official forecast. A registry recipe cites a ledger
  run id and `registry.verify_against_log` checks that the quoted figures match it, so a
  model with no ledger row has nothing to cite.

## What the checks will do to a bad model

Worth knowing before you spend an afternoon on something.

**If it is less accurate than the naive rule, it is finished.** The accuracy check is a
ratio against repeating the same weekday last week, so the threshold of 1.0 is not a
chosen number: above it, a documented trivial alternative is more accurate, and there is
no reading under which your model's figures are the best available estimate. There is
deliberately no safety margin. This is not hypothetical: on the balance line, twelve
models have been measured and none of them clears it.

**If it only tracks the usual level, it will be published as a guide and not as a
forecast.** The signal self-test shuffles the historical answers and refits. If destroying
the link between inputs and outcome barely hurts your model, it was not using the inputs.
Its numbers still get shown, because a decent estimate of the typical level is useful; the
claim that it anticipates individual days is what gets withheld.

**If it memorised the history, it is excluded from selection.** A model more than three
times better on days it trained on than on days it did not is describing the training
window rather than the process.

**If it is a delayed copy, that is detected.** A "forecast" that reproduces a recent
actual shifted by a few days scores well and predicts nothing.

**If any input carries information from after the forecast origin, the run stops.** That
is leakage, and it is a refusal rather than a warning.

**If you evaluate it on the sealed window, you will be refused.** The 2025 holdout is read
once, deliberately, at the end of a milestone. Every exploratory path in the app is bound
to train and dev data by construction; if you construct a configuration that reaches past
that, `assert_selection_free` stops it and the app explains why in plain words.

None of these can be turned off from the interface, and none of them should be. A model
that cannot pass them is telling you something true.

---

## Adding a statistical model instead

The statistical family does not build scikit-learn estimators, so it works slightly
differently. Two edits in `backend/run_a_stat.py`:

1. an entry in `A_STAT_MODELS` with a `role` (`"forecast"` or `"baseline"`) and a
   `summary`;
2. a branch in `_fc` returning `(y_pred, y_lo, y_hi)`, with NaN intervals if the method
   has none.

Add the name to `STAT_MODEL_NAMES` in `backend/model_catalog.py` as well; a test asserts
the two lists agree. `_fc` refuses a name it has no branch for rather than falling through
to a naive forecast, which is what it used to do, and which meant an unimplemented name
would be published under the requested model's label.

`ETS_DAMPED` was added exactly this way and is a short worked example. `SES` and `HOLT`
were added on 2026-08-19 as a second one; they share the same `ExponentialSmoothing` call
as `ETS` but sit in their own branch rather than widening the condition `ETS` and
`ETS_DAMPED` share, because those two have measured numbers behind them and a new code
path inside their branch would be a new code path inside the thing that produced them.

---

## Adding a quantile model instead

The quantile family produces a range rather than a single number, and it builds its
estimators per fold like the statistical family. Two edits in
`backend/e_quantile_daily_pipeline.py`:

1. an entry in `registry_models()`, `{name: one-line description}`;
2. a branch in `_predict_quantiles` returning `({quantile: array}, n_crossed_rows)`.

Add the name to `QUANTILE_MODEL_NAMES` in `backend/model_catalog.py` too, for the same
reason as the statistical list: a test asserts the two agree.

**The one thing that is easy to get wrong.** If you fit each quantile as its own model,
nothing guarantees the p10 lands below the p50. A band whose lower edge is above its upper
edge is not a wide interval, it is an invalid one. Nothing downstream fixes it for you: the
predictions go straight into the `yhat_p10` / `yhat_p50` / `yhat_p90` columns, and the
`n_cross` your branch returns is only ever *printed*. So pass your predictions through
`_enforce_monotone`, which sorts each row and returns the count of rows it had to fix:

```python
if model_name == "MyQuantile":
    return _enforce_monotone({q: _fit_mine(X_tr, y_tr, X_new, q) for q in quantiles},
                             quantiles)
```

Return the count rather than zero. Constant crossing is what a misconfigured quantile
model looks like from outside, and a silent repair means nobody ever finds out.

`LinearQuantile`, `HistGBQuantile` and `XGBQuantile` were added this way on 2026-08-19.
Two of the three cross on real rows, which is why the paragraph above exists.

---

## What you do NOT have to update

Nothing in `frontend/`. The Lab's model pickers read the registry through
`frontend/backend_consts.py`, which imports `model_catalog` directly. That is why the
catalogue keeps no heavy imports at module level.

This is worth stating because it was not always true, and the failure was quiet. Until
2026-08-19 those lists were typed by hand with a comment asking whoever added a model to
remember. Nobody had: the registry offered 15 machine-learning models and the Lab named 8,
the quantile family offered 3 and the Lab named 1, and `ETS_DAMPED` was missing entirely.
**Ten registered models could not be run from the Lab at all**, while the Models page told
readers they could run any untested model there. A test now fails if a model list
reappears in `08_Lab.py`.

One trap if you ever touch that file. `ModelSpec.installed` calls `find_spec` in whichever
interpreter asks, and there it is the Streamlit one, where XGBoost, LightGBM and CatBoost
are absent because they live in `backend/.venv`. Filtering the Lab's list on that flag
would delete XGBoost and LightGBM from the interface while the runs that use them work
perfectly, so the lists are deliberately unfiltered.

## What you DO have to update

One test, on purpose:
`test_regression_mutations.py::test_the_model_counts_are_pinned_so_a_headline_number_cannot_drift`.

It pins the shelf size, the split by family, and how many models anybody has actually
measured. It fails when you add a model, and that is the design: the app shows a count to a
reader, so the count changes deliberately with its composition restated, rather than
drifting. Update the numbers and the docstring's account of what changed.

The number to watch is `evaluated_total`. On 2026-08-19 eleven models were registered and
it did not move: 42 models on the shelf, still 8 measured, so the untested count went from
20 to 31. Growing the shelf and growing the evidence are different events, and quoting the
first as though it were the second is the thing this whole file is arranged to prevent.

---

## Getting your model measured

Being on the shelf and having a recorded result are different things. To move from
UNTESTED to measured, the model needs a run entered in `experiments/log.csv` through the
evaluation protocol, on train and dev data only. That is a deliberate exercise rather than
a side effect of clicking Run, and it is what earns a model the right to be quoted.

## Where to look

| File | What it holds |
|---|---|
| `backend/model_catalog.py` | The shelf. One entry per model, and the factories. |
| `backend/tests/test_model_catalog.py` | What a valid entry must contain, and the proof that moving the definitions here changed no number. |
| `backend/publication_gates.py` | Every check, its threshold, and why that threshold. |
| `backend/evaluation_windows.py` | Which data may be used to choose anything, and which may only be reported on. |
| `docs/DATA_SEMANTICS.md` | What the Treasury lines actually mean. |
