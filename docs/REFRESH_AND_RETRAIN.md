# Refreshing the data, and what happens to the model

This page answers one question: when new Treasury figures arrive, what changes by itself and
what does not?

The short answer. The model is fitted again, on its own, the next time a forecast runs. Which
model is used never changes on its own. Changing that is a separate, deliberate job, and this
page explains why it is kept separate.

---

## The words used here

**Champion.** The one model an official forecast for a Treasury line uses. It was chosen once,
on recorded evidence, from data it had never been fitted on.

**Refit.** Fitting the same model again on more data. The model, its inputs and its settings
stay exactly as they were. Only the numbers inside it move.

**Re-choose.** Picking a different model, or different inputs, or different settings. This is a
different act from a refit, and the rest of this page is about keeping the two apart.

**Sealed window.** The most recent stretch of history the models never saw while being chosen.
We keep it untouched so the final score is honest. It can only be spent once, so no experiment
may be measured on it, and any that tries is refused.

---

## What happens when you upload actuals

You upload a data file on the Scorecard page. In order:

1. The file is checked. It must carry every column the current file has, its dates must be
   readable and unrepeated, and its last day must be later than the last day already held.
2. You see what will change, and nothing is written until you confirm.
3. On confirmation, the file in use is copied to a timestamped backup and the upload takes its
   place.
4. Every published forecast is scored against the new figures. Days still ahead of the data
   stay pending.

That is all the upload does. It replaces a file and scores what can now be scored. It does not
fit anything.

## What happens the next time a forecast runs

An official forecast reads the data file fresh and fits the champion model on all of it,
including the days you just added. There is no button and no step to remember. If you run an
official forecast after an upload, it has already used the new data.

You can confirm it did. Every published issue records the fingerprint of the data it was built
from, in `manifest.json` as `data_sha_at_issue`. Two issues built from different data have
different fingerprints.

## What never changes on its own

Which model is champion. That is recorded in `registry/recipes.json`, and nothing in this
project writes that file. There is no function that saves it and no script that regenerates it.
It is edited by a person.

So new data moves the numbers inside the chosen model. It never changes the choice.

---

## Why the choice is kept separate

Two reasons, and the second is the one that matters.

**A choice made on recent data is not a measured choice.** Data arriving after 2025-08-06 falls
into what the project calls the LIVE window. That window is readable for scoring, which is the
whole point of it, and it is never used to choose anything. The rule is enforced in code:
`evaluation_windows.assert_selection_free` raises if a path that chooses something is handed
LIVE dates. Choosing a model on the same days you then use to score it tells you nothing,
because the model was fitted to look good on exactly those days.

**Re-choosing spends something you cannot get back.** A model chosen once and then scored on
days it never saw gives you a real measurement. Once you re-choose using those days, they stop
being days the model never saw, and the measurement is gone. There is no way to un-spend it.

So re-choosing needs a fresh pass over the training and development data, which is a deliberate
exercise somebody runs and records, not a side effect of loading a file.

## What re-choosing would take

Two things, from `registry.champion_policy()`:

- a selection pass over the training and development data, never the live data;
- a hand edit to `registry/recipes.json`, because nothing writes it.

## The risk this leaves, stated plainly

A model chosen in one period can be the wrong choice in another, and a completed run will not
tell you that. It will fit the chosen model to the new data and report how it did, which is a
different question from whether it was the right model to choose.

The registry already records one measured instance of this. The Revenues recipe uses a `ratio`
transform whose advantage depends on how much the series has drifted: worth about 1.3% when
drift is low and about 24% when drift is high. Quoting one figure as though it were the
model's constant advantage would overstate it.

This is why the Scorecard page shows a health verdict. If a champion stops beating the simple
rule of thumb it is measured against, or its published range stops covering as many days as it
claims, that is the signal to look at the choice again. It is a signal for a person, not a
trigger for the system.

## What to do when the verdict says degrading

1. Look at how many days it is based on. A verdict from four rows is not a verdict.
2. Check whether the recent period differs from the period the model was chosen in. The
   registry's caveats are the place to start.
3. If a fresh selection pass is warranted, run it on training and development data and record
   it, then edit `registry/recipes.json` by hand.
4. Nothing about steps 1 to 3 happens in the app, and that is on purpose.

---

## Related files

| What | Where |
|---|---|
| The recipes, and which model is champion | `registry/recipes.json` |
| The policy this page describes, in machine-readable form | `registry.champion_policy()` |
| The window definitions and the selection guard | `backend/evaluation_windows.py` |
| The upload checks and the install | `backend/ingest_actuals.py` |
| Scoring published forecasts against arrived actuals | `backend/published_forecasts.py` |
