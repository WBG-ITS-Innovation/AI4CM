# Demo runbook — 09:00

Read the **DO NOT CLAIM** list at the bottom before you start. It is short and it matters
more than the click-path.

---

## 0 · Launch (fresh-clone friendly)

Two terminals. Everything below is copy-paste from the repository root.

```bash
# ── Terminal 1: generate tonight's artifacts (~2 minutes) ─────────────────────
cd /path/to/AI4CM
git checkout model/excellence

# sanity: the suite must be green before you demo anything
./backend/.venv/bin/python -m pytest -q                     # expect 377 passed

# the forward forecast (writes to backend/forecast_runs/forward/latest/ — gitignored,
# so it is regenerated rather than committed)
./backend/.venv/bin/python backend/run_forward_forecast.py

# the Treasury report (writes reports/treasury_report.html, self-contained)
./backend/.venv/bin/python scripts/build_treasury_report.py

# ── Terminal 2: the app ───────────────────────────────────────────────────────
cd /path/to/AI4CM/frontend
./.venv/bin/streamlit run Overview.py
# → http://localhost:8501
```

**If the venvs are missing** on the demo machine:

```bash
python3 -m venv backend/.venv  && ./backend/.venv/bin/pip install -q -r backend/requirements.txt
python3 -m venv frontend/.venv && ./frontend/.venv/bin/pip install -q -r frontend/requirements.txt
```

**Pre-flight, 60 seconds before you present:**

```bash
ls -la reports/treasury_report.html                                   # exists, today's date
ls -la backend/forecast_runs/forward/latest/                          # 3 files
./backend/.venv/bin/python -c "import sys;sys.path.insert(0,'backend');\
from registry import verify_against_log as v;print(v())"              # ok: True
cat experiments/test_access.log | grep -c TEST_READ || echo "0 holdout reads"
```

If the forward run is missing, the Forecast page says so and prints the exact command —
it fails visibly rather than showing a blank chart.

---

## 1 · Click-path (12–15 minutes)

### A · Open `reports/treasury_report.html` first — in a browser, not the app

This is the artifact a Treasury official would actually receive. Opening it *before* the
app frames everything that follows as "here is the product", not "here is a notebook".

**Say:** "This is a single file. No internet, no server, no login. It opens on a laptop
in a ministry office."

Point at, in order:

1. **The short version** — the conclusion is the first thing on the page.
2. **The red banner**: *validated against 2024, not 2025*. Say it out loud. It is the most
   credible thing on the page.
3. **"What the system can and cannot do"** — this is the centrepiece. See §2.
4. **Line-by-line**: one green banner (State budget balance), two red WITHHELD banners.
5. **Provenance block** — data fingerprint, code version, calendar version, "2025 holdout
   touched: No".

### B · Forecast page (🔭) in the app

Same numbers, interactive. Show the band chart, then the **gate badges** underneath — one
row per check, each with a plain-language reason. On the two flow targets the signal check
shows a red ❌ and explains itself.

Open the **"This is not the single best 2024 result"** expander on Expenditure. Say: "the
system records where its own choice was not the top score, and why."

### C · Models page (🧩)

Scroll to **Promoted recipes**. Point at two columns:

- **Approved by** → *"— nobody —"* on all three rows.
- The green box: *"All 12 quoted figures reconcile against experiments/log.csv."*

**Say:** "Every number on the previous screen traces to a logged run. Nothing here claims
approval, because no approval workflow exists yet."

### D · Dashboard (📈)

Brief. This is the backtesting view — how the models were evaluated, not what they predict.
One sentence: "this is where the evidence comes from; the Forecast page is what it produces."

### E · The audit-trail closer — do not skip this

```bash
# 91 rows, every reported number, with data + code fingerprints
column -s, -t < experiments/log.csv | less -S

# and a session record: the reasoning, the failures, the corrections
open reports/phase9_session_record.md
```

**Say:** "Nine sessions of work, and every number we have ever quoted is in that file with
the code version that produced it. Including the ones that turned out to be wrong — those
are in there too, with the correction."

If you want one example: `reports/phase7_session_record.md` §3 records that a set of figures
we had reported as 2024-only turned out to include 2025 data, how it was caught, and the
corrected numbers.

---

## 2 · The talking point that carries the demo

This is the part to rehearse. The finding sounds like bad news and is the strongest thing
we have.

> "Two of these three budget lines come with a warning that says: this is a guide to the
> typical level, not a forecast of individual days.
>
> We did not decide that by judgement. The system tests itself — it shuffles the historical
> answers, refits the model, and checks whether the error gets meaningfully worse. If a
> model is genuinely using its inputs, destroying the link between inputs and answers should
> hurt it badly. On the state budget balance it does: the error gets seven times worse. On
> revenues and expenditure it barely moves.
>
> So the system distinguishes genuine forecasting from tracking an average, and it labels
> them differently, automatically, without anyone deciding to be modest.
>
> And it names what would fix it. The days these models cannot anticipate are debt-operation
> days — bond auctions and redemptions. Those are not on a fixed date in the month, so no
> amount of statistical work reaches them. We already have the historical debt figures and
> we tested them: they do not help, because knowing what happened last week does not tell
> you what is scheduled next week. What would help is the Treasury's own forward auction
> and redemption calendar.
>
> That is a one-email fix worth more than another month of modelling."

Supporting facts, if asked:

| Question | Answer |
|---|---|
| How sure are you it's not just your test being weak? | We repeated it with three different statistical methods — a linear one and two tree-based ones. All three agree. Written up in `reports/sentinel_probe_study.md`. |
| Are the models any good at all? | On 2024, errors are 29–41% below the benchmark of "assume today repeats in five working days". That is real. It is just not the same as anticipating an unusual day. |
| Why only three lines? | The data has 41. These three are the ones the Treasury named as priorities. The pipeline is not target-specific. |
| Is this tuned? | No. No hyperparameter search, no target scaling. All defaults. That work is scheduled — see `docs/ROADMAP.md`. |

---

## 3 · If something breaks

| Symptom | Fix |
|---|---|
| Forecast page: "No forward run found" | Run `./backend/.venv/bin/python backend/run_forward_forecast.py`. It prints the command itself. |
| Models page: red "does not reconcile" box | Do **not** hand-wave it. Say "the system is telling us its own records disagree, and we would not present numbers in that state." Then move to the HTML report, which is generated from the same registry and independent of the app. |
| Streamlit port in use | `./.venv/bin/streamlit run Overview.py --server.port 8502` |
| A page throws | Skip it. The HTML report is self-contained and needs no app. |
| Asked for 2025 accuracy | See DO NOT CLAIM. The answer is "that evaluation has not been run; it is a one-shot check and we have deliberately not spent it." |

---

## 4 · DO NOT CLAIM

Read this list twice.

1. **No 2025 or "test set" accuracy numbers.** None exist. The 2025 data has never been
   evaluated against — the ledger `experiments/test_access.log` records zero reads. It is a
   one-shot check and spending it early would destroy its value. If pushed: *"it is
   scheduled, and it happens once."*
   - The one honest caveat, if someone technical asks precisely: earlier in the project an
     internal comparison did compute 2025 figures outside the gated harness, before the
     rule was tightened. **No model was selected from them.** It is disclosed in
     `reports/HANDOFF.md` §0a. Do not volunteer this; do not deny it if asked.
2. **No model is "approved", "production-ready", "validated", or "signed off."** Every
   recipe's status is *candidate — pre-tuning* and `approved_by` is null. Say "candidate".
3. **Never quote flow-target skill without the caveat.** "41% better than the benchmark on
   revenues" alone is misleading. Always: *"…and the signal check tells us that is tracking
   the typical level, not anticipating individual days."* The two numbers only mean
   something together.
4. **Do not call the forward figures forecasts for revenues and expenditure.** The report
   says "guide to the typical level". Use its words.
5. **Do not describe the fiscal calendar dates as confirmed.** One rule is confirmed against
   the legislation (the deadline-shift rule). The monthly tax deadlines rest on professional
   sources, not the primary text. Salary dates, pension dates and auction dates are **not
   sourced at all** and contribute nothing to the model. `docs/FISCAL_CALENDAR_SOURCES.md`
   is the document we want Treasury to correct.
6. **Do not claim coverage of the budget.** Three lines of 41, five working days. Say so.
7. **Do not promise the prediction ranges are well calibrated.** On the largest one-third of
   days they currently capture about half the outcomes rather than the intended eight in ten.
   The fix is scheduled. If ranges come up, lead with this rather than waiting to be caught.
8. **Do not claim the LLM writes the analysis.** It is off by default. When enabled it may
   only rephrase; a test rejects any rephrasing that changes a single digit, and the output
   is labelled. If asked "is this AI-generated?", the answer for tonight is *"no — the words
   are templates and the numbers come from the pipeline."*

### One-line summary if you only get 30 seconds

> "It forecasts the next five working days for three Treasury lines, it tells you which of
> those numbers it actually trusts and why, every figure traces to a logged run, and it
> names the one dataset that would make it better."
