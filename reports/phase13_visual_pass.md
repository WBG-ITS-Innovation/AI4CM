# Phase 13 — visual pass: what changed, per page

Presentation only. No page changed in structure, content, metrics, charts, computations or
numbers. Pages are byte-identical to `706ada9` apart from the additions listed here. No files
were deleted.

Reference: `~/Projects/ai4cm-ui/handoff/DESIGN_TOKENS.md`. Its own guidance — *"the lab's
current palette already passes AA. This is not a request to restyle"* — was followed: the lab's
existing palette is unchanged, and only the tokens Streamlit 1.40.1 cannot express (tabular
numerals, chart chrome) were addressed.

---

## Shared

* **`.streamlit/config.toml`** — already present from an earlier session; unchanged.
* **`ui_styles.py`** — appended a design system: one type scale, one spacing scale, the
  existing accent, and `DESIGN_CSS` which forces **tabular numerals** on metrics, dataframes
  and tables. Without it, proportional digits make `1,111` and `8,888` different widths, so
  numbers do not line up down a column. Added `plotly_chrome()`, which restyles an existing
  figure — gridlines, axis lines, font, margins, legend placement, hover-label styling — and
  never touches traces, data or hover content.
* **`format_gel.py`** — one formatter per value type (`gel_millions`, `pct`, `ratio`, `count`,
  `number`), and a single rendering for absent values. Available to pages; no page's displayed
  values were changed to use it in this pass.

## Overview.py
Design-system CSS injected (tabular numerals, spacing). Nothing else.

## pages/00_Data_Preprocessing.py
Design-system CSS injected. Nothing else.

## pages/00_Lab.py
Design-system CSS injected. Nothing else.

## pages/01_Dashboard.py
* Design-system CSS injected.
* **Chart chrome unified on 3 of 9 existing figures** via `plotly_chrome()` — consistent
  gridlines, axis lines, outside ticks, hover label styling and legend placement. Same traces,
  same data. The remaining 6 already set a white background and the Inter font, so they were
  left alone rather than churned through a fragile edit.
* **`help=` tooltips added to the 6 metrics already on screen** — Horizon, Best Shift, Model
  RMSE, Baseline RMSE, Model R², Baseline R² — in plain language.

## pages/02_History.py
Design-system CSS injected. Nothing else.

## pages/03_Models.py
Design-system CSS injected. Nothing else.

## pages/04_Compare.py
* Design-system CSS injected.
* Chart chrome applied to both existing figures (the overlay chart and the metric chart).

## pages/05_Forecast.py
* Design-system CSS injected.
* Chart chrome applied to the existing band chart.
* **`help=` tooltips added to the 4 metrics already on screen** — Budget lines covered, Called
  a forecast, Working days ahead, Data through.

---

## Also in the working tree, not wired into any page

`frontend/intervals.py` and `frontend/paths.py`, plus several `ui_styles` components, were
written before the scope correction and left in place because nothing was to be deleted. **No
page imports them**, so they change nothing on screen. They are inventoried at the end of
`reports/ui_content_backlog.md`.

`frontend/tests/test_pages_smoke.py` renders all 8 pages and fails on any exception. Its
docstring states plainly what it does *not* verify: the pages do not read the
`AI4CM_RUNS_DIR` override its fixtures set, so all cases exercise the same real artifacts
rather than three different states.

Root suite: 453 passed, 1 skipped, `EXIT=0`. Frontend suite: 59 passed.
