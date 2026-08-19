# Session records

One record per working session: what was attempted, what was actually found
(including where the brief's premise turned out to be wrong), what was verified
with real output, and what was deliberately left undone.

They are written to be read after the fact by someone who was not there. Where a
session found a defect, the record says how it was found — because "a test can
name a flag without proving it works" is the kind of lesson that only survives
if it is written down next to the bug it produced.

Several records also carry a **suite count** at the top. Read together they are
the growth curve of the test suite, from 569 to 942.

## The records

| Date | Record | What it covers |
|---|---|---|
| 2026-08-10 | [C_DL alignment integrity](2026-08-10-alignment-integrity.md) | `alignment_ok: True` was a literal with no check behind it. Replaced with a real check, and an audit of every other attested artifact field |
| 2026-08-10 | [The Forecast baseline series](2026-08-10-item-3-forecast-baseline.md) | The h-step persistence benchmark plotted from audited artifacts, never recomputed on the page — one implementation of a published number |
| 2026-08-10 | [Persisted estimators — plan](2026-08-10-item-4-persisted-estimators.md) | Plan only, recorded separately from its implementation so the approval boundary stays visible |
| 2026-08-11 | [Persisted estimators — implementation](2026-08-11-item-4-persisted-estimators.md) | The fitted estimator saved beside each published forecast, with library versions and a loader. Retention `keep_last=8`, unscored issues protected |
| 2026-08-11 | [Quality gates and regression hardening](2026-08-11-item-5-quality-gates.md) | 26 mutation/gap tests and the four distinct failure verdicts. Full audit in `reports/gate_audit.md` |
| 2026-08-11 | [The Agent artifact contract](2026-08-11-item-6-agent-contract.md) | `docs/AGENT_ARTIFACT_CONTRACT.md` and the validator that fails loudly rather than publishing a malformed artifact |
| 2026-08-11 | [The three Lab-side gaps the Agent audit found](2026-08-11-lab-side-agent-gaps.md) | Fixes on this side of the boundary, raised by the Agent repo's contract-consumption audit |
| 2026-08-12 | [The a_stat verdict, and syncing the Agent](2026-08-12-a-stat-verdict-and-agent-sync.md) | The a_stat leaderboard contract failure that made the validator refuse, and the paired Agent commit |
| 2026-08-12 | [Writer fields, target eligibility](2026-08-12-lab-writer-fields-and-eligibility.md) | Three writer gaps the Agent audit surfaced, plus eligibility and family-capability round-tripping |
| 2026-08-13 | [P0 — the two defects in realized accuracy](2026-08-13-p0-scoring-correctness.md) | What made published realized accuracy wrong, and the fixes |
| 2026-08-13 | [P2 — gating on MASE, calibrating the sentinel](2026-08-13-p2-gates.md) | MASE < 1.0 becomes binding; the sentinel threshold moves 1.50 → **1.15** against a measured null; `skill_vs_ruler` demoted to a reported diagnostic |
| 2026-08-13 | [P2 follow-ups](2026-08-13-p2-followups.md) | Verdict reconciliation, the remaining guard holes, one `is_stock` |
| 2026-08-13 | [The LIVE window, and champions stated as fixed](2026-08-13-live-window-and-fixed-champions.md) | TEST closed at the extent it actually holds, so data arriving after sealing gets its own window instead of spending the final read |
| 2026-08-13 | [Refresh → retrain → score diagnostic](2026-08-13-refresh-retrain-score-diagnostic.md) | Diagnostic only; nothing changed. Every run wrote to a temporary directory |
| 2026-08-13 | [Public exposure audit](2026-08-13-public-exposure-audit.md) | **The complete Treasury daily series was public.** What was visible, for how long, and the three-tier remediation plan. Read-only; no secret value printed |
| 2026-08-14 | [The holdout read, recorded; verdict history rendered](2026-08-14-holdout-log-and-verdict-history.md) | The holdout ledger made visible, and the verdict history given a UI |
| 2026-08-14 | [Session 2 — a clean re-issue](2026-08-14-session2-reissue.md) | Regenerating a published issue after the P2 gate rewrite. Includes why the issue is dated a day later than the filename |
| 2026-08-15 | [Session 2.5 — retention](2026-08-15-session2p5-retention.md) | Publishing dual-writes to the private vault; five open items closed |
| 2026-08-17 | [Closing the B_ML holdout ledger gap](2026-08-17-b-ml-holdout-ledger.md) | The last of the four families routed through `require_test_access`, so no family reads the holdout silently |
| 2026-08-17 | [Interval calibration](2026-08-17-interval-calibration.md) | The 80% bands. Big-day coverage measured against the **forecast** rather than the outcome — the earlier framing scored a different question |
| 2026-08-18 | [Session 6-prep — the scoring loop](2026-08-18-session6-prep.md) | `--issue-date` on the publish CLI; the scorecard schema fixed before its first row existed; **and the finding that the Ops baseline returned identically zero**, overstating the client-facing margin roughly fourfold |
| 2026-08-18 | [Regenerating stale artifacts](2026-08-18-artifact-regeneration.md) | Two more live copies of the zero-baseline bug; a new sealed-window harness; **the client-facing measured table**, superseding the July deck; and the finding that the champion credentials are not reproducible |
| 2026-08-18 | [The ws2_tune DEV-fold holdout leak](2026-08-18-dev-fold-holdout-leak.md) | A fold selected rows by origin and scored them against holdout truth. Folds now require an evaluation row's **target date** to sit in an allowed window. No verdict or champion moved |
| 2026-08-19 | [README and docs hygiene](2026-08-19-readme-and-docs-hygiene.md) | Making the repository presentable to a Treasury reader. Found four durable documents stating the opposite of the code — `SIGNAL_FINDING.md`'s conclusion is inverted by the sentinel recalibration. Documentation only |

## How to read them

- **The most recent three carry the current numbers.** Anything earlier may have
  been superseded; where it has, the later record says so explicitly.
- **A record's "Open items" section is the real backlog.** The top item as of
  2026-08-19 is the training embargo, in
  [the DEV-fold record](2026-08-18-dev-fold-holdout-leak.md).
- **Inverted pins.** Several defects are held by a test that *passes while the
  defect exists* and fails once it is fixed, with an assertion message telling
  the fixer to delete it. So a known gap cannot be quietly lost, and cannot be
  "fixed" without someone noticing.

## Related, in the Agent repository

The AI4CM Agent keeps its own session records, and the two repositories cite
each other where work crossed the boundary — the Agent's Session 5 §8 proposed
the `--issue-date` fix that this repository's Session 6-prep applied.
