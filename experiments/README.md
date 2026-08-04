# Experiments

Append-only record of every Phase-2 modelling run. Nothing in `reports/` may quote a
number that is not reproducible from a row here.

| Path | Tracked | What |
|---|---|---|
| `log.csv` | yes | one row per run: timestamp, git SHA, data SHA-256, target, feature-set hash, params, seed, fold scheme, DEV MAE, MASE, skill vs unified ruler, sentinel ratio, per-tercile coverage |
| `runs/<run_id>.json` | no | full per-run detail (all params, per-fold metrics, fold definitions) |
| `test_access.log` | no | every read of the locked TEST window, with reason and timestamp |

`test_access.log` is untracked on purpose: it is a local audit trail, and its value is
that it exists at all. During model search it should stay empty. `runs/` is untracked
because it is bulky and regenerable; `log.csv` is the durable index.

Reading the TEST window requires `AI4CM_ALLOW_TEST_READ=1` and raises otherwise. See
`backend/evaluation_windows.py`.
