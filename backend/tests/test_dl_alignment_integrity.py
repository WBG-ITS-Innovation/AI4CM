"""C_DL's `alignment_ok` must be earned by a check, not asserted.

It was the literal `True` (c_dl_pipeline.py:958) inside the integrity update dict. Nothing verified
it. Combined with the dashboard defaulting a *missing* key to True, that made "never checked" and
"passed" indistinguishable — an artifact actively asserting something it had not established, which
is worse than a missing field because a missing field can at least be detected.

The property being attested is the one every other family checks: in the modelling index, a
prediction's target date sits exactly `h` positions after its origin date.
`build_sequences()` constructs them as `idx[end_i]` and `idx[end_i + horizon]` over `F.index`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from forecast_integrity import validate_alignment_step_based  # noqa: E402

H = 5


def _index(n: int = 120) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _aligned(idx: pd.DatetimeIndex, h: int = H, n: int = 40) -> pd.DataFrame:
    """Predictions where the target really is h positions after the origin."""
    rows = []
    for pos in range(10, 10 + n):
        rows.append({"origin_date": idx[pos], "target_date": idx[pos + h], "horizon": h,
                     "model": "LSTM", "y_true": 1.0, "y_pred": 1.0, "origin_value": 1.0})
    return pd.DataFrame(rows)


# ── the requirement: a deliberately misaligned input must produce False ───────

def test_deliberately_misaligned_predictions_yield_alignment_ok_false():
    """The regression that stops the field lying again.

    One prediction is shifted by a single index position — the smallest possible defect, and the
    kind a resampling or off-by-one bug produces.
    """
    idx = _index()
    df = _aligned(idx)
    good = validate_alignment_step_based(df, idx, H)
    assert good["alignment_ok"] is True, "the aligned fixture should pass, or the test proves nothing"

    bad = df.copy()
    pos = list(idx).index(pd.Timestamp(bad.loc[3, "origin_date"]))
    bad.loc[3, "target_date"] = idx[pos + H + 1]          # off by one step

    out = validate_alignment_step_based(bad, idx, H)
    assert out["alignment_ok"] is False, (
        "a target one step out of place was reported as aligned; the field can still lie"
    )
    assert out["n_misaligned"] == 1
    assert out["misaligned_examples"], "a failure must name at least one offending row"


@pytest.mark.parametrize("offset", [-3, -1, 1, 2, 7])
def test_any_wrong_offset_is_caught(offset):
    idx = _index()
    bad = _aligned(idx)
    pos = list(idx).index(pd.Timestamp(bad.loc[5, "origin_date"]))
    tgt = pos + H + offset
    if not (0 <= tgt < len(idx)):
        pytest.skip("offset falls outside the index")
    bad.loc[5, "target_date"] = idx[tgt]
    assert validate_alignment_step_based(bad, idx, H)["alignment_ok"] is False, offset


def test_calendar_gap_does_not_masquerade_as_misalignment():
    """The check is in INDEX POSITIONS, not calendar days.

    On a plain business-day index 5 steps is always 7 calendar days, so that alone would not
    distinguish the two. A HOLIDAY is what makes them differ: dropping a day from the index means
    some 5-step spans become 8 calendar days. A day-difference check would flag exactly those, and
    Georgian holidays are removed from this project's modelling index -- so the distinction is not
    hypothetical.
    """
    idx = _index()
    holiday = idx[25]
    idx = idx.drop(holiday)                      # the index the model actually saw
    df = _aligned(idx)

    spans = (pd.to_datetime(df["target_date"]) - pd.to_datetime(df["origin_date"])).dt.days
    assert spans.nunique() > 1, f"fixture should span differing calendar gaps, got {set(spans)}"

    out = validate_alignment_step_based(df, idx, H)
    assert out["alignment_ok"] is True, (
        f"a holiday gap was mistaken for misalignment: {out.get('n_misaligned')} flagged"
    )


def test_missing_date_columns_yield_false_not_a_pass():
    idx = _index()
    df = _aligned(idx).drop(columns=["target_date"])
    out = validate_alignment_step_based(df, idx, H)
    assert out["alignment_ok"] is False
    assert "error" in out


def test_origin_absent_from_the_modelling_index_is_a_failure():
    """An origin the model never saw cannot have produced that prediction."""
    idx = _index()
    df = _aligned(idx)
    df.loc[0, "origin_date"] = pd.Timestamp("1999-01-04")
    assert validate_alignment_step_based(df, idx, H)["alignment_ok"] is False


# ── the pipeline no longer asserts it ────────────────────────────────────────

def test_c_dl_no_longer_writes_an_unconditional_true():
    src = (BACKEND / "c_dl_pipeline.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert '"alignment_ok": True' not in code, (
        "c_dl writes alignment_ok as a literal True again; the field would assert a check that "
        "never ran"
    )


def test_c_dl_writes_alignment_from_the_shared_checker():
    src = (BACKEND / "c_dl_pipeline.py").read_text()
    assert "validate_alignment_step_based" in src, (
        "c_dl does not call the canonical checker, so its verdict would be its own second "
        "implementation"
    )
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert 'bool(\n' in code or "bool(" in code
    assert '_dl_integrity["alignment_checked"]' in code, (
        "there is no way for a consumer to tell a real verdict from an absent one"
    )


def test_c_dl_omits_the_verdict_when_the_check_cannot_run():
    """On failure the key must be POPPED, not set. An absent verdict reads as 'not checked';
    a favourable default would be the original bug in a new place."""
    src = (BACKEND / "c_dl_pipeline.py").read_text()
    blk = src[src.index("except Exception as _ax:"):]
    blk = blk[:blk.index("# Quality gate")] if "# Quality gate" in blk else blk[:1200]
    assert '_dl_integrity.pop("alignment_ok", None)' in blk, (
        "the failure path does not remove the verdict, so a stale or default value could survive"
    )
    assert '"alignment_ok"] = True' not in blk


def test_the_checker_is_called_against_the_index_sequences_were_built_from():
    """F.index is what build_sequences indexes into. Passing any other index would compare
    positions in one frame against dates from another."""
    src = (BACKEND / "c_dl_pipeline.py").read_text()
    call = src[src.index("validate_alignment_step_based(\n"):]
    call = call[:call.index(")") + 1]
    assert "F.index" in call, f"checker called against something other than F.index: {call!r}"

    # and build_sequences really does index into F.index
    bs = src[src.index("def build_sequences"):]
    bs = bs[:bs.index("\ndef ", 1)]
    assert "idx = F.index" in bs
    assert "idx[end_i + horizon]" in bs and "idx[end_i]" in bs


def test_c_dl_module_still_parses_and_the_check_is_inside_the_integrity_block():
    src = (BACKEND / "c_dl_pipeline.py").read_text()
    ast.parse(src)                                   # syntax
    i_upd = src.index('"mask_target_at_origin": stock')
    i_chk = src.index("validate_alignment_step_based(\n")
    i_gate = src.index("_QUALITY_GATE = 5.0")
    assert i_upd < i_chk < i_gate, (
        "the alignment check is not between the integrity update and the quality gate, so its "
        "verdict may not reach the written report"
    )
