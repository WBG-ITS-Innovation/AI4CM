# frontend/tests/test_lab_run_folders.py
"""
Tests for the run-folder creation logic that caused the NameError crash.

The original bug:
    NameError: name 'out_root' is not defined
    at line 618 of pages/00_Lab.py

Root cause: `new_run_folders()` was imported but never called, so
`out_root`, `run_dir`, and `run_id` were undefined at module scope,
causing the page to crash on every load — even before the user clicks
"Run experiment".

These tests verify:
1. `new_run_folders()` returns valid, usable paths.
2. The returned `out_root` directory is created and exists.
3. `run_dir` is a parent of `out_root`.
4. The env dict can be constructed safely once folders are created.
5. Edge-case run labels (special characters, very long names) are safe.
6. The module-level code in 00_Lab.py no longer references out_root/run_dir.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parents[1]
LAB_PAGE = FRONTEND_DIR / "pages" / "00_Lab.py"


# ---------------------------------------------------------------------------
# 1. new_run_folders() produces valid, existing paths
# ---------------------------------------------------------------------------
class TestNewRunFolders:
    """Tests for utils_frontend.new_run_folders()."""

    def test_returns_three_values(self, tmp_path, monkeypatch):
        """new_run_folders must return (run_id, run_dir, out_root)."""
        import importlib
        import sys

        # Patch RUNS_ROOT to tmp_path so tests don't write to real runs/
        monkeypatch.setenv("_TEST_MODE", "1")
        # We need to monkeypatch the module-level RUNS_ROOT
        sys.path.insert(0, str(FRONTEND_DIR))
        try:
            import utils_frontend
            original_root = utils_frontend.RUNS_ROOT
            utils_frontend.RUNS_ROOT = tmp_path / "runs"
            utils_frontend.RUNS_ROOT.mkdir(parents=True, exist_ok=True)

            run_id, run_dir, out_root = utils_frontend.new_run_folders("test_run")
            assert isinstance(run_id, str)
            assert isinstance(run_dir, Path)
            assert isinstance(out_root, Path)
        finally:
            utils_frontend.RUNS_ROOT = original_root

    def test_out_root_exists_after_creation(self, tmp_path, monkeypatch):
        """out_root directory must exist after new_run_folders() returns."""
        import sys
        sys.path.insert(0, str(FRONTEND_DIR))
        import utils_frontend

        original_root = utils_frontend.RUNS_ROOT
        utils_frontend.RUNS_ROOT = tmp_path / "runs"
        utils_frontend.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            _, run_dir, out_root = utils_frontend.new_run_folders("test_run_exist")
            assert out_root.exists(), "out_root must exist after creation"
            assert out_root.is_dir(), "out_root must be a directory"
            assert run_dir.exists(), "run_dir must exist after creation"
        finally:
            utils_frontend.RUNS_ROOT = original_root

    def test_out_root_is_child_of_run_dir(self, tmp_path, monkeypatch):
        """out_root must be under run_dir (run_dir/outputs/)."""
        import sys
        sys.path.insert(0, str(FRONTEND_DIR))
        import utils_frontend

        original_root = utils_frontend.RUNS_ROOT
        utils_frontend.RUNS_ROOT = tmp_path / "runs"
        utils_frontend.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            _, run_dir, out_root = utils_frontend.new_run_folders("test_child")
            assert str(out_root).startswith(str(run_dir)), \
                f"out_root ({out_root}) must be under run_dir ({run_dir})"
        finally:
            utils_frontend.RUNS_ROOT = original_root

    def test_resolve_works_on_out_root(self, tmp_path, monkeypatch):
        """str(out_root.resolve()) must not raise — this was the crash line."""
        import sys
        sys.path.insert(0, str(FRONTEND_DIR))
        import utils_frontend

        original_root = utils_frontend.RUNS_ROOT
        utils_frontend.RUNS_ROOT = tmp_path / "runs"
        utils_frontend.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            _, _, out_root = utils_frontend.new_run_folders("test_resolve")
            # This is the exact expression that crashed:
            result = str(out_root.resolve())
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            utils_frontend.RUNS_ROOT = original_root

    def test_env_dict_can_be_built(self, tmp_path, monkeypatch):
        """The full env dict can be constructed after new_run_folders()."""
        import sys
        sys.path.insert(0, str(FRONTEND_DIR))
        import utils_frontend

        original_root = utils_frontend.RUNS_ROOT
        utils_frontend.RUNS_ROOT = tmp_path / "runs"
        utils_frontend.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            _, run_dir, out_root = utils_frontend.new_run_folders("test_env")
            env = {
                "TG_FAMILY": "B_ML",
                "TG_MODEL_FILTER": "Ridge",
                "TG_TARGET": "revenue",
                "TG_CADENCE": "Daily",
                "TG_HORIZON": "6",
                "TG_DATA_PATH": "/tmp/test.csv",
                "TG_DATE_COL": "date",
                "TG_PARAM_OVERRIDES": json.dumps({"folds": 1}),
                "TG_OUT_ROOT": str(out_root.resolve()),
            }
            assert env["TG_OUT_ROOT"]
            assert Path(env["TG_OUT_ROOT"]).exists()
        finally:
            utils_frontend.RUNS_ROOT = original_root


# ---------------------------------------------------------------------------
# 2. Edge cases for run labels
# ---------------------------------------------------------------------------
class TestRunLabelEdgeCases:
    """Ensure special characters / long names don't break folder creation."""

    @pytest.fixture(autouse=True)
    def _patch_runs_root(self, tmp_path, monkeypatch):
        import sys
        sys.path.insert(0, str(FRONTEND_DIR))
        import utils_frontend
        self._original = utils_frontend.RUNS_ROOT
        utils_frontend.RUNS_ROOT = tmp_path / "runs"
        utils_frontend.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self.utils = utils_frontend
        yield
        utils_frontend.RUNS_ROOT = self._original

    def test_special_characters_in_label(self):
        """Labels with spaces/slashes/colons get sanitised."""
        _, run_dir, out_root = self.utils.new_run_folders("run A/B:C D@2024")
        assert out_root.exists()
        # No slashes or colons in the folder name
        assert "/" not in run_dir.name
        assert ":" not in run_dir.name

    def test_very_long_label_truncated(self):
        """Labels > 160 chars get truncated safely."""
        label = "a" * 300
        _, run_dir, out_root = self.utils.new_run_folders(label)
        assert out_root.exists()
        assert len(run_dir.name) <= 160

    def test_empty_label_gets_uuid(self):
        """Empty label falls back to UUID-based name."""
        _, run_dir, out_root = self.utils.new_run_folders("")
        assert out_root.exists()
        assert "run_" in run_dir.name  # UUID fallback starts with run_

    def test_none_label_gets_uuid(self):
        """None label falls back to UUID-based name."""
        _, run_dir, out_root = self.utils.new_run_folders(None)
        assert out_root.exists()


# ---------------------------------------------------------------------------
# 3. Static analysis: module-level code must NOT reference out_root/run_dir
# ---------------------------------------------------------------------------
class TestLabPageStaticSafety:
    """Verify 00_Lab.py doesn't reference out_root or run_dir at module level."""

    def test_out_root_not_at_module_level(self):
        """out_root must only appear inside the button handler or in functions."""
        source = LAB_PAGE.read_text(encoding="utf-8")
        lines = source.splitlines()

        # Find the line numbers where `if st.button` starts and ends
        # Module-level = indent 0, not inside a function/if block
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            # Skip comments, blank lines, function defs, string literals
            if not stripped or stripped.startswith("#") or stripped.startswith(("def ", "class ")):
                continue
            if stripped.startswith(('"""', "'''")):
                continue
            # Check if this is module-level (no indentation)
            indent = len(line) - len(line.lstrip())
            if indent == 0 and "out_root" in stripped:
                # This is only OK inside a function definition (def _baseline_series)
                # or type annotation. Check context.
                # Look backwards for a function def
                in_function = False
                for j in range(i - 1, max(0, i - 30), -1):
                    prev = lines[j - 1].lstrip()
                    if prev.startswith("def "):
                        in_function = True
                        break
                    if prev and not prev.startswith("#") and not prev.startswith(("'", '"')):
                        prev_indent = len(lines[j - 1]) - len(lines[j - 1].lstrip())
                        if prev_indent == 0 and not prev.startswith(("def ", "@")):
                            break
                if not in_function:
                    pytest.fail(
                        f"Line {i}: 'out_root' referenced at module level "
                        f"(outside function/button handler): {line.strip()!r}"
                    )

    def test_run_dir_not_at_module_level_in_env_or_launch(self):
        """run_dir must not appear in the env dict or launch_backend call
        at module level (it should only be inside the button handler)."""
        source = LAB_PAGE.read_text(encoding="utf-8")

        # The env dict and launch_backend call must be inside the button
        # handler (indented under `if st.button`).
        # Quick check: find 'run_dir=run_dir' and 'TG_OUT_ROOT' and verify
        # they are indented (inside a block).
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.lstrip()
            indent = len(line) - len(line.lstrip())
            if indent == 0 and "run_dir=run_dir" in stripped:
                pytest.fail(
                    f"Line {i}: 'run_dir=run_dir' at module level: {stripped!r}"
                )
            if indent == 0 and '"TG_OUT_ROOT"' in stripped:
                pytest.fail(
                    f"Line {i}: 'TG_OUT_ROOT' at module level: {stripped!r}"
                )

    def test_no_undefined_results_list(self):
        """The old 'results' variable (from batch-run code) must not remain."""
        source = LAB_PAGE.read_text(encoding="utf-8")
        # 'for r in results' was the old batch-run pattern
        assert "for r in results" not in source, \
            "Old batch-run 'results' variable still referenced in Lab page"


# ---------------------------------------------------------------------------
# 4. backend_bridge: launch_backend handles missing TG_OUT_ROOT gracefully
# ---------------------------------------------------------------------------
class TestBackendBridgeFallback:
    """Verify backend_bridge falls back to run_dir/outputs if TG_OUT_ROOT missing."""

    def test_missing_tg_out_root_uses_fallback(self, tmp_path):
        import sys
        sys.path.insert(0, str(FRONTEND_DIR))
        from backend_bridge import launch_backend

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # Don't set TG_OUT_ROOT in env — bridge should fall back
        env = {"TG_FAMILY": "B_ML"}
        # launch_backend will fail because python doesn't exist, but
        # it should NOT crash on out_root construction
        rc, elapsed, log_path, out_real = launch_backend(
            backend_py="/nonexistent/python",
            runner_script="/nonexistent/runner.py",
            backend_dir=str(tmp_path),
            env_vars=env,
            run_dir=run_dir,
        )
        # Should have created fallback out_root = run_dir/outputs
        fallback = run_dir / "outputs"
        assert fallback.exists(), "Fallback out_root (run_dir/outputs) should be created"
        assert rc != 0  # Expected: python doesn't exist
