"""
Tests for benchmark runner safety and output destination handling.

Failure class: Benchmark safety gap (run_50_canonical.py).

Defect summary:
---------------
Historical runner at `tests/diagnostics/run_50_canonical.py` hardcoded:
  REPORT_JSON = ROOT / "tests" / "diagnostics_50_canonical.json"
  REPORT_TXT  = ROOT / "tests" / "diagnostics_50_canonical.txt"
It provided no `--output-dir` option, overwriting the Oct 6 baseline silently,
and had no pre-flight destination validation or collision refusal.

Test design:
------------
- Unit tests with mocked evaluation — fast (<1s), deterministic, NO Ollama calls.
- Verifies:
  1. Separate output destination can be selected via `--output-dir`.
  2. Both JSON and text reports are written to that destination.
  3. Historical baseline files in `tests/` remain untouched.
  4. Existing result files are not silently overwritten.
  5. Invalid destinations fail before the expensive evaluation loop begins.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from tests.diagnostics import run_50_canonical


def _make_dummy_case_result(case: dict[str, Any]) -> dict[str, Any]:
    """Fast dummy result matching evaluate_case contract."""
    return {
        "id": case.get("id", "q_test"),
        "category": case.get("category", "test"),
        "question": case.get("question", "test question"),
        "expected": case.get("expected", "test expected"),
        "elapsed_seconds": 0.001,
        "status": "PASS_STRUCTURAL",
        "bottleneck": "NONE",
        "diagnosis": "mocked diagnostic pass",
        "query": {"resolved_question": case.get("question", "")},
        "retrieval": {
            "channel_counts": [2, 2],
            "retrieval_result_count": 2,
            "retrieval_weights": [0.5, 0.5],
            "fused_count": 2,
            "verified_count": 2,
            "uncertain_count": 0,
            "rejected_count": 0,
            "ranked_count": 2,
        },
        "evidence": {
            "evidence_status": "supported",
            "coverage_status": "supported",
            "context_characters": 100,
        },
        "answer_generation": {
            "answer": "Mock authoritative answer.",
            "internal_leaks": [],
            "unknown_markers": [],
            "status": "PASS",
        },
    }


@pytest.fixture
def mock_runner_environment(monkeypatch: pytest.MonkeyPatch):
    """Fixture providing dummy cases and mocked evaluate_case to avoid LLM calls."""
    dummy_cases = [
        {"id": f"q_{i}", "category": "test", "question": f"Question {i}?"}
        for i in range(1, 51)
    ]
    monkeypatch.setattr(
        run_50_canonical,
        "evaluate_case",
        _make_dummy_case_result,
    )
    # Ensure QUESTIONS_FILE reads dummy cases if accessed
    original_read_text = Path.read_text

    def mock_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == run_50_canonical.QUESTIONS_FILE:
            return json.dumps(dummy_cases)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    return dummy_cases


class TestRunnerSafety:
    """Safety contracts for tests/diagnostics/run_50_canonical.py."""

    def test_separate_output_destination_selected_and_written(
        self,
        tmp_path: Path,
        mock_runner_environment: list[dict[str, Any]],
    ) -> None:
        """A separate output destination can be selected and both JSON and TXT are written."""
        output_dir = tmp_path / "custom_run_01"

        # The runner main should accept argv list
        run_50_canonical.main(["--output-dir", str(output_dir)])

        report_json = output_dir / "diagnostics_50_canonical.json"
        report_txt = output_dir / "diagnostics_50_canonical.txt"

        assert report_json.is_file(), f"Expected JSON report at {report_json}"
        assert report_txt.is_file(), f"Expected TXT report at {report_txt}"

        data = json.loads(report_json.read_text(encoding="utf-8"))
        assert data.get("total") == 50
        assert len(data.get("results", [])) == 50
        assert "BOTTLENECK DISTRIBUTION" in report_txt.read_text(encoding="utf-8")

    def test_historical_baseline_remains_untouched(
        self,
        tmp_path: Path,
        mock_runner_environment: list[dict[str, Any]],
    ) -> None:
        """Running with an explicit output destination never modifies historical baseline in tests/."""
        baseline_json = run_50_canonical.ROOT / "tests" / "diagnostics_50_canonical.json"
        baseline_txt = run_50_canonical.ROOT / "tests" / "diagnostics_50_canonical.txt"

        assert baseline_json.exists(), "Historical baseline JSON must exist"
        assert baseline_txt.exists(), "Historical baseline TXT must exist"

        json_stat_before = baseline_json.stat()
        txt_stat_before = baseline_txt.stat()
        json_bytes_before = baseline_json.read_bytes()[:512]
        txt_bytes_before = baseline_txt.read_bytes()[:512]

        output_dir = tmp_path / "safe_isolated_run"
        run_50_canonical.main(["--output-dir", str(output_dir)])

        json_stat_after = baseline_json.stat()
        txt_stat_after = baseline_txt.stat()

        assert json_stat_after.st_mtime == json_stat_before.st_mtime, "Historical JSON mtime modified!"
        assert txt_stat_after.st_mtime == txt_stat_before.st_mtime, "Historical TXT mtime modified!"
        assert baseline_json.read_bytes()[:512] == json_bytes_before, "Historical JSON content modified!"
        assert baseline_txt.read_bytes()[:512] == txt_bytes_before, "Historical TXT content modified!"

    def test_existing_result_files_not_silently_overwritten(
        self,
        tmp_path: Path,
        mock_runner_environment: list[dict[str, Any]],
    ) -> None:
        """The runner must refuse to silently overwrite an existing run directory."""
        output_dir = tmp_path / "already_exists_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        existing_json = output_dir / "diagnostics_50_canonical.json"
        sentinel_content = '{"sentinel": "DO_NOT_OVERWRITE"}'
        existing_json.write_text(sentinel_content, encoding="utf-8")

        with pytest.raises((FileExistsError, SystemExit)) as exc_info:
            run_50_canonical.main(["--output-dir", str(output_dir)])

        # Existing content must be intact
        assert existing_json.read_text(encoding="utf-8") == sentinel_content

    def test_invalid_destination_fails_before_question_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid destination paths fail before the expensive evaluation loop starts."""
        dummy_cases = [
            {"id": f"q_{i}", "category": "test", "question": f"Question {i}?"}
            for i in range(1, 51)
        ]
        eval_call_count = 0

        def recording_eval(case: dict[str, Any]) -> dict[str, Any]:
            nonlocal eval_call_count
            eval_call_count += 1
            return _make_dummy_case_result(case)

        monkeypatch.setattr(run_50_canonical, "evaluate_case", recording_eval)

        # Create a file so that using it as a directory component is invalid
        blocking_file = tmp_path / "not_a_dir"
        blocking_file.write_text("blocked", encoding="utf-8")
        invalid_output_dir = blocking_file / "sub_dir"

        with pytest.raises((NotADirectoryError, OSError, SystemExit)):
            run_50_canonical.main(["--output-dir", str(invalid_output_dir)])

        # Must have failed BEFORE evaluating any case
        assert eval_call_count == 0, f"evaluate_case was called {eval_call_count} times before failure!"

    def test_default_generates_unique_output_dir_under_test_outputs(
        self,
        tmp_path: Path,
        mock_runner_environment: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no --output-dir is specified, a unique directory is created under test_outputs/rag_quality/."""
        # Redirect default base to tmp_path to prevent polluting workspace during tests
        custom_base = tmp_path / "test_outputs" / "rag_quality"
        monkeypatch.setattr(run_50_canonical, "DEFAULT_OUTPUT_BASE", custom_base, raising=False)

        run_50_canonical.main([])

        # Verify a subdirectory was created under custom_base
        created_subdirs = [p for p in custom_base.iterdir() if p.is_dir()]
        assert len(created_subdirs) == 1, f"Expected 1 unique run directory, found {created_subdirs}"

        run_dir = created_subdirs[0]
        assert (run_dir / "diagnostics_50_canonical.json").is_file()
        assert (run_dir / "diagnostics_50_canonical.txt").is_file()
