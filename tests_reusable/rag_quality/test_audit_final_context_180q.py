"""Targeted unit tests for scripts/audit_final_context_180q.py.

Strictly adheres to Step 6 requirements:
- Exact question-to-context mapping.
- Correct stable-ID preservation.
- Empty context and abstention formatting.
- Error handling and continuation to the next question.
- Resume without duplicating completed records.
- Correct batch boundaries, including the final 20-question batch.
- No answer-generation API calls.
- Exact context preservation without trimming, rewriting or reordering.
- Dataset hash validation.

All tests use mocked diagnostics and do NOT invoke Ollama or external models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from scripts.audit_final_context_180q import (
    BATCH_DEFINITIONS,
    DEFAULT_BENCHMARK_PATH,
    EXPECTED_DATASET_SHA256,
    compute_sha256,
    format_txt_record,
    run_audit,
    validate_dataset_hash,
)


def test_batch_boundaries():
    """Verify batch boundaries strictly match the 5 batches (180 questions total)."""
    assert len(BATCH_DEFINITIONS) == 5
    b1, b2, b3, b4, b5 = BATCH_DEFINITIONS

    assert b1["batch"] == 1 and b1["start"] == 0 and b1["end"] == 40
    assert (b1["end"] - b1["start"]) == 40
    assert b1["filename"] == "batch_01_questions_001_040.txt"

    assert b2["batch"] == 2 and b2["start"] == 40 and b2["end"] == 80
    assert (b2["end"] - b2["start"]) == 40
    assert b2["filename"] == "batch_02_questions_041_080.txt"

    assert b3["batch"] == 3 and b3["start"] == 80 and b3["end"] == 120
    assert (b3["end"] - b3["start"]) == 40
    assert b3["filename"] == "batch_03_questions_081_120.txt"

    assert b4["batch"] == 4 and b4["start"] == 120 and b4["end"] == 160
    assert (b4["end"] - b4["start"]) == 40
    assert b4["filename"] == "batch_04_questions_121_160.txt"

    assert b5["batch"] == 5 and b5["start"] == 160 and b5["end"] == 180
    assert (b5["end"] - b5["start"]) == 20
    assert b5["filename"] == "batch_05_questions_161_180.txt"

    total = sum(b["end"] - b["start"] for b in BATCH_DEFINITIONS)
    assert total == 180


def test_dataset_hash_validation(tmp_path: Path):
    """Verify dataset SHA-256 validation passes on frozen benchmark and fails on modified dataset."""
    actual_hash = validate_dataset_hash(DEFAULT_BENCHMARK_PATH)
    assert actual_hash == EXPECTED_DATASET_SHA256

    # Test tampering detection
    tampered_file = tmp_path / "tampered.json"
    tampered_file.write_text('[{"id": "fake_01"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset hash mismatch"):
        validate_dataset_hash(tampered_file, expected_hash=EXPECTED_DATASET_SHA256)


def test_record_formatting_success_and_exact_context():
    """Verify exact formatting and character-for-character context preservation for successful inspection."""
    scenario = {
        "id": "test_q_01",
        "question": "What is the B.Tech branch change policy?",
        "group": "A_known_anchor",
        "is_control": False,
        "intents": [
            {
                "intent_index": 0,
                "gold_sources": ["ordinance/btech.docx"],
                "facets": ["CGPA 8.5", "Semester II"],
            }
        ],
    }

    raw_context = (
        "Evidence 1 [Academics - B.Tech]\n"
        "Students with CGPA 8.5 or higher at the end of Semester II may apply for branch change.\n\n"
        "Evidence 2 [Academics - Reservation]\n"
        "SC/ST/PwD category students require CGPA 7.5."
    )

    diag = {
        "question": scenario["question"],
        "resolved_question": scenario["question"],
        "pipeline_stop": False,
        "is_multi_intent": False,
        "evidence_status": "supported",
        "evidence_coverage_status": "available",
        "ready_for_generation": True,
        "will_abstain": False,
        "abstention_reason": None,
        "fallback_response": None,
        "final_evidence_context": raw_context,
        "context_character_count": len(raw_context),
        "estimated_token_count": len(raw_context) // 4,
        "evidence_units": [
            {
                "evidence_id": "unit_1",
                "source": "ordinance/btech.docx",
                "heading": "Academics - B.Tech",
                "text": "Students with CGPA 8.5 or higher at the end of Semester II may apply for branch change.",
            }
        ],
        "evidence_unit_count": 1,
        "package_reasons": (),
        "omitted_units_count": 0,
        "omitted_claim_ids": (),
        "rejected_candidate_count": 0,
    }

    txt = format_txt_record(scenario, batch_num=1, diag=diag)

    # Verifications
    assert "QUESTION ID: test_q_01" in txt
    assert "BATCH: 1" in txt
    assert "BENCHMARK GROUP: A_known_anchor" in txt
    assert "STATUS: SUCCESS" in txt
    assert "ANSWER: NOT GENERATED — ZERO-INFERENCE MODE" in txt
    assert "ORIGINAL QUESTION:\nWhat is the B.Tech branch change policy?" in txt
    assert "INTENT CLASSIFICATION:\nSingle-Intent" in txt
    assert raw_context in txt
    assert f"- Character count: {len(raw_context)}" in txt
    assert "- Ready for generation: True" in txt
    assert "- Will abstain: False" in txt


def test_record_formatting_empty_context_abstention():
    """Verify that abstentions explicitly print the required EMPTY notice."""
    scenario = {
        "id": "ctrl_q_99",
        "question": "What is the recipe for butter chicken?",
        "group": "control",
        "is_control": True,
    }

    diag = {
        "question": scenario["question"],
        "resolved_question": scenario["question"],
        "pipeline_stop": True,
        "stop_reason": "unrelated_cooking_recipe_request",
        "is_multi_intent": False,
        "ready_for_generation": False,
        "will_abstain": True,
        "abstention_reason": "out_of_scope",
        "fallback_response": "I am an institutional assistant and can only answer college-related questions.",
        "final_evidence_context": "",
        "context_character_count": 0,
        "estimated_token_count": 0,
        "evidence_units": [],
        "evidence_unit_count": 0,
    }

    txt = format_txt_record(scenario, batch_num=2, diag=diag)

    assert "QUESTION ID: ctrl_q_99" in txt
    assert "STATUS: ABSTAIN" in txt
    assert "ANSWER: NOT GENERATED — ZERO-INFERENCE MODE" in txt
    assert "FINAL CONTEXT: EMPTY — PIPELINE ABSTAINED OR FAILED CLOSED" in txt
    assert "- Character count: 0" in txt
    assert "- Ready for generation: False" in txt
    assert "- Will abstain: True" in txt


def test_audit_execution_and_resumption(tmp_path: Path):
    """Test full audit flow on a mock dataset verifying file generation and safe resumption."""
    # Create mock dataset with 3 scenarios
    mock_data = [
        {
            "id": f"mock_q_{i:02d}",
            "question": f"Sample question number {i}?",
            "group": "A_known_anchor",
            "is_control": (i == 2),
            "chat_history": [],
            "intents": [],
        }
        for i in range(1, 4)
    ]
    mock_dataset_path = tmp_path / "mock_benchmark.json"
    mock_dataset_path.write_text(json.dumps(mock_data, indent=2), encoding="utf-8")
    mock_hash = compute_sha256(mock_dataset_path)

    out_dir = tmp_path / "audit_output"

    call_counts = {"count": 0}

    def mock_inspect(q: str, **kwargs: Any) -> dict[str, Any]:
        call_counts["count"] += 1
        return {
            "question": q,
            "resolved_question": q,
            "pipeline_stop": False,
            "is_multi_intent": False,
            "evidence_status": "supported",
            "ready_for_generation": True,
            "will_abstain": False,
            "final_evidence_context": f"Evidence for {q}",
            "context_character_count": len(f"Evidence for {q}"),
            "estimated_token_count": len(f"Evidence for {q}") // 4,
            "evidence_units": [{"evidence_id": "u1", "source": "src1.docx", "heading": "H1", "text": "evidence text"}],
            "evidence_unit_count": 1,
        }

    # Run 1: process first 2 questions
    summary1 = run_audit(
        benchmark_path=mock_dataset_path,
        out_dir=out_dir,
        batch_selector="all",
        resume=True,
        max_questions=2,
        inspect_fn=mock_inspect,
        expected_hash=mock_hash,
    )
    assert call_counts["count"] == 2
    assert summary1["total_completed_questions"] == 2

    # Verify results.jsonl and progress.json
    results_jsonl = out_dir / "results.jsonl"
    assert results_jsonl.exists()
    lines = results_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # Run 2: resume and finish remaining
    summary2 = run_audit(
        benchmark_path=mock_dataset_path,
        out_dir=out_dir,
        batch_selector="all",
        resume=True,
        inspect_fn=mock_inspect,
        expected_hash=mock_hash,
    )
    # Only 1 additional call should have been made!
    assert call_counts["count"] == 3
    assert summary2["total_completed_questions"] == 3

    # Check results.jsonl now has exactly 3 records, no duplicates!
    lines_after = results_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after) == 3
    ids = [json.loads(l)["id"] for l in lines_after]
    assert ids == ["mock_q_01", "mock_q_02", "mock_q_03"]

    # Verify generated files
    assert (out_dir / "all_180_questions_final_context.txt").exists()
    assert (out_dir / "audit_summary.txt").exists()
    assert (out_dir / "audit_summary.json").exists()


def test_error_handling_and_continuation(tmp_path: Path):
    """Test that a diagnostic exception is recorded as ERROR and audit proceeds to next questions."""
    mock_data = [
        {"id": "q1", "question": "Question 1?", "group": "A", "is_control": False, "intents": []},
        {"id": "q2", "question": "Question 2 with error?", "group": "A", "is_control": False, "intents": []},
        {"id": "q3", "question": "Question 3?", "group": "A", "is_control": False, "intents": []},
    ]
    mock_dataset_path = tmp_path / "mock_dataset.json"
    mock_dataset_path.write_text(json.dumps(mock_data), encoding="utf-8")

    out_dir = tmp_path / "error_test_out"

    def flaky_inspect(q: str, **kwargs: Any) -> dict[str, Any]:
        if "error" in q:
            raise RuntimeError("Synthetic vectorstore connection failure")
        return {
            "question": q,
            "resolved_question": q,
            "pipeline_stop": False,
            "is_multi_intent": False,
            "ready_for_generation": True,
            "will_abstain": False,
            "final_evidence_context": "Valid evidence",
            "context_character_count": 14,
            "estimated_token_count": 3,
            "evidence_units": [],
            "evidence_unit_count": 0,
        }

    summary = run_audit(
        benchmark_path=mock_dataset_path,
        out_dir=out_dir,
        batch_selector="all",
        resume=False,
        inspect_fn=flaky_inspect,
        expected_hash=None,
    )

    # All 3 scenarios should have completed records
    assert summary["total_completed_questions"] == 3
    assert summary["counts"]["diagnostic_errors"] == 1
    assert summary["counts"]["success_ready_for_generation"] == 2

    # Check failures_and_warnings.txt has record of q2
    failures_file = out_dir / "failures_and_warnings.txt"
    assert failures_file.exists()
    fail_content = failures_file.read_text(encoding="utf-8")
    assert "q2" in fail_content
    assert "Synthetic vectorstore connection failure" in fail_content
