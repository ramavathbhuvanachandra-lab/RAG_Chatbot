"""Hard tests for generic evidence scope/conflict protection."""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Document:
    page_content: str
    metadata: dict | None = None

from backend.core.evidence.scope import (
    assess_scope,
    filter_scope_conflicts,
    is_scope_compatible,
)


def test_same_topic_is_compatible():
    query = "What are the admission requirements?"
    doc = Document(page_content="Admission requirements include a qualifying degree and minimum marks.")
    assert is_scope_compatible(query, doc)


def test_explicit_topic_conflict_is_rejected():
    query = "What are the hostel fees?"
    doc = Document(page_content="The admission requirements include a qualifying degree and minimum marks.")
    decision = assess_scope(query, doc)
    assert not decision.compatible
    assert "explicit_topic_mismatch" in decision.reasons


def test_explicit_program_mismatch_is_rejected():
    query = "What are the M.Tech admission requirements?"
    doc = Document(page_content="The M.Sc. admission requirements include a relevant bachelor's degree.")
    decision = assess_scope(query, doc)
    assert not decision.compatible
    assert "explicit_program_mismatch" in decision.reasons


def test_same_program_is_compatible():
    query = "What are the M.Tech admission requirements?"
    doc = Document(page_content="M.Tech admission requires a qualifying bachelor's degree.")
    assert is_scope_compatible(query, doc)


def test_broad_query_allows_narrow_department_evidence():
    query = "What research areas are available?"
    doc = Document(page_content="Department of Electrical Engineering research areas include VLSI and control systems.")
    assert is_scope_compatible(query, doc)


def test_explicit_department_mismatch_is_rejected():
    query = "What research areas are available in Electrical Engineering?"
    doc = Document(page_content="Department of Mechanical Engineering research areas include thermofluids.")
    decision = assess_scope(query, doc)
    assert not decision.compatible
    assert "explicit_organization_mismatch" in decision.reasons


def test_same_department_is_compatible():
    query = "What research areas are available in Electrical Engineering?"
    doc = Document(page_content="Department of Electrical Engineering research areas include VLSI and signal processing.")
    assert is_scope_compatible(query, doc)


def test_explicit_mode_mismatch_is_rejected():
    query = "What are the regular Ph.D. admission requirements?"
    doc = Document(page_content="The part-time Ph.D. admission route has separate requirements.")
    decision = assess_scope(query, doc)
    assert not decision.compatible
    assert "explicit_mode_mismatch" in decision.reasons


def test_exact_year_range_mismatch_is_rejected():
    query = "What are the fees for AY 2026-2027?"
    doc = Document(page_content="The fees for AY 2025-2026 are listed below.")
    decision = assess_scope(query, doc)
    assert not decision.compatible
    assert "explicit_year_range_mismatch" in decision.reasons


def test_year_inside_document_range_is_compatible_for_ordinary_query():
    query = "What were the fees in 2026?"
    doc = Document(page_content="The fees for AY 2026-2027 are listed below.")
    assert is_scope_compatible(query, doc)


def test_unknown_terms_are_neutral():
    query = "What is the zorbax rule for students?"
    doc = Document(page_content="Student rules and regulations are published by the institution.")
    assert is_scope_compatible(query, doc)


def test_filter_preserves_order_and_removes_only_conflicts():
    query = "What are the M.Tech admission requirements?"
    docs = [
        Document(page_content="M.Sc. admission requirements."),
        Document(page_content="M.Tech admission requirements."),
        Document(page_content="Admission information."),
    ]
    kept = filter_scope_conflicts(query, docs)
    assert len(kept) == 2
    assert kept[0].page_content == "M.Tech admission requirements."
    assert kept[1].page_content == "Admission information."


def test_structured_semantics_can_add_a_target_signal():
    @dataclass
    class Meaning:
        programs: tuple[str, ...] = ("mtech",)
        entities: tuple[str, ...] = ()
        topics: tuple[str, ...] = ("admission",)
        scope: tuple[str, ...] = ()
        qualifiers: tuple[str, ...] = ()

    query = "What are the admission requirements?"
    doc = Document(page_content="M.Sc. admission requirements.")
    decision = assess_scope(query, doc, query_semantics=Meaning())
    assert not decision.compatible
    assert "explicit_program_mismatch" in decision.reasons


def run_tests() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"E3 EVIDENCE SCOPE HARD TESTS: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    run_tests()
