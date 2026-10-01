"""Reusable regression tests for semantic query understanding backstops."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.query.understanding import (  # noqa: E402
    SemanticInterpretation,
    understand_query,
)


class FakeSemanticModel:
    """Return a controlled structured interpretation without network/LLM calls."""

    def __init__(self, response: dict):
        self.response = response
        self.calls = 0
        self.payloads = []

    def invoke(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        return self.response


class FakeSafetyModel:
    """Safety model that records calls and approves the supplied interpretation."""

    def __init__(self, approved: bool = True):
        self.approved = approved
        self.calls = 0
        self.payloads = []

    def invoke(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        return SimpleNamespace(approved=self.approved, reason="test")


def _run(
    query: str,
    response: dict,
):
    semantic = FakeSemanticModel(response)
    safety = FakeSafetyModel()
    result = understand_query(
        query,
        semantic_model=semantic,
        safety_model=safety,
    )
    return result, semantic, safety


def test_list_backstop_recovers_department_collection():
    result, semantic, _ = _run(
        "What departments are there?",
        {
            "target": "departments",
            "request_type": None,
            "semantic_query": "departments",
            "confidence": 0.0,
        },
    )

    assert result.list_intent.is_list is True
    assert result.target is not None
    assert result.target.text == "departments"
    assert semantic.calls == 1


def test_list_backstop_recovers_academic_program_collection():
    result, _, _ = _run(
        "What academic programs are available?",
        {
            "target": "academic programs",
            "request_type": None,
            "semantic_query": "academic programs",
            "confidence": 0.0,
        },
    )

    assert result.list_intent.is_list is True
    assert result.target.text == "academic programs"


def test_list_backstop_recovers_facility_collection():
    result, _, _ = _run(
        "What facilities are available?",
        {
            "target": "facilities",
            "request_type": None,
            "semantic_query": "facilities",
            "confidence": 0.0,
        },
    )

    assert result.list_intent.is_list is True
    assert result.target.text == "facilities"


def test_list_backstop_recovers_research_area_collection():
    result, _, _ = _run(
        "What research areas are available?",
        {
            "target": "research areas",
            "request_type": None,
            "semantic_query": "research areas",
            "confidence": 0.0,
        },
    )

    assert result.list_intent.is_list is True
    assert result.request_type == "research"
    assert result.target.text == "research areas"


def test_list_backstop_recovers_research_opportunity_collection():
    result, _, _ = _run(
        "What research opportunities are available?",
        {
            "target": "research opportunities",
            "request_type": None,
            "semantic_query": "research opportunities",
            "confidence": 0.0,
        },
    )

    assert result.list_intent.is_list is True
    assert result.request_type == "research"
    assert result.target.text == "research opportunities"


def test_eligibility_backstop_adds_requested_attribute():
    result, _, _ = _run(
        "What are the M.Tech eligibility requirements?",
        {
            "target": "M.Tech",
            "request_type": None,
            "semantic_query": "M.Tech eligibility",
            "confidence": 0.0,
        },
    )

    assert result.request_type == "eligibility"
    assert result.target.text == "M.Tech"
    assert result.retrieval_requirement.require_attribute_alignment is True
    assert result.confidence >= 0.92
    assert result.retrieval_requirement.mode == "exact"
    assert result.retrieval_requirement.require_target_alignment is True
    assert result.retrieval_requirement.require_attribute_alignment is True


def test_cost_backstop_distinguishes_admission_fee_from_generic_admission():
    result, _, _ = _run(
        "What is the admission fee?",
        {
            "target": "admission",
            "request_type": None,
            "semantic_query": "admission",
            "confidence": 0.0,
        },
    )

    assert result.request_type == "cost"
    assert result.target.text == "admission fee"
    assert result.retrieval_requirement.require_attribute_alignment is True
    assert result.retrieval_requirement.mode == "exact"


def test_hostel_fee_target_is_specific_and_not_generic_fee():
    result, _, _ = _run(
        "What are the hostel fees for students?",
        {
            "target": "hostel fees",
            "request_type": None,
            "semantic_query": "hostel fees",
            "confidence": 0.0,
            "qualifiers": ["students"],
        },
    )

    assert result.request_type == "cost"
    assert result.target.text == "hostel fees"
    assert result.retrieval_requirement.require_attribute_alignment is True


def test_application_procedure_gets_procedure_intent():
    result, _, _ = _run(
        "How do I apply for admission?",
        {
            "target": None,
            "request_type": None,
            "semantic_query": "procedure",
            "confidence": 0.0,
        },
    )

    assert result.request_type == "procedure"
    assert result.target.text == "admission"
    assert result.retrieval_requirement.require_attribute_alignment is True


def test_location_question_gets_directions_intent():
    result, _, _ = _run(
        "Where is the institute located?",
        {
            "target": "institute",
            "request_type": None,
            "semantic_query": "institute",
            "confidence": 0.0,
        },
    )

    assert result.request_type == "directions"
    assert result.target.text == "institute"
    assert result.retrieval_requirement.require_attribute_alignment is True


def test_application_deadline_target_is_preserved():
    result, _, _ = _run(
        "What are the application deadlines?",
        {
            "target": "application deadlines",
            "request_type": None,
            "semantic_query": "application deadlines",
            "confidence": 0.0,
        },
    )

    assert result.request_type == "admission"
    assert result.target.text == "application deadlines"
    assert result.list_intent.is_list is False
    assert result.retrieval_requirement.mode == "exact"


def test_existing_specific_llm_request_type_is_preserved():
    result, _, _ = _run(
        "Tell me about admission routes for M.Sc.",
        {
            "target": "admission routes",
            "request_type": "admission",
            "semantic_query": "admission routes M.Sc",
            "confidence": 0.88,
        },
    )

    assert result.request_type == "admission"
    assert result.target.text == "M.Sc"


def test_generic_information_request_type_is_replaced_by_strong_cost_signal():
    result, _, _ = _run(
        "What is the tuition fee?",
        {
            "target": "tuition fee",
            "request_type": "information",
            "semantic_query": "tuition fee",
            "confidence": 0.40,
        },
    )

    assert result.request_type == "cost"
    assert result.confidence >= 0.92


def test_generic_information_target_is_replaced_by_literal_target():
    result, _, _ = _run(
        "What departments are there?",
        {
            "target": "information",
            "request_type": "information",
            "semantic_query": "information",
            "confidence": 0.20,
        },
    )

    assert result.target.text == "departments"
    assert result.list_intent.is_list is True


def test_clear_query_can_skip_second_safety_llm():
    result, semantic, safety = _run(
        "What are the M.Tech eligibility requirements?",
        {
            "target": "M.Tech",
            "request_type": None,
            "semantic_query": "M.Tech eligibility",
            "confidence": 0.0,
        },
    )

    assert result.semantic_grounded is True
    assert semantic.calls == 1
    assert safety.calls == 0


def test_ambiguous_query_can_still_use_safety_verifier():
    result, semantic, safety = _run(
        "What is it?",
        {
            "target": "it",
            "request_type": None,
            "semantic_query": "it",
            "confidence": 0.10,
            "ambiguous": True,
        },
    )

    assert semantic.calls == 1
    assert safety.calls == 1
    assert result.verification_status == "approved"


def test_original_question_remains_authoritative():
    result, _, _ = _run(
        "What are the departments?",
        {
            "target": "some invented department",
            "request_type": "information",
            "semantic_query": "some invented department",
            "confidence": 0.99,
        },
    )

    assert result.target is not None
    assert result.target.text == "departments"
    assert "some invented department" not in result.search_query


def test_unknown_terms_not_in_query_are_dropped():
    result, _, _ = _run(
        "What departments are there?",
        {
            "target": "departments",
            "request_type": None,
            "semantic_query": "departments",
            "confidence": 0.90,
            "unknown_terms": ["zorpulax"],
        },
    )

    assert result.unknown_terms == ()


def test_deterministic_backstop_does_not_introduce_institution_specific_names():
    source = ROOT / "backend" / "core" / "query" / "understanding.py"
    text = source.read_text(encoding="utf-8").casefold()

    forbidden = (
        "iit jodhpur",
        "iitj",
        "iitj.ac.in",
        "mtech_admissions.docx",
    )

    found = [item for item in forbidden if item in text]
    assert not found, found


def test_semantic_interpretation_round_trip_remains_valid():
    frame = SemanticInterpretation(
        target="departments",
        request_type=None,
        semantic_query="departments",
        confidence=0.0,
    )

    assert frame.target == "departments"
    assert frame.semantic_query == "departments"


def test_empty_query_is_rejected():
    try:
        understand_query("   ")
    except ValueError as exc:
        assert "query cannot be empty" in str(exc)
    else:
        raise AssertionError("empty query must be rejected")