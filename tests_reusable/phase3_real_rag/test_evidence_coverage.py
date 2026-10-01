"""Hard deterministic tests for reusable evidence coverage/completeness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "_env"
sys.path.insert(0, str(ENV))
sys.path.insert(0, str(ROOT))

from backend.core.evidence.coverage import (  # noqa: E402
    CoverageAssessment,
    assess_coverage,
    is_coverage_sufficient,
)
from backend.core.query.models import (  # noqa: E402
    ListIntent,
    NumericRequirement,
    Query,
    Target,
    TemporalConstraint,
)
from backend.core.retrieval_contracts import (  # noqa: E402
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
)


@dataclass
class Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def candidate(
    text: str,
    *,
    source: str = "source-a.docx",
    semantic: float = 0.9,
    coverage: float = 0.9,
    topic: float = 0.9,
    intent: float = 0.9,
    target: float = 0.9,
    attribute: float = 0.9,
    scope: float = 0.9,
    programs: tuple[str, ...] = (),
    entities: tuple[str, ...] = (),
) -> RetrievalCandidate:
    document = Doc(text, {"source": source})
    return RetrievalCandidate.from_document(
        document,
        meaning=DocumentMeaning(
            programs=programs,
            entities=entities,
        ),
        alignment=CandidateAlignment(
            program_match=target if programs else 0.0,
            entity_match=target if entities else 0.0,
            topic_match=topic,
            attribute_match=attribute,
            intent_match=intent,
            scope_match=scope,
            coverage=coverage,
            semantic_match=semantic,
        ),
        quality=EvidenceQuality(
            content_quality=1.0,
            structural_quality=1.0,
            noise=0.0,
        ),
        source=source,
    )


def test_empty_evidence_is_insufficient():
    result = assess_coverage(Query("What is the admission process?"), [])
    assert result.status == "insufficient"
    assert result.relevant_documents == 0
    assert not is_coverage_sufficient(result)


def test_strong_descriptive_candidate_is_supported():
    query = Query(
        "What is the admission process?",
        request_type="admission",
        target=Target("admission", confidence=1.0),
    )
    result = assess_coverage(
        query,
        [candidate("The admission process starts with online application submission.")],
    )
    assert result.status == "supported"
    assert result.strong_documents == 1
    assert is_coverage_sufficient(result)


def test_weak_candidate_is_not_called_supported():
    query = Query("What is the admission process?", request_type="admission")
    result = assess_coverage(
        query,
        [candidate(
            "Some general information.",
            semantic=0.40,
            coverage=0.20,
            topic=0.20,
            intent=0.20,
            target=0.20,
            attribute=0.10,
            scope=0.10,
        )],
    )
    assert result.status == "insufficient"
    assert not is_coverage_sufficient(result)


def test_requirements_need_requirement_signals():
    query = Query(
        "What are the eligibility requirements?",
        request_type="eligibility",
    )
    result = assess_coverage(
        query,
        [candidate("The application deadline is announced online.")],
    )
    assert result.status in {"partial", "insufficient"}
    assert not is_coverage_sufficient(result)


def test_requirements_with_direct_qualification_evidence_are_supported():
    query = Query(
        "What are the eligibility requirements?",
        request_type="eligibility",
    )
    result = assess_coverage(
        query,
        [candidate(
            "Eligibility requires a bachelor's degree and the published admission criteria.",
        )],
    )
    assert result.status == "supported"


def test_numeric_question_requires_numeric_evidence():
    query = Query(
        "What is the hostel fee?",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    missing = assess_coverage(
        query,
        [candidate("Hostel fees depend on the applicable category.")],
    )
    assert missing.status in {"partial", "insufficient"}

    present = assess_coverage(
        query,
        [candidate("The hostel fee is Rs 42,000 per year.")],
    )
    assert present.status == "supported"


def test_temporal_question_requires_requested_period_evidence():
    query = Query(
        "What is the fee for 2027?",
        temporal_constraints=(TemporalConstraint(kind="year", value="2027", confidence=1.0),),
    )
    missing = assess_coverage(
        query,
        [candidate("The published fee schedule is under review for a later session.")],
    )
    assert missing.status in {"partial", "insufficient"}

    present = assess_coverage(
        query,
        [candidate("The fee schedule for 2027 is Rs 42,000 per year.")],
    )
    assert present.status == "supported"


def test_list_question_needs_breadth_not_one_semantic_match():
    query = Query(
        "What programs are offered?",
        list_intent=ListIntent(is_list=True, item_type="program"),
    )
    result = assess_coverage(
        query,
        [candidate("The institute offers B.Tech in Electrical Engineering.", programs=("btech",))],
    )
    assert result.status == "partial"
    assert result.inventory_items >= 1


def test_list_strong_single_inventory_is_supported():
    query = Query(
        "What programs are offered?",
        list_intent=ListIntent(is_list=True, item_type="program"),
    )
    text = (
        "Programs:\n"
        "- B.Tech in Electrical Engineering\n"
        "- B.Tech in Computer Science\n"
        "- M.Tech in AI\n"
        "- M.Sc in Physics\n"
        "- Ph.D. in Chemistry\n"
    )
    result = assess_coverage(
        query,
        [candidate(text, programs=("btech", "mtech", "msc", "phd"))],
    )
    assert result.status == "supported"
    assert result.inventory_items >= 5


def test_list_same_source_duplicates_do_not_create_breadth_by_duplication():
    query = Query(
        "What programs are offered?",
        list_intent=ListIntent(is_list=True, item_type="program"),
    )
    text = "Programs:\n- B.Tech in Electrical Engineering\n- B.Tech in Computer Science\n"
    result = assess_coverage(
        query,
        [candidate(text), candidate(text, source="source-a.docx")],
    )
    assert result.unique_sources == 1
    assert result.inventory_items < 7
    assert result.status != "supported"


def test_distributed_list_evidence_can_be_supported():
    query = Query(
        "What departments are available?",
        list_intent=ListIntent(is_list=True, item_type="department"),
    )
    docs = [
        candidate(
            "Departments:\n- Electrical Engineering\n- Computer Science\n",
            source="departments-a.docx",
            entities=("Electrical Engineering", "Computer Science"),
        ),
        candidate(
            "Departments:\n- Mechanical Engineering\n- Mathematics\n",
            source="departments-b.docx",
            entities=("Mechanical Engineering", "Mathematics"),
        ),
        candidate(
            "Departments:\n- Physics\n",
            source="departments-c.docx",
            entities=("Physics",),
        ),
    ]
    result = assess_coverage(query, docs)
    assert result.status == "supported"
    assert result.unique_sources == 3


def test_multi_part_requires_all_required_units():
    query = Query(
        "Tell me the M.Tech admission eligibility and fee.",
        request_type="admission",
        target=Target("M.Tech", confidence=1.0),
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
        is_multi_part=True,
    )

    only_eligibility = assess_coverage(
        query,
        [candidate(
            "M.Tech admission eligibility requires a relevant bachelor's degree.",
            programs=("mtech",),
            attribute=0.9,
            target=0.9,
        )],
    )
    assert only_eligibility.status in {"partial", "insufficient"}
    assert any(unit.startswith("numeric:fee=") for unit in only_eligibility.uncovered_units)

    both = assess_coverage(
        query,
        [candidate(
            "M.Tech admission eligibility requires a relevant bachelor's degree. The fee is Rs 42,000 per year.",
            programs=("mtech",),
            attribute=0.9,
            target=0.9,
        )],
    )
    assert both.status == "supported"
    assert not both.uncovered_units


def test_multi_part_partial_evidence_is_explicitly_reported():
    query = Query(
        "Tell me the hostel rules and fee.",
        request_type="hostel",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
        is_multi_part=True,
    )
    result = assess_coverage(
        query,
        [candidate("Hostel residents must follow the published hostel rules.")],
    )
    assert result.status in {"partial", "insufficient"}
    assert any(unit.startswith("numeric:fee=") for unit in result.uncovered_units)


def test_exact_duplicate_content_from_different_sources_does_not_inflate_document_breadth():
    query = Query(
        "What programs are offered?",
        list_intent=ListIntent(is_list=True, item_type="program"),
    )
    text = (
        "Programs:\n"
        "- B.Tech in Electrical Engineering\n"
        "- B.Tech in Computer Science\n"
        "- M.Tech in AI\n"
    )
    result = assess_coverage(
        query,
        [
            candidate(text, source="a.docx"),
            candidate(text, source="b.docx"),
        ],
    )
    # Exact-content duplicates from separate sources remain two source records
    # for provenance, but the inventory must still be below the broad-list
    # threshold; duplication cannot manufacture a complete institutional list.
    assert result.inventory_items < 7
    assert result.status != "supported"


def test_to_dict_contract_is_stable():
    query = Query("What is the admission process?")
    result = assess_coverage(query, [candidate("The admission process starts online.")])
    payload = result.to_dict()
    assert set(payload) >= {
        "status",
        "score",
        "question_type",
        "strong_documents",
        "partial_documents",
        "relevant_documents",
        "unique_sources",
        "required_units",
        "covered_units",
        "uncovered_units",
        "reasons",
        "units",
    }


if __name__ == "__main__":
    tests = [
        value for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"E4 EVIDENCE COVERAGE HARD TESTS: PASS ({len(tests)} tests)")