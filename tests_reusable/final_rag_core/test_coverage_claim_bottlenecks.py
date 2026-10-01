from __future__ import annotations

from types import SimpleNamespace

from backend.core.evidence.claims import audit_claims, extract_claims, extract_evidence_units
from backend.core.evidence.coverage import assess_coverage
from backend.core.query.models import ListIntent, Query, Target
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
)


def candidate(
    text: str,
    *,
    source: str = "source.docx",
    programs: tuple[str, ...] = (),
    entities: tuple[str, ...] = (),
    score: float = 0.9,
    quality: tuple[float, float, float] = (1.0, 1.0, 0.0),
) -> RetrievalCandidate:
    doc = SimpleNamespace(
        page_content=text,
        metadata={"source": source},
    )
    content_quality, structural_quality, noise = quality
    return RetrievalCandidate.from_document(
        doc,
        meaning=DocumentMeaning(
            programs=programs,
            entities=entities,
        ),
        alignment=CandidateAlignment(
            semantic_match=score,
            coverage=score,
            topic_match=score,
            intent_match=score,
            entity_match=score,
            program_match=score,
            attribute_match=score,
            scope_match=score,
        ),
        quality=EvidenceQuality(
            content_quality=content_quality,
            structural_quality=structural_quality,
            noise=noise,
        ),
    )


def test_implicit_program_collection_is_detected_and_covered() -> None:
    query = Query(
        "What academic programs are available?",
        target=Target("academic programs", confidence=1.0),
        list_intent=ListIntent(is_list=False),
    )
    docs = [
        candidate("B.Tech overview", programs=("btech",), source="a.docx"),
        candidate("M.Tech overview", programs=("mtech",), source="b.docx"),
        candidate("M.Sc overview", programs=("msc",), source="c.docx"),
        candidate("Ph.D overview", programs=("phd",), source="d.docx"),
    ]
    result = assess_coverage(query, docs)
    assert result.question_type == "list"
    assert result.status == "supported", result.to_dict()
    assert result.inventory_items >= 4


def test_collection_breadth_works_when_semantic_inventory_is_flattened() -> None:
    query = Query(
        "What academic programs are available?",
        target=Target("academic programs", confidence=1.0),
        list_intent=ListIntent(is_list=False),
    )
    docs = [
        candidate("Program information section.", source="a.docx", programs=()),
        candidate("Program information section two.", source="b.docx", programs=()),
        candidate("Program information section three.", source="c.docx", programs=()),
        candidate("Program information section four.", source="d.docx", programs=()),
        candidate("Program information section five.", source="e.docx", programs=()),
        candidate("Program information section six.", source="f.docx", programs=()),
    ]
    result = assess_coverage(query, docs)
    assert result.question_type == "list"
    assert result.status == "supported", result.to_dict()


def test_procedure_question_has_focused_procedure_coverage() -> None:
    query = Query(
        "What is the procedure for applying?",
        request_type="procedure",
    )
    docs = [
        candidate(
            "The application process requires submission through the online portal.",
        )
    ]
    result = assess_coverage(query, docs)
    assert result.question_type == "procedure"
    assert result.status == "supported", result.to_dict()


def test_procedure_can_use_semantic_alignment_when_lexical_overlap_is_low() -> None:
    query = Query(
        "What is the procedure for applying?",
        request_type="procedure",
    )
    docs = [
        candidate(
            "Applicants submit the required information through the designated portal.",
            score=0.55,
            quality=(0.45, 0.45, 0.0),
        )
    ]
    result = assess_coverage(query, docs)
    assert result.question_type == "procedure"
    assert result.status == "supported", result.to_dict()


def test_location_question_has_focused_location_coverage() -> None:
    query = Query(
        "Where is the institute located?",
        request_type="directions",
    )
    docs = [
        candidate("The institute is located at the campus address.")
    ]
    result = assess_coverage(query, docs)
    assert result.question_type == "location"
    assert result.status == "supported", result.to_dict()


def test_location_can_use_location_fact_with_low_global_support() -> None:
    query = Query(
        "Where is the institute located?",
        request_type="directions",
    )
    docs = [
        candidate(
            "Located on the main campus road in the city.",
            score=0.55,
            quality=(0.45, 0.45, 0.0),
        )
    ]
    result = assess_coverage(query, docs)
    assert result.question_type == "location"
    assert result.status == "supported", result.to_dict()


def test_scoped_tuition_values_are_not_false_conflicts() -> None:
    query = Query(
        "What is the tuition fee?",
        request_type="cost",
        target=Target("tuition fee", confidence=1.0),
    )
    docs = [
        candidate("Tuition Fee ₹50,000 for B.Tech students.", source="btech.docx"),
        candidate("Tuition Fee ₹60,000 for M.Tech students.", source="mtech.docx"),
    ]
    audit = audit_claims(extract_claims(query), extract_evidence_units(docs))
    assert audit.status == "supported", audit.to_dict()
    assert audit.conflicting_claim_ids == ()


def test_same_scope_tuition_values_are_a_conflict() -> None:
    query = Query(
        "What is the tuition fee?",
        request_type="cost",
        target=Target("tuition fee", confidence=1.0),
    )
    docs = [
        candidate("Tuition Fee ₹50,000.", source="a.docx"),
        candidate("Tuition Fee ₹60,000.", source="b.docx"),
    ]
    audit = audit_claims(extract_claims(query), extract_evidence_units(docs))
    assert audit.status == "conflict", audit.to_dict()
    assert audit.conflicting_claim_ids


def test_tuition_question_ignores_admission_fee_value_in_same_table() -> None:
    query = Query(
        "What is the tuition fee?",
        request_type="cost",
        target=Target("tuition fee", confidence=1.0),
    )
    docs = [
        candidate(
            "Tuition Fee ₹50,000; Admission Fee ₹3,800.",
            source="fees.docx",
        )
    ]
    audit = audit_claims(extract_claims(query), extract_evidence_units(docs))
    assert audit.status == "supported", audit.to_dict()
    assert audit.conflicting_claim_ids == ()


def test_tuition_request_does_not_accept_unscoped_admission_fee_as_supported() -> None:
    query = Query(
        "What is the tuition fee?",
        request_type="cost",
        target=Target("tuition fee", confidence=1.0),
    )
    docs = [
        candidate("Admission Fee ₹3,800.", source="fees.docx")
    ]
    audit = audit_claims(extract_claims(query), extract_evidence_units(docs))
    assert audit.status == "insufficient", audit.to_dict()


def test_live_shape_procedure_is_detected_from_user_wording() -> None:
    query = Query("What is the procedure for applying?")
    docs = [candidate("The application process requires submission through the online portal.")]
    result = assess_coverage(query, docs)
    assert result.question_type == "procedure"
    assert result.status == "supported", result.to_dict()


def test_live_shape_location_is_detected_from_user_wording() -> None:
    query = Query("Where is the institute located?")
    docs = [candidate("The institute is located at the campus address.")]
    result = assess_coverage(query, docs)
    assert result.question_type == "location"
    assert result.status == "supported", result.to_dict()


def test_collection_expands_beyond_support_threshold_for_verified_breadth() -> None:
    query = Query("What academic programs are available?", target=Target("academic programs", confidence=1.0))
    docs = [
        candidate("Generic program information.", source=f"high{i}.docx", score=0.90)
        for i in range(5)
    ] + [
        candidate(f"Program overview {name}.", source=f"low{i}.docx", programs=(name,), score=0.35)
        for i, name in enumerate(("btech", "mtech", "msc", "phd"))
    ]
    result = assess_coverage(query, docs)
    assert result.question_type == "list"
    assert result.status == "supported", result.to_dict()
    assert result.inventory_items >= 4
