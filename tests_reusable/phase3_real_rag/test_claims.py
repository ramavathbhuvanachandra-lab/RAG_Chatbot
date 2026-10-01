"""E5 hard tests for generic claim extraction and claim/evidence auditing."""

from types import SimpleNamespace

from backend.core.evidence.claims import (
    Claim,
    factual_markers,
    EvidenceUnit,
    audit_claims,
    extract_claims,
    extract_evidence_units,
    extract_factual_markers,
    match_claim_to_evidence,
    split_evidence_units,
)
from backend.core.query.models import (
    Constraint,
    ListIntent,
    NumericRequirement,
    Query,
    Relation,
    Target,
    TemporalConstraint,
)
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    EvidenceQuality,
    RetrievalCandidate,
)


def candidate(text: str, source: str = "example.docx") -> RetrievalCandidate:
    doc = SimpleNamespace(page_content=text, metadata={"source": source})
    return RetrievalCandidate.from_document(
        doc,
        source=source,
        meaning=None,
        alignment=CandidateAlignment(),
        quality=EvidenceQuality(content_quality=1.0, structural_quality=1.0, noise=0.0),
    )


def test_numeric_claim_is_extracted_without_domain_vocabulary():
    query = Query(
        "What is the accommodation charge?",
        numeric_requirements=(NumericRequirement("charge", 1, unit="currency", confidence=1.0),),
    )
    claims = extract_claims(query)
    numeric = [claim for claim in claims if claim.kind == "numeric"]
    assert len(numeric) == 1
    assert numeric[0].name == "charge"
    assert numeric[0].unit == "currency"


def test_temporal_claim_is_extracted_structurally():
    query = Query(
        "What is the policy for 2027?",
        temporal_constraints=(TemporalConstraint(kind="year", value="2027", confidence=1.0),),
    )
    claims = extract_claims(query)
    assert any(claim.kind == "temporal" and claim.value == "2027" for claim in claims)


def test_relation_claim_uses_subject_relation_object():
    query = Query(
        "Which unit offers this program?",
        relations=(Relation("offers", "unit", "program", confidence=1.0),),
    )
    claims = extract_claims(query)
    relation = next(claim for claim in claims if claim.kind == "relation")
    assert relation.name == "offers"
    assert relation.value == ("unit", "program")


def test_list_claim_does_not_define_institution_inventory():
    query = Query(
        "What departments are available?",
        list_intent=ListIntent(is_list=True, item_type="department"),
    )
    claims = extract_claims(query)
    list_claim = next(claim for claim in claims if claim.kind == "list")
    assert list_claim.name == "department"
    assert "college" not in list_claim.text.casefold()


def test_factual_marker_extraction_is_normalized():
    markers = extract_factual_markers("The fee is ₹ 42,000 and the minimum is 60% for 2027.")
    assert "rs 42,000" in markers
    assert "60%" in markers
    assert "2027" in markers


def test_numeric_currency_claim_rejects_unrelated_percentage():
    query = Query(
        "What is the fee?",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    claim = next(item for item in extract_claims(query) if item.kind == "numeric")
    evidence = EvidenceUnit("e1", "Applicants need a minimum of 60% marks.", "a.docx", 0)
    result = match_claim_to_evidence(claim, evidence)
    assert result.status == "unsupported"


def test_numeric_currency_claim_requires_local_field_attachment():
    query = Query(
        "What is the fee?",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    claim = next(item for item in extract_claims(query) if item.kind == "numeric")
    unrelated = EvidenceUnit("e1", "A deposit of Rs 42,000 is required.", "a.docx", 0)
    result = match_claim_to_evidence(claim, unrelated)
    assert result.status == "unsupported"


def test_numeric_query_context_prevents_cross_scope_fee_match():
    query = Query(
        "What is the hostel fee?",
        target=Target("hostel", confidence=1.0, resolution_state="resolved"),
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    claim = next(item for item in extract_claims(query) if item.kind == "numeric")
    evidence = EvidenceUnit("e1", "The registration fee is Rs 1,000.", "registration.docx", 0)
    result = match_claim_to_evidence(claim, evidence)
    assert result.status == "unsupported"


def test_numeric_query_context_accepts_locally_attached_fee():
    query = Query(
        "What is the hostel fee?",
        target=Target("hostel", confidence=1.0, resolution_state="resolved"),
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    claim = next(item for item in extract_claims(query) if item.kind == "numeric")
    evidence = EvidenceUnit("e1", "The hostel fee is Rs 42,000 per year.", "fees.docx", 0)
    result = match_claim_to_evidence(claim, evidence)
    assert result.status == "supported"


def test_empty_factual_marker_groups_are_not_treated_as_present():
    assert factual_markers("Tuition Fee*") == {
        "money": (),
        "percentages": (),
        "ratios": (),
        "years": (),
        "academic_years": (),
        "durations": (),
    }


def test_docx_table_cell_value_keeps_local_fee_attachment():
    query = Query(
        "What is the fee?",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    # DOCX text extractors can serialize table cells as separate lines with
    # blank lines between them. The evidence layer must reunite a field label
    # with its immediately following monetary value.
    c = candidate(
        "Fee Structure\n\n(AY 2026-2027)\n\nA.\n\nTuition Fee*\n\n₹50,000/-"
    )
    units = split_evidence_units(c)
    assert any("tuition fee* ₹50,000/-" in unit.text.casefold() for unit in units), [u.text for u in units]
    audit = audit_claims(extract_claims(query), units)
    numeric = next(item for item in audit.claims if item.kind == "numeric")
    assert numeric.claim_id in audit.supported_claim_ids, audit.to_dict()


def test_temporal_claim_rejects_wrong_year():
    query = Query(
        "What is the fee for 2099?",
        temporal_constraints=(TemporalConstraint(kind="year", value="2099", confidence=1.0),),
    )
    claim = next(item for item in extract_claims(query) if item.kind == "temporal")
    evidence = EvidenceUnit("e1", "The fee schedule for 2026-2027 is published.", "fees.docx", 0)
    result = match_claim_to_evidence(claim, evidence)
    assert result.status == "unsupported"


def test_relation_claim_requires_colocated_semantic_parts():
    claim = Claim(
        claim_id="relation_1",
        text="unit offers program",
        kind="relation",
        name="offers",
        value=("unit", "program"),
        anchors=("unit", "offers", "program"),
    )
    good = EvidenceUnit("e1", "The engineering unit offers the program.", "a.docx", 0)
    bad = EvidenceUnit("e2", "The unit is described here. The separate document lists a program.", "a.docx", 0)
    assert match_claim_to_evidence(claim, good).status == "supported"
    assert match_claim_to_evidence(claim, bad).status != "supported"


def test_evidence_segmentation_preserves_heading_and_bullets():
    c = candidate("Research Themes:\n- Robust control\n- Signal processing\n- Robotics")
    units = split_evidence_units(c)
    assert len(units) >= 3
    assert all(unit.heading == "Research Themes" for unit in units)


def test_evidence_units_are_exact_content_deduplicated():
    c1 = candidate("The fee is Rs 42,000.", "a.docx")
    c2 = candidate("The fee is Rs 42,000.", "a.docx")
    units = extract_evidence_units([c1, c2])
    assert len(units) == 1


def test_audit_reports_partial_when_one_required_claim_is_missing():
    query = Query(
        "Tell me the fee and the policy year.",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
        temporal_constraints=(TemporalConstraint(kind="year", value="2027", confidence=1.0),),
    )
    claims = extract_claims(query)
    evidence = extract_evidence_units([candidate("The fee is Rs 42,000.")])
    result = audit_claims(claims, evidence)
    assert result.status == "partial"
    assert result.unsupported_claim_ids


def test_audit_supported_when_all_required_claims_have_evidence():
    query = Query(
        "Tell me the fee and the policy year.",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
        temporal_constraints=(TemporalConstraint(kind="year", value="2027", confidence=1.0),),
    )
    claims = extract_claims(query)
    evidence = extract_evidence_units([candidate("The fee for 2027 is Rs 42,000.")])
    result = audit_claims(claims, evidence)
    assert result.status == "supported"


def test_constraint_claim_is_structural():
    query = Query(
        "What is the rule?",
        constraints=(Constraint("mode", "remote", operator="eq", required=True, importance=1.0),),
    )
    claims = extract_claims(query)
    claim = next(item for item in claims if item.kind == "constraint")
    assert claim.name == "mode"
    assert claim.value == "remote"


def test_request_level_claim_preserves_request_type_without_domain_taxonomy():
    query = Query("What are the eligibility requirements?", request_type="eligibility")
    claims = extract_claims(query)
    request = next(item for item in claims if item.kind == "request")
    assert request.name == "eligibility"
    assert "eligibility" in request.anchors


def test_currency_claim_accepts_generic_rent_alias_with_scope():
    query = Query(
        "What is the hostel fee?",
        target=Target("hostel", confidence=1.0, resolution_state="resolved"),
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    claim = next(item for item in extract_claims(query) if item.kind == "numeric")
    evidence = EvidenceUnit("e1", "Hostel room rent is Rs 500 per day.", "finance.docx", 0)
    assert match_claim_to_evidence(claim, evidence).status == "supported"


def test_temporal_claim_can_match_temporal_value_in_heading():
    query = Query(
        "What is the fee for AY 2026-2027?",
        temporal_constraints=(TemporalConstraint(kind="academic_year", start="2026", end="2027", confidence=1.0),),
    )
    claim = next(item for item in extract_claims(query) if item.kind == "temporal")
    evidence = EvidenceUnit(
        "e1", "Tuition Fee Rs 50,000.", "finance.docx", 0,
        heading="Fee Structure (AY 2026-2027)",
    )
    assert match_claim_to_evidence(claim, evidence).status == "supported"


def test_multi_value_fee_unit_is_not_false_positive_conflict():
    claims = [
        Claim("numeric_fee_1", "fee eq 1 currency", "numeric", name="fee", value=1, unit="currency", anchors=("fee",)),
    ]
    evidence = [
        EvidenceUnit("e1", "Tuition Fee Rs 50,000; Admission Fee Rs 3,800; Semester Fee Rs 32,250.", "a.docx", 0),
        EvidenceUnit("e2", "Tuition Fee Rs 55,000; Admission Fee Rs 4,000.", "b.docx", 0),
    ]
    result = audit_claims(claims, evidence)
    assert result.conflicting_claim_ids == ()


def test_request_level_claim_matches_structured_request_type_in_heading():
    query = Query(
        "What are the M.Tech eligibility requirements?",
        request_type="eligibility",
    )
    claim = next(item for item in extract_claims(query) if item.kind == "request")
    evidence = EvidenceUnit(
        "e1",
        "The applicant must have a bachelor's degree in engineering or science.",
        "admissions.docx",
        0,
        heading="Eligibility Requirements",
    )
    result = match_claim_to_evidence(claim, evidence)
    assert result.status == "supported", result.to_dict()
    assert result.score >= 0.82


def test_request_level_claim_accepts_simple_morphological_variant():
    query = Query("What are the eligibility requirements?", request_type="eligibility")
    claim = next(item for item in extract_claims(query) if item.kind == "request")
    evidence = EvidenceUnit("e1", "Applicants are eligible when they meet the stated criteria.", "a.docx", 0)
    assert match_claim_to_evidence(claim, evidence).status == "supported"


def test_request_level_claim_matches_indirect_real_requirement_language():
    query = Query(
        "What are the M.Tech eligibility requirements?",
        request_type="eligibility",
    )
    claim = next(item for item in extract_claims(query) if item.kind == "request")
    evidence = EvidenceUnit(
        "e1",
        "The applicant must have a bachelor's degree in engineering or science.",
        "mtech_admissions.docx",
        0,
    )
    result = match_claim_to_evidence(claim, evidence)
    assert result.status == "supported", result.to_dict()


def test_request_level_claim_does_not_treat_any_must_sentence_as_eligibility():
    query = Query("What are the eligibility requirements?", request_type="eligibility")
    claim = next(item for item in extract_claims(query) if item.kind == "request")
    evidence = EvidenceUnit("e1", "Students must return equipment after use.", "policy.docx", 0)
    assert match_claim_to_evidence(claim, evidence).status != "supported"


def test_request_level_claim_rejects_unrelated_request_type():
    query = Query("What is the scholarship process?", request_type="scholarship")
    claim = next(item for item in extract_claims(query) if item.kind == "request")
    evidence = EvidenceUnit("e1", "The fee structure lists tuition and hostel charges.", "fees.docx", 0)
    assert match_claim_to_evidence(claim, evidence).status == "unsupported"


def test_request_level_claim_does_not_require_full_question_word_overlap():
    query = Query("What are the M.Tech eligibility requirements?", request_type="eligibility")
    claim = next(item for item in extract_claims(query) if item.kind == "request")
    assert "what are the m.tech eligibility requirements?" not in claim.anchors
    evidence = EvidenceUnit(
        "e1",
        "The applicant must have a bachelor’s degree in engineering or science.",
        "admissions.docx",
        0,
    )
    assert match_claim_to_evidence(claim, evidence).status == "supported"


if __name__ == "__main__":
    tests = [
        obj for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
    print(f"E5 CLAIM HARD TESTS: PASS ({len(tests)} tests)")