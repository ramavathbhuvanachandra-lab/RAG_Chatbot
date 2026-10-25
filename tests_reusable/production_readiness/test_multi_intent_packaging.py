"""Tests for Intent-Aware Evidence Packaging in the Reusable Core.

Verifies:
1. 2-intent, 3-intent, and 4+-intent budget allocation and no-starvation.
2. Total budget exhaustion protection (character and unit caps).
3. Duplicate evidence avoidance across intents (global deduplication).
4. Missing evidence handling and explicit unsupported-intent tracking.
5. Single-intent source diversity (preventing single-document context capture).
6. Safe abstention on false-premise negative controls.
"""

from __future__ import annotations

import pytest

from ai_platform.core.evidence.claims import (
    Claim,
    ClaimAudit,
    ClaimEvidenceMatch,
    EvidenceUnit,
)
from ai_platform.core.evidence.packaging import (
    EvidencePackage,
    IntentEvidenceGroup,
    build_evidence_package,
    package_intent_aware_evidence,
)


def _make_unit(
    evidence_id: str,
    text: str,
    source: str = "source_1.docx",
    heading: str | None = None,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        text=text,
        source=source,
        position=0,
        heading=heading,
    )


def _make_audit(
    *,
    claim_ids: tuple[str, ...],
    matches: tuple[ClaimEvidenceMatch, ...],
    status: str = "supported",
    conflicts: tuple[str, ...] = (),
) -> ClaimAudit:
    claims = tuple(
        Claim(
            claim_id=cid,
            text=f"Claim text for {cid}",
            kind="request",
            name="req",
            required=True,
        )
        for cid in claim_ids
    )
    supported = tuple(claim_ids) if status == "supported" else ()
    partial = tuple(claim_ids) if status == "partial" else ()
    unsupported = tuple(claim_ids) if status == "insufficient" else ()
    return ClaimAudit(
        claims=claims,
        matches=matches,
        supported_claim_ids=supported,
        partial_claim_ids=partial,
        unsupported_claim_ids=unsupported,
        conflicting_claim_ids=conflicts,
    )


def test_two_intent_fair_share_no_starvation():
    """Verify Intent 1 (with 20 heavy units) does not starve Intent 2 (with 3 units)."""
    # Intent 1 has 20 units from one heavy source
    units_1 = tuple(
        _make_unit(f"i1_u{i}", f"Intent 1 heavy evidence text statement #{i} about progression rules.", source="progression.docx")
        for i in range(20)
    )
    # Intent 2 has 3 units from finance
    units_2 = tuple(
        _make_unit(f"i2_u{i}", f"Intent 2 finance refund statement #{i} refundable deposit is Rs 8000.", source="finance.docx")
        for i in range(3)
    )

    group_1 = IntentEvidenceGroup(
        intent_id=1,
        question="What are academic progression rules?",
        evidence_units=units_1,
        claim_audit=_make_audit(
            claim_ids=("c1",),
            matches=(ClaimEvidenceMatch("c1", "i1_u0", 0.9, "supported"),),
            status="supported",
        ),
    )
    group_2 = IntentEvidenceGroup(
        intent_id=2,
        question="What is the refundable deposit?",
        evidence_units=units_2,
        claim_audit=_make_audit(
            claim_ids=("c2",),
            matches=(ClaimEvidenceMatch("c2", "i2_u0", 0.9, "supported"),),
            status="supported",
        ),
    )

    package = package_intent_aware_evidence(
        [group_1, group_2],
        max_units=10,
        max_context_chars=4000,
    )

    assert package.status == "ready"
    assert len(package.items) > 0
    # Both sources must be represented in packaged items
    sources = {item.source for item in package.items}
    assert "progression.docx" in sources
    assert "finance.docx" in sources

    # Intent 2 must have survived into packaged items
    i2_items = [it for it in package.items if "Intent 2" in it.text]
    assert len(i2_items) >= 2, f"Intent 2 was starved: only got {len(i2_items)} items"


def test_three_and_four_intent_budget_allocation():
    """Verify 3-intent and 4-intent queries allocate fair share across all sub-intents."""
    for n_intents in (3, 4):
        groups = []
        for i in range(n_intents):
            units = tuple(
                _make_unit(f"i{i}_u{j}", f"Sub-intent {i} rule statement {j} details.", source=f"src_{i}.docx")
                for j in range(5)
            )
            groups.append(
                IntentEvidenceGroup(
                    intent_id=i + 1,
                    question=f"Question for sub-intent {i}?",
                    evidence_units=units,
                    claim_audit=_make_audit(
                        claim_ids=(f"c{i}",),
                        matches=(ClaimEvidenceMatch(f"c{i}", f"i{i}_u0", 0.9, "supported"),),
                        status="supported",
                    ),
                )
            )

        pkg = package_intent_aware_evidence(
            groups,
            max_units=12,
            max_context_chars=6000,
        )

        assert pkg.status == "ready"
        # Each intent must have at least one unit selected
        for i in range(n_intents):
            found = any(f"Sub-intent {i}" in item.text for item in pkg.items)
            assert found, f"Sub-intent {i} had no items selected in {n_intents}-intent query"


def test_total_budget_exhaustion_protection():
    """Verify total context budget and unit limits are strictly enforced."""
    groups = []
    for i in range(3):
        # 10 large units each
        units = tuple(
            _make_unit(f"i{i}_u{j}", f"Very long text paragraph {j} " * 20, source=f"src_{i}.docx")
            for j in range(10)
        )
        groups.append(
            IntentEvidenceGroup(
                intent_id=i + 1,
                question=f"Question {i}?",
                evidence_units=units,
            )
        )

    pkg = package_intent_aware_evidence(
        groups,
        max_units=6,
        max_context_chars=1500,
    )

    assert len(pkg.items) <= 6
    assert len(pkg.context) <= 1500
    assert pkg.omitted_count > 0


def test_global_deduplication_across_intents():
    """Verify identical text from different intents is not packaged twice."""
    shared_unit_1 = _make_unit("u_shared", "IIT Jodhpur offers merit-cum-means scholarships of Rs 1000 per month.", source="scholarship.docx")
    shared_unit_2 = _make_unit("u_shared_dup", "IIT Jodhpur offers merit-cum-means scholarships of Rs 1000 per month.", source="scholarship.docx")

    group_1 = IntentEvidenceGroup(
        intent_id=1,
        question="What is the MCM scholarship amount?",
        evidence_units=(shared_unit_1,),
    )
    group_2 = IntentEvidenceGroup(
        intent_id=2,
        question="What are financial assistance options?",
        evidence_units=(shared_unit_2,),
    )

    pkg = package_intent_aware_evidence(
        [group_1, group_2],
        max_units=10,
        max_context_chars=5000,
    )

    # Should only appear once in items
    matching_items = [it for it in pkg.items if "merit-cum-means" in it.text]
    assert len(matching_items) == 1, f"Expected 1 unique item, got {len(matching_items)}"


def test_missing_evidence_handling():
    """Verify that if one intent has no qualifying evidence, reasons explicitly note it."""
    group_1 = IntentEvidenceGroup(
        intent_id=1,
        question="What is the tuition fee?",
        evidence_units=(_make_unit("u1", "The tuition fee per semester is Rs 1,00,000.", source="fee.docx"),),
        claim_audit=_make_audit(
            claim_ids=("c1",),
            matches=(ClaimEvidenceMatch("c1", "u1", 0.9, "supported"),),
            status="supported",
        ),
    )
    group_2 = IntentEvidenceGroup(
        intent_id=2,
        question="What is the flight simulator syllabus?",
        evidence_units=(),  # No units found
    )

    pkg = package_intent_aware_evidence(
        [group_1, group_2],
        max_units=10,
        max_context_chars=5000,
    )

    assert pkg.status == "partial"
    assert any("flight simulator syllabus" in r for r in pkg.reasons)


def test_single_intent_two_pass_source_diversity():
    """Verify single-intent two-pass source diversity preserves secondary sources (challenge_05 fix)."""
    # 25 units from source A, 2 units from source B
    units = [
        _make_unit(f"sa_{i}", f"Source A detailed clause {i} regarding curriculum structure.", source="curriculum.docx")
        for i in range(25)
    ]
    units.extend([
        _make_unit(f"sb_{i}", f"Source B critical rule {i} regarding summer term backlog courses.", source="academic_progression.docx")
        for i in range(2)
    ])

    pkg = build_evidence_package(
        units,
        max_units=12,
        max_units_per_source=8,
        max_context_chars=5000,
        verified_evidence=True,
    )

    sources = {item.source for item in pkg.items}
    assert "curriculum.docx" in sources
    assert "academic_progression.docx" in sources, "Secondary source was starved by high-volume primary source"


def test_false_premise_control_abstention():
    """Verify false-premise questions with unsupported required constraints abstain safely (challenge_40 fix)."""
    # Query with a supported constraint (e.g. 2024) but an unsupported required constraint (e.g. cancel)
    claim_year = Claim(claim_id="c_year", text="year 2024", kind="constraint", name="year", required=True)
    claim_cancel = Claim(claim_id="c_cancel", text="cancel degrees", kind="constraint", name="action", required=True)

    unit = _make_unit("u1", "In 2024, the institute held its annual convocation.", source="convocation.docx")
    audit = ClaimAudit(
        claims=(claim_year, claim_cancel),
        matches=(ClaimEvidenceMatch("c_year", "u1", 0.9, "supported"),),
        supported_claim_ids=("c_year",),
        partial_claim_ids=(),
        unsupported_claim_ids=("c_cancel",),
        conflicting_claim_ids=(),
    )

    # ClaimAudit.status should be "insufficient" because required constraint "c_cancel" is unsupported
    assert audit.status == "insufficient"

    # Packaging with this audit must return status="empty" and ready_for_generation=False
    pkg = build_evidence_package(
        [unit],
        claim_audit=audit,
        verified_evidence=True,
    )

    assert pkg.status == "empty"
    assert pkg.ready_for_generation is False
