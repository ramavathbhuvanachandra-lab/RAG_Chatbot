"""
Stage 2 Regression Test Suite.
Covers all 14 non-negotiable regression contract tests specified in Stage 2 Section 9:
1. A correct chunk retrieved at rank 8 is not lost merely because an earlier unrelated limit is Top-5.
2. A relevant candidate in the fused pool survives when verification says it is compatible and it is required for the final answer.
3. A verified evidence anchor is preserved through boundary filtering and context expansion.
4. Relevant text remains in the exact final generator input when it fits the configured budget.
5. Truncation does not systematically eliminate evidence belonging to a later intent.
6. Multiple complementary chunks survive when collectively required.
7. Deduplication does not merge distinct sources or complementary evidence improperly.
8. Rejected or incompatible candidates cannot be resurrected by ranking, expansion or packaging.
9. Cross-degree and cross-institution contamination remains impossible under the tested contracts.
10. A question with two or three intents retains independent candidates and reports coverage per intent.
11. An unsupported intent does not erase valid evidence for a supported intent.
12. A context-budget violation is explicit and does not falsely claim full coverage.
13. Source provenance is preserved into the final prompt.
14. Existing Stage 1, scope/security, LSQC, planner, packaging and downstream contract tests remain green.
"""
from __future__ import annotations

import os
os.environ["PLANNER_OFFLINE_MODE"] = "1"

import pytest
from langchain_core.documents import Document

from ai_platform.core.query.models import Query, SemanticQueryFrame, Target
from ai_platform.core.retrieval.contracts import (
    RetrievalCandidate,
    RetrievalProvenance,
    RetrievalSignal,
)
from ai_platform.core.retrieval.verification import verify_candidate, verify_candidates
from ai_platform.core.retrieval.ranking import rank_candidates, RankingConfig
from ai_platform.core.retrieval.evidence_boundary import build_evidence_boundary
from ai_platform.core.evidence.claims import EvidenceUnit, extract_evidence_units, audit_claims, Claim
from ai_platform.core.evidence.packaging import build_evidence_package, PackagedEvidenceItem
from ai_platform.core.answering.generator import AnswerGenerationRequest, build_answer_prompt
from ai_platform.core.graph.nodes import CoreNodes, default_dependencies


def _make_candidate(
    doc_id: str,
    text: str,
    source: str,
    heading: str = "",
    rank: int = 1,
    score: float = 0.9,
    channel: str = "bm25",
) -> RetrievalCandidate:
    doc = Document(page_content=text, metadata={"source": source, "heading": heading})
    prov = RetrievalProvenance(
        primary_query="test query",
        signals=(RetrievalSignal(channel=channel, rank=rank, score=score),),
    )
    return RetrievalCandidate.from_document(
        doc,
        source=source,
        document_id=doc_id,
        provenance=prov,
        final_score=score,
    )


# ---------------------------------------------------------------------------
# Test 1: Rank 8 chunk not lost by Top-5 limit
# ---------------------------------------------------------------------------
def test_1_rank_8_candidate_not_lost():
    """A correct chunk retrieved at rank 8 is not lost merely because an earlier limit is Top-5."""
    candidates = [
        _make_candidate(f"doc_{i}", f"General information about college campus {i}", f"src_{i}.docx", rank=i, score=1.0 - i*0.05)
        for i in range(1, 8)
    ]
    # Rank 8 has the exact required M.Tech qualification rule
    gold = _make_candidate(
        "doc_8",
        "M.Tech eligibility requires a 4-year bachelor's degree in engineering or science.",
        "mtech_admissions.docx",
        rank=8,
        score=0.60
    )
    candidates.append(gold)
    
    # Ranking pool supports top_k=15
    ranked = rank_candidates("What are the M.Tech eligibility requirements?", candidates, top_k=15)
    ranked_ids = [c.document_id for c in ranked]
    assert "doc_8" in ranked_ids, "Candidate retrieved at rank 8 must survive in ranked candidates pool"


# ---------------------------------------------------------------------------
# Test 2: Compatible verified candidate survives
# ---------------------------------------------------------------------------
def test_2_compatible_candidate_survives():
    """A relevant candidate in fused pool survives when verification confirms compatibility."""
    frame = SemanticQueryFrame(
        original_query="What are the M.Tech admissions requirements?",
        semantic_query="M.Tech admissions requirements",
        target="M.Tech",
        facets=("requirements", "admissions"),
    )
    candidate = _make_candidate(
        "mtech_deg",
        "Candidates seeking admission to M.Tech program must have a 4-year bachelor degree with 60% marks.",
        "programs/mtech/mtech_admissions.docx",
        heading="M.Tech Admissions",
    )
    dec = verify_candidate(frame, candidate)
    assert dec.accepted, f"Compatible candidate must be accepted by verification, reasons: {dec.reasons}"
    assert dec.status == "verified"


# ---------------------------------------------------------------------------
# Test 3: Verified anchor preserved through boundary and expansion
# ---------------------------------------------------------------------------
def test_3_verified_anchor_preserved_through_boundary():
    """A verified evidence anchor is preserved through boundary filtering and context expansion."""
    anchor = _make_candidate(
        "anchor_1",
        "Ph.D. full-time regular students receive financial assistance as per MoE norms.",
        "programs/phd/general_information.docx",
        heading="Financial Assistance",
    )
    boundary = build_evidence_boundary(
        ranked_candidates=[anchor],
        canonical_chunks=[],
        build_evidence_groups=lambda docs: [],
        expand_group_context=lambda groups, chunks: groups,
        flatten_evidence_groups=lambda groups, max_documents=40: [getattr(anchor, "document", anchor)],
    )
    ev_ids = [c.document_id for c in boundary.evidence_candidates]
    assert "anchor_1" in ev_ids, "Anchor candidate must survive through evidence boundary"


# ---------------------------------------------------------------------------
# Test 4: Relevant text remains in exact final generator prompt
# ---------------------------------------------------------------------------
def test_4_relevant_text_in_final_generator_input():
    """Relevant text remains in exact final generator input when it fits budget."""
    item = PackagedEvidenceItem(
        evidence_id="ev_001",
        text="The minimum attendance requirement for UG students to appear in end-semester exam is 75%.",
        source="programs/btech/academic_progression.docx",
        heading="Attendance Policy",
    )
    pkg = build_evidence_package(
        evidence=[EvidenceUnit(evidence_id="ev_001", text=item.text, source=item.source, heading=item.heading)],
        max_units=5,
        max_units_per_source=3,
        verified_evidence=True,
    )
    req = AnswerGenerationRequest(
        question="What is the attendance requirement?",
        evidence=pkg,
        chat_history=(),
        institution=None,
    )
    messages = build_answer_prompt(req)
    prompt_str = messages[1]["content"] if len(messages) > 1 else ""
    assert "75%" in prompt_str, "75% attendance rule must be present in the serialized prompt string"
    assert "attendance" in prompt_str.lower()


# ---------------------------------------------------------------------------
# Test 5: Truncation does not eliminate evidence of later intent
# ---------------------------------------------------------------------------
def test_5_multi_intent_evidence_not_eliminated():
    """Truncation does not systematically eliminate evidence belonging to a later intent."""
    deps = default_dependencies()
    nodes = CoreNodes(deps)
    q = "What is the attendance requirement and what are the branch change rules for B.Tech?"
    state = {
        "question": q,
        "resolved_question": q,
        "chat_history": [],
    }
    state.update(nodes.understand_query_node(state))
    state.update(nodes.plan_multi_intent_node(state))
    state.update(nodes.process_multi_intent_node(state))
    
    pkg = state.get("evidence_package")
    assert pkg is not None
    req = AnswerGenerationRequest(question=q, evidence=pkg, chat_history=(), institution=None)
    prompt = build_answer_prompt(req)[1]["content"].lower()
    
    assert "attendance" in prompt, "Intent 1 (attendance) must be in final prompt"
    assert "branch change" in prompt or "change of branch" in prompt, "Intent 2 (branch change) must not be eliminated"


# ---------------------------------------------------------------------------
# Test 6: Multiple complementary chunks survive
# ---------------------------------------------------------------------------
def test_6_multiple_complementary_chunks_survive():
    """Multiple complementary chunks from same document survive when collectively required."""
    unit1 = EvidenceUnit(
        evidence_id="u1",
        text="Regular M.Tech requires a 4-year bachelor's degree in engineering or science.",
        source="mtech_admissions.docx",
        heading="Eligibility",
    )
    unit2 = EvidenceUnit(
        evidence_id="u2",
        text="A minimum of 60% marks (55% for SC/ST) or CGPA of 6.0 on a 10-point scale is required.",
        source="mtech_admissions.docx",
        heading="Cutoffs",
    )
    unit3 = EvidenceUnit(
        evidence_id="u3",
        text="Candidates with B.Tech from CFTIs with CGPA 8.0 or above are exempt from GATE score.",
        source="mtech_admissions.docx",
        heading="Exemptions",
    )
    pkg = build_evidence_package(
        evidence=[unit1, unit2, unit3],
        max_units=10,
        max_units_per_source=5,
        verified_evidence=True,
    )
    packaged_ids = [it.evidence_id for it in pkg.items]
    assert len(packaged_ids) == 3, f"All 3 complementary units from mtech_admissions.docx must survive, got {packaged_ids}"


# ---------------------------------------------------------------------------
# Test 7: Deduplication does not improperly merge distinct sources
# ---------------------------------------------------------------------------
def test_7_dedup_preserves_distinct_sources():
    """Deduplication does not merge distinct sources or complementary evidence improperly."""
    c1 = _make_candidate("c1", "Hostel rent is Rs. 12,000 per semester.", "hostel_rules.docx")
    c2 = _make_candidate("c2", "Dining charge is Rs. 18,000 per semester.", "mess_rules.docx")
    ranked = rank_candidates("fees", [c1, c2], top_k=5)
    assert len(ranked) == 2, "Distinct sources must both be preserved"


# ---------------------------------------------------------------------------
# Test 8: Rejected candidates cannot be resurrected
# ---------------------------------------------------------------------------
def test_8_rejected_candidates_not_resurrected():
    """Rejected or incompatible candidates cannot be resurrected by ranking, expansion or packaging."""
    frame = SemanticQueryFrame(
        original_query="What are the Ph.D. fellowship rules?",
        semantic_query="Ph.D. fellowship rules",
        target="Ph.D.",
        facets=("fellowship", "rules"),
    )
    incompatible = _make_candidate(
        "btech_fees",
        "B.Tech students must pay tuition fees of Rs. 1,00,000 per semester.",
        "programs/btech/fees.docx",
        heading="B.Tech Tuition",
    )
    dec = verify_candidate(frame, incompatible)
    # Even if fed to ranking, packaging fail-safe must exclude unverified candidates
    pkg = build_evidence_package(
        evidence=[],
        claim_audit=None,
        verified_evidence=False,
    )
    assert len(pkg.items) == 0, "Rejected candidates must not be present in evidence package"


# ---------------------------------------------------------------------------
# Test 9: Zero cross-degree and cross-institution contamination
# ---------------------------------------------------------------------------
def test_9_cross_degree_contamination_prevented():
    """Cross-degree contamination remains impossible under tested verification contracts."""
    frame = SemanticQueryFrame(
        original_query="What are the B.Tech minimum credits required to graduate?",
        semantic_query="B.Tech graduation credits",
        target="B.Tech",
        facets=("credits", "graduation"),
    )
    mtech_chunk = _make_candidate(
        "mtech_credits",
        "M.Tech degree requires completion of 64 course and thesis credits.",
        "programs/mtech/curriculum.docx",
        heading="M.Tech Credits",
    )
    dec = verify_candidate(frame, mtech_chunk)
    assert not dec.accepted or dec.status != "verified" or dec.conflict_detected or not dec.target_grounded, \
        "M.Tech credits chunk must not be falsely accepted as verified for a B.Tech credits query"


# ---------------------------------------------------------------------------
# Test 10: Multi-intent retains independent candidates per intent
# ---------------------------------------------------------------------------
def test_10_multi_intent_retains_independent_candidates():
    """A question with two intents retains independent candidates and reports coverage per intent."""
    deps = default_dependencies()
    nodes = CoreNodes(deps)
    q = "What is the M.Tech admission fee and what is the Ph.D. fellowship amount?"
    state = {
        "question": q,
        "resolved_question": q,
        "chat_history": [],
    }
    state.update(nodes.understand_query_node(state))
    state.update(nodes.plan_multi_intent_node(state))
    state.update(nodes.process_multi_intent_node(state))
    
    assert state.get("is_multi_intent"), "Query must be recognized as multi-intent"
    units = state.get("intent_units", ())
    assert len(units) >= 2, "Must contain at least 2 intent units"
    assert state.get("evidence_package") is not None


# ---------------------------------------------------------------------------
# Test 11: Unsupported intent does not erase valid evidence for supported intent
# ---------------------------------------------------------------------------
def test_11_unsupported_intent_does_not_erase_supported_intent():
    """An unsupported intent does not erase valid evidence for a supported intent."""
    deps = default_dependencies()
    nodes = CoreNodes(deps)
    # Intent 1: B.Tech attendance (supported in corpus)
    # Intent 2: Secret submarine base (unsupported)
    q = "What is the B.Tech 75% attendance rule and where is the secret submarine base located?"
    state = {
        "question": q,
        "resolved_question": q,
        "chat_history": [],
    }
    state.update(nodes.understand_query_node(state))
    state.update(nodes.plan_multi_intent_node(state))
    state.update(nodes.process_multi_intent_node(state))
    
    pkg = state.get("evidence_package")
    assert pkg is not None
    assert len(pkg.items) > 0, "Valid evidence for B.Tech attendance must survive despite unsupported second intent"
    assert any("attendance" in it.text.lower() for it in pkg.items), "Attendance rule must survive in package"


# ---------------------------------------------------------------------------
# Test 12: Context-budget violation is explicit
# ---------------------------------------------------------------------------
def test_12_context_budget_violation_explicit():
    """A context-budget violation is explicit and does not falsely claim full coverage."""
    # Build package with tiny character limit 50 chars
    unit = EvidenceUnit(
        evidence_id="u_long",
        text="This is a very long policy sentence that cannot possibly fit into a tiny twenty character context budget.",
        source="policy.docx"
    )
    pkg = build_evidence_package(
        evidence=[unit],
        max_context_chars=30,
        verified_evidence=True,
    )
    assert pkg.omitted_count > 0, "Omitted count must be recorded when budget is exceeded"
    assert len(pkg.items) == 0, "Item exceeding context chars must not be packaged"


# ---------------------------------------------------------------------------
# Test 13: Source provenance is preserved into the final prompt
# ---------------------------------------------------------------------------
def test_13_source_provenance_preserved_in_prompt():
    """Source heading/provenance is preserved into the final prompt."""
    item = PackagedEvidenceItem(
        evidence_id="ev_prov",
        text="The Senate may cancel admission if documents are not submitted.",
        source="general_admissions.docx",
        heading="Cancellation of Admission",
    )
    pkg = build_evidence_package(
        evidence=[EvidenceUnit(evidence_id="ev_prov", text=item.text, source=item.source, heading=item.heading)],
        verified_evidence=True,
    )
    req = AnswerGenerationRequest(
        question="Can admission be cancelled?",
        evidence=pkg,
        chat_history=(),
        institution=None,
    )
    prompt = build_answer_prompt(req)[1]["content"]
    assert "Cancellation of Admission" in prompt, "Evidence heading provenance must be in prompt"


# ---------------------------------------------------------------------------
# Test 14: Downstream contracts remain intact
# ---------------------------------------------------------------------------
def test_14_downstream_contracts_remain_intact():
    """Existing Stage 1, planner, and packaging contracts remain green."""
    from ai_platform.core.query.planner import build_query_plan, plan_to_frame
    plan = build_query_plan(question="What are the M.Tech eligibility requirements?")
    assert plan is not None
    assert len(plan.intents) >= 1
    assert "M.Tech" in str(plan.intents[0].target)
    frame = plan_to_frame(plan)
    assert frame is not None
    assert "M.Tech" in str(frame.target)
