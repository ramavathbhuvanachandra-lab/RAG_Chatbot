from __future__ import annotations

from types import SimpleNamespace

from ai_platform.core.retrieval.contracts import CandidateAlignment, DocumentMeaning, EvidenceQuality, RetrievalCandidate
from ai_platform.core.retrieval.evidence_boundary import build_evidence_boundary
from ai_platform.core.retrieval.verification import verify_candidate
from ai_platform.core.query.models import QueryFacet, RetrievalRequirement, SemanticQueryFrame, Target


def _candidate(text: str, *, programs=(), attributes=(), conflicts=()):
    doc = SimpleNamespace(page_content=text, metadata={"source": "data/test/source.docx"})
    return RetrievalCandidate.from_document(
        doc,
        source="data/test/source.docx",
        meaning=DocumentMeaning(programs=tuple(programs), attributes=tuple(attributes)),
        alignment=CandidateAlignment(semantic_match=0.95, attribute_match=0.95, scope_match=1.0, conflicts=tuple(conflicts)),
        quality=EvidenceQuality(),
    )


def _frame(question: str, target: str, attribute: str):
    return SemanticQueryFrame(
        original_query=question,
        normalized_query=question,
        semantic_query=f"{target} {attribute}",
        target=target,
        request_type="eligibility" if "eligib" in attribute.casefold() else "cost",
        facets=(QueryFacet("requested_attribute", attribute),),
        entities=(Target(text=target, confidence=1.0, resolution_state="resolved"),),
        requirement=RetrievalRequirement(mode="exact", require_target_alignment=True, require_attribute_alignment=True, reject_explicit_conflict=True),
    )


def test_semantic_similarity_cannot_replace_target_attribute_relation():
    frame = _frame("What are M.Tech eligibility requirements?", "M.Tech", "eligibility")
    candidate = _candidate("Eligibility requirements are described here.", attributes=("eligibility",))
    decision = verify_candidate(frame, candidate)
    assert decision.status != "verified"
    assert "required_target_not_grounded" in decision.reasons or "required_target_attribute_relation_not_grounded" in decision.reasons


def test_explicit_conflict_is_rejected():
    frame = _frame("What is the M.Tech application fee?", "M.Tech", "application fee")
    candidate = _candidate("M.Tech application fee information.", programs=("mtech",), attributes=("application fee",), conflicts=("program_scope",))
    decision = verify_candidate(frame, candidate)
    assert decision.status == "rejected"
    assert decision.conflict_detected is True


def test_same_document_local_relation_can_verify():
    frame = _frame("What is the M.Tech application fee?", "M.Tech", "application fee")
    candidate = _candidate("The M.Tech application fee is listed in the admission notice.", programs=("mtech",), attributes=("application fee",))
    decision = verify_candidate(frame, candidate)
    assert decision.status == "verified"


def test_evidence_boundary_accepts_only_ranked_verified_anchors():
    anchor = _candidate("M.Tech application fee is listed here.", programs=("mtech",), attributes=("application fee",))
    stray = SimpleNamespace(page_content="Unowned unrelated context.", metadata={"source": "data/test/stray.docx"})

    def groups(docs):
        return [SimpleNamespace(anchor=docs[0], documents=(docs[0], stray))]

    def expand(gs, chunks):
        return gs

    def flatten(gs, **kwargs):
        return tuple(doc for g in gs for doc in g.documents)

    boundary = build_evidence_boundary(
        ranked_candidates=(anchor,),
        canonical_chunks=(),
        build_evidence_groups=groups,
        expand_group_context=expand,
        flatten_evidence_groups=flatten,
        max_documents=40,
    )
    ids = {c.document_id for c in boundary.evidence_candidates}
    assert anchor.document_id in ids
    assert any("unrelated" in str(doc.page_content).casefold() for doc in boundary.evidence_documents)
    assert boundary.diagnostics["ownership_policy"] == "verified_anchor_only"
