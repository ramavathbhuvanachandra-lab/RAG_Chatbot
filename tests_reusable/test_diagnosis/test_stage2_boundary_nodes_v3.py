"""Graph-node integration checks for the Stage-2 candidate/evidence boundary."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from ai_platform.core.graph.nodes import CoreNodes, NodeDependencies
from ai_platform.core.retrieval.contracts import RetrievalCandidate


@dataclass(frozen=True)
class Result:
    status: str = "supported"
    score: float = 0.9
    relevant_documents: int = 2
    question_type: str = "eligibility"
    ready_for_generation: bool = True
    context: str = "evidence"


def document(text: str, source: str = "source.txt"):
    return SimpleNamespace(page_content=text, metadata={"source": source})


def candidate(text: str, key: str):
    return RetrievalCandidate.from_document(
        document(text),
        source="source.txt",
        document_id=key,
        final_score=0.8,
    )


def deps(calls: dict[str, object]) -> NodeDependencies:
    def scope_filter(*_args, **_kwargs):
        calls["scope_filter"] = int(calls.get("scope_filter", 0)) + 1
        raise AssertionError("scope must not run again at the evidence boundary")

    anchor = document("Target eligibility anchor.")
    context = document("The detailed threshold is 60 percent.")

    def groups(anchors):
        return [SimpleNamespace(anchor=anchors[0], documents=[anchors[0]])]

    def expand(gs, _chunks):
        gs[0].documents.append(context)
        return gs

    def flatten(gs, max_documents=40):
        return [doc for g in gs for doc in g.documents][:max_documents]

    def assess(candidates, **_kwargs):
        calls["assessed"] = tuple(candidates)
        return Result()

    def coverage(_query, candidates):
        calls["covered"] = tuple(candidates)
        return Result()

    def extract_claims(_query):
        return ("claim",)

    def extract_units(candidates):
        calls["unit_input"] = tuple(candidates)
        return tuple(candidates)

    def audit(claims, units):
        calls["audit_input"] = (tuple(claims), tuple(units))
        return Result()

    def package(units, **_kwargs):
        calls["package_input"] = tuple(units)
        return Result()

    return NodeDependencies(
        conversation_resolver=lambda **_k: {},
        multi_intent_decomposer=lambda _q: (),
        query_understander=lambda _q: SimpleNamespace(original_query=_q),
        dense_retrieve=lambda _q: (),
        keyword_retrieve=lambda _q: (),
        fuse_ranked_lists=lambda *_a, **_k: (),
        align_query_to_document=lambda *_a, **_k: None,
        rank_candidates=lambda *_a, **_k: (),
        verify_candidates=lambda *_a, **_k: None,
        build_evidence_groups=groups,
        expand_group_context=expand,
        flatten_evidence_groups=flatten,
        filter_scope_conflicts=scope_filter,
        assess_evidence=assess,
        assess_coverage=coverage,
        extract_claims=extract_claims,
        extract_evidence_units=extract_units,
        audit_claims=audit,
        build_evidence_package=package,
        answer_generator=lambda *_a, **_k: None,
        answer_request_type=object,
        assess_answer_grounding=lambda *_a, **_k: None,
        guard_answer=lambda *_a, **_k: None,
        institution_provider=lambda _s: None,
        answer_model_provider=lambda _s: None,
        fallback_provider=lambda _s, _i: "",
        canonical_chunks_provider=lambda _s: (anchor, context),
    )


def test_evidence_context_preserves_verified_anchor_and_owned_context():
    calls = {}
    node = CoreNodes(deps(calls))
    anchor = candidate("Target eligibility anchor.", "anchor-1")
    state = {
        "query": SimpleNamespace(original_query="What is the eligibility?", request_type="eligibility"),
        "ranked_candidates": (anchor,),
    }

    out = node.evidence_context_node(state)

    assert out["evidence_candidates"][0] is anchor
    assert len(out["evidence_candidates"]) == 2
    assert out["evidence_candidates"][1].provenance == anchor.provenance
    assert calls.get("scope_filter", 0) == 0
    assert out["evidence_boundary_trace"]["ownership_policy"] == "verified_anchor_only"


def test_downstream_assessment_uses_evidence_candidates_not_ranked_candidates():
    calls = {}
    node = CoreNodes(deps(calls))
    anchor = candidate("Target eligibility anchor.", "anchor-2")
    context = RetrievalCandidate.from_document(
        document("The detailed threshold is 60 percent."),
        source="source.txt",
        document_id="context-2",
    )
    state = {
        "query": SimpleNamespace(original_query="What is the eligibility?", request_type="eligibility"),
        "ranked_candidates": (anchor,),
        "evidence_candidates": (anchor, context),
    }

    node.assess_evidence_node(state)
    node.assess_coverage_node(state)
    node.audit_claims_node(state)

    assert calls["assessed"] == (anchor, context)
    assert calls["covered"] == (anchor, context)
    assert calls["unit_input"] == (anchor, context)
    assert calls["audit_input"][1] == (anchor, context)


def test_packaging_fails_closed_when_evidence_boundary_is_empty():
    calls = {}
    node = CoreNodes(deps(calls))
    state = {
        "evidence_candidates": (),
        "evidence_units": ("stale-unit",),
        "claim_audit": Result(),
    }

    out = node.package_evidence_node(state)

    assert calls["package_input"] == ()
    assert out["ready_for_generation"] is False
