from types import SimpleNamespace

from ai_platform.core.evidence.packaging import EvidencePackage, PackagedEvidenceItem
from ai_platform.core.graph.nodes import CoreNodes, NodeDependencies


def _deps(*, generator):
    def boundary(**kwargs):
        ranked = tuple(kwargs["ranked_candidates"])
        return SimpleNamespace(
            to_state=lambda: {
                "evidence_candidates": ranked,
                "evidence_documents": tuple(getattr(x, "document", x) for x in ranked),
                "evidence_groups": (),
                "evidence_boundary_trace": {"input_ranked_candidates": len(ranked)},
                "rejected_evidence_anchor_ids": (),
            }
        )

    def guard(answer, fallback):
        if answer:
            return SimpleNamespace(
                status="accepted",
                answer=answer,
                fallback_used=False,
                reason="accepted",
            )
        return SimpleNamespace(
            status="fallback",
            answer=fallback,
            fallback_used=True,
            reason="insufficient_evidence",
        )

    def grounding(**kwargs):
        return SimpleNamespace(status="grounded", score=1.0)

    return NodeDependencies(
        conversation_resolver=lambda **_: {},
        multi_intent_decomposer=lambda _: (),
        query_understander=lambda **_: None,
        query_planner=lambda **_: None,
        dense_retrieve=lambda _: (),
        keyword_retrieve=lambda _: (),
        fuse_ranked_lists=lambda **_: (),
        align_query_to_document=lambda *a, **k: (None, None, None),
        rank_candidates=lambda *a, **k: (),
        verify_candidates=lambda *a, **k: SimpleNamespace(verified=(), uncertain=(), rejected=(), decisions=()),
        build_evidence_groups=lambda docs: (),
        expand_group_context=lambda groups, chunks: groups,
        flatten_evidence_groups=lambda groups, **_: (),
        build_evidence_boundary=boundary,
        filter_scope_conflicts=lambda *a, **k: (),
        assess_evidence=lambda *a, **k: SimpleNamespace(status="supported", score=1.0, relevant_documents=1),
        assess_coverage=lambda *a, **k: SimpleNamespace(status="supported", question_type="descriptive"),
        extract_claims=lambda _: (),
        extract_evidence_units=lambda _: (),
        audit_claims=lambda *a, **k: None,
        build_evidence_package=lambda *a, **k: EvidencePackage(status="empty", context=""),
        answer_generator=generator,
        answer_request_type=lambda **kwargs: kwargs,
        assess_answer_grounding=grounding,
        guard_answer=guard,
        institution_provider=lambda _: SimpleNamespace(name="IIT Jodhpur"),
        answer_model_provider=lambda _: object(),
        fallback_provider=lambda *_: "I don't know based on the available information.",
        canonical_chunks_provider=lambda _: (),
        scope_policy_provider=lambda _: None,
    )


def _ready_package():
    item = PackagedEvidenceItem(evidence_id="e1", text="The fee is Rs. 300.", source="test")
    return EvidencePackage(status="ready", context="Evidence 1\nThe fee is Rs. 300.", items=(item,))


def test_phase1_zero_boundary_blocks_generation():
    calls = {"generator": 0}

    def generator(*args, **kwargs):
        calls["generator"] += 1
        return SimpleNamespace(answer="SHOULD NOT RUN", model_calls=1, answer_mode="model")

    nodes = CoreNodes(_deps(generator=generator))
    state = {
        "question": "What is the fee?",
        "evidence_candidates": (),
        "evidence_package": _ready_package(),
    }

    result = nodes.answer_node(state)

    assert calls["generator"] == 0
    assert result["model_calls"] == 0
    assert result["answer_generation"] is None


def test_phase1_evidence_package_is_only_answer_contract():
    calls = {"generator": 0}

    def generator(request, model):
        calls["generator"] += 1
        return SimpleNamespace(answer="The fee is Rs. 300.", model_calls=1, answer_mode="model")

    nodes = CoreNodes(_deps(generator=generator))
    state = {
        "question": "What is the fee?",
        "evidence_candidates": (object(),),
        "ranked_candidates": (object(),),
        "final_context_package": _ready_package(),
        "evidence_package": _ready_package(),
    }

    result = nodes.answer_node(state)

    assert calls["generator"] == 1
    assert result["answer"] == "The fee is Rs. 300."
    assert result["context"] == "Evidence 1\nThe fee is Rs. 300."


def test_phase1_boundary_state_is_explicit():
    nodes = CoreNodes(_deps(generator=lambda *a, **k: None))
    anchor = object()
    result = nodes.evidence_boundary_node({"ranked_candidates": (anchor,)})

    assert result["evidence_candidates"] == (anchor,)
    assert result["evidence_boundary_trace"]["input_ranked_candidates"] == 1


def test_phase1_context_preserves_evidence_layout():
    nodes = CoreNodes(_deps(generator=lambda *a, **k: SimpleNamespace(
        answer="The fee is Rs. 300.", model_calls=1, answer_mode="model"
    )))
    package = _ready_package()
    result = nodes.answer_node({
        "question": "What is the fee?",
        "evidence_candidates": (object(),),
        "evidence_package": package,
    })
    assert result["context"] == package.context
    assert "\n" in result["context"]
