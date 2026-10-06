from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from ai_platform.core.retrieval.contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
)
from ai_platform.core.retrieval.evidence_boundary import build_evidence_boundary


@dataclass
class Group:
    anchor: object
    documents: list[object]


def doc(text: str, source: str = "data/a.docx"):
    return SimpleNamespace(page_content=text, metadata={"source": source})


def cand(text: str, source: str = "data/a.docx", score: float = 0.8):
    d = doc(text, source)
    return RetrievalCandidate.from_document(
        d,
        source=source,
        meaning=DocumentMeaning(programs=("target",), attributes=("topic",)),
        alignment=CandidateAlignment(semantic_match=.95, coverage=.95, scope_match=1.0),
        quality=EvidenceQuality(content_quality=1.0, structural_quality=1.0, noise=0.0),
        final_score=score,
    )


def builders(expanded_extra: bool = True):
    def build(anchors):
        return [Group(a, [a]) for a in anchors]

    def expand(groups, canonical):
        out = list(groups)
        if expanded_extra and canonical:
            # Attach one owned neighbor to first group, and one unowned doc that
            # the boundary must reject.
            out[0].documents.append(canonical[0])
            out.append(Group(doc("unowned", "data/unowned.docx"), [doc("unowned", "data/unowned.docx")]))
        return out

    def flatten(groups, max_documents=40):
        result=[]
        for g in groups:
            result.extend(g.documents)
        return result[:max_documents]

    return build, expand, flatten


def test_verified_anchor_survives_exactly():
    anchor = cand("Target topic is documented.")
    build, expand, flatten = builders(expanded_extra=False)
    result = build_evidence_boundary(
        ranked_candidates=[anchor], canonical_chunks=[],
        build_evidence_groups=build, expand_group_context=expand, flatten_evidence_groups=flatten,
    )
    assert result.evidence_candidates == (anchor,)


def test_context_inherits_anchor_contract():
    anchor = cand("Target heading.")
    neighbor = doc("minimum requirement is 60%", "data/a.docx")
    build, _, flatten = builders(expanded_extra=False)
    def expand(groups, canonical):
        groups[0].documents.append(neighbor)
        return groups
    result = build_evidence_boundary(
        ranked_candidates=[anchor], canonical_chunks=[neighbor],
        build_evidence_groups=build, expand_group_context=expand, flatten_evidence_groups=flatten,
    )
    assert len(result.evidence_candidates) == 2
    derived = result.evidence_candidates[1]
    assert derived.meaning == anchor.meaning
    assert derived.alignment == anchor.alignment
    assert derived.provenance == anchor.provenance


def test_unowned_expansion_fails_closed():
    anchor = cand("Target heading.")
    build, expand, flatten = builders(expanded_extra=True)
    result = build_evidence_boundary(
        ranked_candidates=[anchor], canonical_chunks=[doc("owned neighbor", "data/a.docx")],
        build_evidence_groups=build, expand_group_context=expand, flatten_evidence_groups=flatten,
    )
    texts = [d.page_content for d in result.evidence_documents]
    assert "unowned" not in texts
    assert result.diagnostics["dropped_unowned_context_documents"] >= 1


def test_empty_ranked_is_empty():
    build, expand, flatten = builders()
    result = build_evidence_boundary(
        ranked_candidates=[], canonical_chunks=[],
        build_evidence_groups=build, expand_group_context=expand, flatten_evidence_groups=flatten,
    )
    assert result.evidence_candidates == ()


def test_scope_is_not_called_at_boundary():
    anchor = cand("Target topic.")
    build, expand, flatten = builders(expanded_extra=False)
    result = build_evidence_boundary(
        ranked_candidates=[anchor], canonical_chunks=[],
        build_evidence_groups=build, expand_group_context=expand, flatten_evidence_groups=flatten,
    )
    assert result.diagnostics["scope_checks_at_evidence_boundary"] == 0
