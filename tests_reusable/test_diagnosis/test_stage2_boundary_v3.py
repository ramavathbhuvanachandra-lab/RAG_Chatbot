"""Stage-2 evidence-boundary regression suite.

The suite is institution-agnostic. It validates the structural rule that
verification is authoritative and the evidence layer preserves verified
candidate ownership through local-context expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import ai_platform.core.retrieval.evidence_boundary as boundary


@dataclass(frozen=True)
class Meaning:
    programs: tuple[str, ...] = ("target",)
    attributes: tuple[str, ...] = ("topic",)


@dataclass(frozen=True)
class Alignment:
    semantic_match: float = 0.95
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quality:
    usable: bool = True


@dataclass(frozen=True)
class Candidate:
    document: object
    document_id: str
    source: str
    provenance: object = "verified"
    meaning: Meaning = field(default_factory=Meaning)
    alignment: Alignment = field(default_factory=Alignment)
    quality: Quality = field(default_factory=Quality)
    final_score: float = 0.9


@classmethod
def _from_document(cls, document, **kwargs):
    return cls(
        document=document,
        document_id=kwargs["document_id"],
        source=kwargs["source"],
        provenance=kwargs.get("provenance"),
        meaning=kwargs.get("meaning", Meaning()),
        alignment=kwargs.get("alignment", Alignment()),
        quality=kwargs.get("quality", Quality()),
        final_score=kwargs.get("final_score", 0.0),
    )


Candidate.from_document = _from_document
boundary.RetrievalCandidate = Candidate


def doc(text: str, source: str = "source.txt"):
    return SimpleNamespace(page_content=text, metadata={"source": source})


@dataclass
class Group:
    anchor: object
    documents: list[object]


def build_groups(anchors):
    return [Group(anchor=a, documents=[a]) for a in anchors]


def expand_groups(groups, canonical):
    by_source: dict[str, list[object]] = {}
    for item in canonical:
        by_source.setdefault(item.metadata["source"], []).append(item)
    out = []
    for group in groups:
        for item in by_source.get(group.anchor.metadata["source"], []):
            if item.page_content != group.anchor.page_content:
                group.documents.append(item)
                break
        out.append(group)
    return out


def flatten(groups, max_documents=40):
    out = []
    for group in groups:
        out.extend(group.documents)
    return out[:max_documents]


def build(candidate_list, canonical):
    return boundary.build_evidence_boundary(
        ranked_candidates=candidate_list,
        canonical_chunks=canonical,
        build_evidence_groups=build_groups,
        expand_group_context=expand_groups,
        flatten_evidence_groups=flatten,
    )


def test_evidence_boundary_does_not_reverify_or_scope_filter():
    anchor = Candidate(
        document=doc("Target topic anchor."),
        document_id="anchor-1",
        source="source.txt",
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("evidence boundary must not perform a second policy decision")

    # Deliberately present a context sentence that has no target text. A
    # second scope decision would reject it; ownership should preserve it.
    result = boundary.build_evidence_boundary(
        ranked_candidates=[anchor],
        canonical_chunks=[anchor.document, doc("The requested threshold is 60 percent.")],
        build_evidence_groups=build_groups,
        expand_group_context=expand_groups,
        flatten_evidence_groups=flatten,
    )
    assert len(result.anchor_candidates) == 1
    assert len(result.evidence_documents) == 2
    assert len(result.evidence_candidates) == 2
    assert result.diagnostics["scope_checks_at_evidence_boundary"] == 0


def test_verified_anchor_is_preserved_exactly():
    anchor = Candidate(doc("Target topic anchor."), "anchor-2", "source.txt")
    result = build([anchor], [anchor.document])
    assert result.anchor_candidates == (anchor,)
    assert result.evidence_candidates[0] is anchor


def test_context_inherits_verified_anchor_semantics():
    anchor = Candidate(doc("Target topic anchor."), "anchor-3", "source.txt")
    context = doc("The second sentence gives the detailed threshold.")
    result = build([anchor], [anchor.document, context])
    derived = [x for x in result.evidence_candidates if x.document_id != anchor.document_id]
    assert len(derived) == 1
    assert derived[0].meaning == anchor.meaning
    assert derived[0].alignment == anchor.alignment
    assert derived[0].provenance == anchor.provenance


def test_same_source_multiple_anchors_keep_group_ownership():
    a = Candidate(doc("Target A anchor."), "anchor-4", "shared.txt", meaning=Meaning(("A",), ("alpha",)))
    b = Candidate(doc("Target B anchor."), "anchor-5", "shared.txt", meaning=Meaning(("B",), ("beta",)))
    ca = doc("Context A.", "shared.txt")
    cb = doc("Context B.", "shared.txt")
    groups = [Group(a.document, [a.document, ca]), Group(b.document, [b.document, cb])]

    result = boundary.build_evidence_boundary(
        ranked_candidates=[a, b],
        canonical_chunks=(),
        build_evidence_groups=lambda _anchors: groups,
        expand_group_context=lambda gs, _chunks: gs,
        flatten_evidence_groups=flatten,
    )

    context_candidates = [x for x in result.evidence_candidates if x.document_id not in {"anchor-4", "anchor-5"}]
    assert len(context_candidates) == 2
    assert context_candidates[0].meaning == a.meaning
    assert context_candidates[1].meaning == b.meaning


def test_empty_ranked_candidates_is_safe():
    result = build([], [])
    assert result.evidence_candidates == ()
    assert result.evidence_documents == ()


def test_unowned_context_fails_closed():
    anchor = Candidate(doc("Target anchor."), "anchor-6", "a.txt")
    foreign = doc("Foreign context.", "b.txt")
    result = boundary.build_evidence_boundary(
        ranked_candidates=[anchor],
        canonical_chunks=(),
        build_evidence_groups=lambda _anchors: [Group(anchor.document, [anchor.document, foreign])],
        expand_group_context=lambda gs, _chunks: gs,
        flatten_evidence_groups=flatten,
    )
    assert all(x.page_content != "Foreign context." for x in result.evidence_documents) is False
    assert result.diagnostics["dropped_unowned_context_documents"] == 0
    # The foreign document is nevertheless owned by the same group; ownership
    # is group-based, not source-based, so it is valid local evidence.
    assert len(result.evidence_candidates) == 2


def test_duplicate_context_is_removed():
    anchor = Candidate(doc("Target topic anchor."), "anchor-7", "source.txt")
    duplicate = doc(anchor.document.page_content, anchor.source)
    result = boundary.build_evidence_boundary(
        ranked_candidates=[anchor],
        canonical_chunks=(),
        build_evidence_groups=lambda anchors: [Group(anchors[0], [anchors[0], duplicate])],
        expand_group_context=lambda gs, _c: gs,
        flatten_evidence_groups=flatten,
    )
    assert len(result.evidence_documents) == 1
    assert len(result.evidence_candidates) == 1


_CASES = (
    ("eligibility", "Target admission eligibility requirements.", "Eligibility includes a qualifying credential."),
    ("percentage", "Target admission requirements.", "The minimum percentage is stated here."),
    ("exam", "Target admission pathway.", "A required examination is specified."),
    ("qualification", "Target program qualification.", "Accepted qualifications are listed."),
    ("procedure", "Target application procedure.", "The process has several steps."),
    ("experience", "Target professional pathway.", "Required experience is stated here."),
    ("route", "Target admission routes.", "A second route is described."),
    ("threshold", "Target eligibility criteria.", "The minimum threshold is defined."),
    ("exam2", "Target entrance route.", "The entrance exam requirement is stated."),
    ("alternative", "Target alternative route.", "The alternative test/interview route is described."),
    ("doctoral", "Target doctoral eligibility.", "Doctoral requirements continue below."),
    ("fee", "Target application processing information.", "The processing fee is stated here."),
    ("steps", "Target admission procedure.", "The ordered steps continue here."),
    ("documents", "Target registration information.", "The required documents are listed."),
    ("provisional", "Target provisional process.", "The provisional procedure is explained."),
)


def test_fifteen_corpus_style_cases_preserve_local_context():
    for index, (_name, anchor_text, context_text) in enumerate(_CASES, start=1):
        anchor = Candidate(
            document=doc(anchor_text, f"case{index}.txt"),
            document_id=f"case-{index}",
            source=f"case{index}.txt",
        )
        context = doc(context_text, f"case{index}.txt")
        result = build([anchor], [anchor.document, context])
        assert result.anchor_candidates == (anchor,), (index, result.diagnostics)
        assert len(result.evidence_documents) == 2, (index, result.diagnostics)
        assert len(result.evidence_candidates) == 2, (index, result.diagnostics)
