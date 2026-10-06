"""Candidate-preserving evidence boundary for the reusable RAG core.

This module sits after authoritative candidate verification/ranking and before
evidence assessment, claim auditing, and packaging. It never performs a second
query->document relevance/scope decision.

Principles
----------
* Ranked candidates are trusted anchors because the verifier already decided
  their relevance.
* Local expansion may add context only when it can be traced to an anchor.
* Context inherits the anchor's provenance/meaning/alignment/quality.
* Unowned expansion is discarded rather than becoming an implicit retrieval
  candidate.
* No institution-specific vocabulary or policy is embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from ai_platform.core.retrieval.contracts import RetrievalCandidate

Document = Any


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _document_text(document: Document) -> str:
    if isinstance(document, Mapping):
        return _clean(document.get("page_content"))
    return _clean(getattr(document, "page_content", ""))


def _document_source(document: Document) -> str:
    if isinstance(document, Mapping):
        metadata = document.get("metadata") or {}
    else:
        metadata = getattr(document, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return ""
    for key in ("source", "source_path", "path", "url"):
        value = _clean(metadata.get(key))
        if value:
            return value
    return ""


def document_key(document: Document) -> tuple[str, str]:
    return (
        _document_source(document).casefold(),
        _document_text(document).casefold(),
    )


def candidate_key(candidate: "RetrievalCandidate") -> str:
    value = _clean(getattr(candidate, "document_id", ""))
    if value:
        return value
    return "|".join(document_key(getattr(candidate, "document", candidate)))


def _derived_context_id(anchor_id: str, document: Document) -> str:
    raw = "\n".join((anchor_id, _document_source(document), _document_text(document)))
    return "context::" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _candidate_from_context(
    anchor: "RetrievalCandidate",
    document: Document,
) -> "RetrievalCandidate":
    # Runtime import keeps this boundary import-safe for graph startup.
    from ai_platform.core.retrieval.contracts import RetrievalCandidate

    return RetrievalCandidate.from_document(
        document,
        source=_document_source(document) or anchor.source,
        provenance=anchor.provenance,
        meaning=anchor.meaning,
        alignment=anchor.alignment,
        quality=anchor.quality,
        document_id=_derived_context_id(candidate_key(anchor), document),
        final_score=max(0.0, float(getattr(anchor, "final_score", 0.0) or 0.0)),
    )


def _dedupe_documents(documents: Sequence[Document]) -> tuple[Document, ...]:
    result: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        key = document_key(document)
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EvidenceBoundary:
    anchor_candidates: tuple["RetrievalCandidate", ...]
    evidence_candidates: tuple["RetrievalCandidate", ...]
    evidence_documents: tuple[Document, ...]
    evidence_groups: tuple[Any, ...]
    rejected_anchor_ids: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def to_state(self) -> dict[str, Any]:
        return {
            "evidence_groups": self.evidence_groups,
            "evidence_documents": self.evidence_documents,
            "evidence_candidates": self.evidence_candidates,
            "evidence_boundary_trace": self.diagnostics,
            "rejected_evidence_anchor_ids": self.rejected_anchor_ids,
        }


def _candidate_document_map(candidates: Sequence["RetrievalCandidate"]) -> dict[tuple[str, str], "RetrievalCandidate"]:
    mapping: dict[tuple[str, str], "RetrievalCandidate"] = {}
    for candidate in candidates:
        document = getattr(candidate, "document", candidate)
        mapping.setdefault(document_key(document), candidate)
    return mapping


def _group_owner_map(
    groups: Sequence[Any],
    anchors: Sequence["RetrievalCandidate"],
) -> dict[tuple[str, str], "RetrievalCandidate"]:
    anchor_by_key = {
        document_key(getattr(anchor, "document", anchor)): anchor
        for anchor in anchors
    }
    owners: dict[tuple[str, str], "RetrievalCandidate"] = {}
    for group in groups:
        group_anchor = getattr(group, "anchor", None)
        if group_anchor is None:
            continue
        owner = anchor_by_key.get(document_key(group_anchor))
        if owner is None:
            continue
        for document in (getattr(group, "documents", ()) or ()):
            owners.setdefault(document_key(document), owner)
    return owners


def build_evidence_boundary(
    *,
    ranked_candidates: Sequence["RetrievalCandidate"],
    canonical_chunks: Sequence[Document],
    build_evidence_groups: Callable[[Sequence[Document]], Sequence[Any]],
    expand_group_context: Callable[[Sequence[Any], Sequence[Document]], Sequence[Any]],
    flatten_evidence_groups: Callable[..., Sequence[Document]],
    max_documents: int = 40,
) -> EvidenceBoundary:
    ranked = tuple(ranked_candidates or ())
    if not ranked:
        return EvidenceBoundary(
            anchor_candidates=(),
            evidence_candidates=(),
            evidence_documents=(),
            evidence_groups=(),
            rejected_anchor_ids=(),
            diagnostics={
                "input_ranked_candidates": 0,
                "trusted_anchor_candidates": 0,
                "group_count": 0,
                "flattened_document_count": 0,
                "evidence_candidate_count": 0,
                "derived_context_candidate_count": 0,
                "dropped_unowned_context_documents": 0,
                "scope_checks_at_evidence_boundary": 0,
                "ownership_policy": "verified_anchor_only",
            },
        )

    anchors = ranked
    by_document = _candidate_document_map(anchors)
    anchor_docs = tuple(getattr(candidate, "document", candidate) for candidate in anchors)
    groups = list(build_evidence_groups(anchor_docs))
    owner_by_document = _group_owner_map(groups, anchors)

    if canonical_chunks:
        groups = list(expand_group_context(groups, tuple(canonical_chunks)))

    for key, owner in _group_owner_map(groups, anchors).items():
        owner_by_document.setdefault(key, owner)

    try:
        flattened = tuple(flatten_evidence_groups(groups, max_documents=max_documents))
    except TypeError:
        flattened = tuple(flatten_evidence_groups(groups))

    flattened = _dedupe_documents(flattened)

    evidence_candidates: list[RetrievalCandidate] = []
    evidence_documents: list[Document] = []
    derived_context_count = 0
    dropped_unowned_count = 0

    for document in flattened:
        existing = by_document.get(document_key(document))
        if existing is not None:
            evidence_candidates.append(existing)
            evidence_documents.append(document)
            continue

        owner = owner_by_document.get(document_key(document))
        if owner is None:
            dropped_unowned_count += 1
            continue

        evidence_candidates.append(_candidate_from_context(owner, document))
        evidence_documents.append(document)
        derived_context_count += 1

    return EvidenceBoundary(
        anchor_candidates=anchors,
        evidence_candidates=tuple(evidence_candidates),
        evidence_documents=tuple(evidence_documents),
        evidence_groups=tuple(groups),
        rejected_anchor_ids=(),
        diagnostics={
            "input_ranked_candidates": len(ranked),
            "trusted_anchor_candidates": len(anchors),
            "group_count": len(groups),
            "flattened_document_count": len(flattened),
            "evidence_candidate_count": len(evidence_candidates),
            "derived_context_candidate_count": derived_context_count,
            "dropped_unowned_context_documents": dropped_unowned_count,
            "scope_checks_at_evidence_boundary": 0,
            "ownership_policy": "verified_anchor_only",
        },
    )


__all__ = ["EvidenceBoundary", "build_evidence_boundary", "candidate_key", "document_key"]
