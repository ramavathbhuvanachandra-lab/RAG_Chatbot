"""Reusable evidence sufficiency for the institutional RAG core.

Pipeline position
-----------------
    ranked retrieval candidates -> this module -> evidence / answering

Purpose
-------
Decide whether ranked retrieval candidates provide enough *grounded*
information to continue toward answer generation.

This module deliberately does not:
    * call an LLM;
    * retrieve documents;
    * parse institution-specific vocabulary;
    * generate answers;
    * infer facts from source filenames.

Instead it consumes the semantic/retrieval contracts already established by
upstream stages and makes a conservative evidence decision.

The old implementation contained substantial behavior around evidence
sufficiency, question-type handling, conflicts, numeric/temporal compatibility,
and conservative fallback.  This module starts the clean-core migration of
that behavior while keeping the public decision model small and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

from ai_platform.core.query.models import Query, RetrievalRequirement
from ai_platform.core.retrieval.contracts import RetrievalCandidate


EvidenceStatus = Literal[
    "supported",
    "partial",
    "insufficient",
    "conflicted",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_THRESHOLD = 0.68
PARTIAL_THRESHOLD = 0.42

EXACT_REQUIRED_ALIGNMENT = 0.72
FOCUSED_REQUIRED_ALIGNMENT = 0.52
STANDARD_REQUIRED_ALIGNMENT = 0.40

EXACT_MIN_COVERAGE = 0.55
FOCUSED_MIN_COVERAGE = 0.40
STANDARD_MIN_COVERAGE = 0.30

LIST_STRONG_SINGLE_DOCUMENT = 0.68
LIST_MIN_SUPPORTING_DOCUMENTS = 2
LIST_PARTIAL_SINGLE_DOCUMENT = 0.42

DEFAULT_MAX_DOCUMENTS = 10
DEFAULT_MAX_PER_SOURCE = 2


# ---------------------------------------------------------------------------
# Public result contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """Per-candidate evidence assessment."""

    candidate_id: str
    source: str
    score: float
    status: EvidenceStatus
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "score": self.score,
            "status": self.status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Aggregate decision for the current ranked evidence set."""

    status: EvidenceStatus
    score: float
    relevant_documents: int
    strong_documents: int
    partial_documents: int
    conflicted_documents: int
    selected_candidate_ids: tuple[str, ...] = ()
    items: tuple[EvidenceItem, ...] = ()
    reasons: tuple[str, ...] = ()
    qualification_mode: Literal["verified", "independent"] = "independent"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "score": self.score,
            "relevant_documents": self.relevant_documents,
            "strong_documents": self.strong_documents,
            "partial_documents": self.partial_documents,
            "conflicted_documents": self.conflicted_documents,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "items": [item.to_dict() for item in self.items],
            "reasons": list(self.reasons),
            "qualification_mode": self.qualification_mode,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _requirement_for(
    query: Query | None,
    requirement: RetrievalRequirement | None,
) -> RetrievalRequirement:
    if requirement is not None:
        return requirement
    if query is not None:
        return query.retrieval_requirement
    return RetrievalRequirement()


def _is_list_question(query: Query | None) -> bool:
    return bool(query and query.list_intent.is_list)


def _quality_score(candidate: RetrievalCandidate) -> float:
    quality = candidate.quality
    return _clamp01(
        (0.60 * quality.content_quality)
        + (0.40 * quality.structural_quality)
        - (0.50 * quality.noise)
    )


def _has_content(candidate: RetrievalCandidate) -> bool:
    """Require actual source content before treating a candidate as evidence."""
    document = candidate.document
    if isinstance(document, dict):
        value = document.get("page_content", "")
    else:
        value = getattr(document, "page_content", "")
    return bool(str(value or "").strip())


def _alignment_score(
    candidate: RetrievalCandidate,
    requirement: RetrievalRequirement,
) -> float:
    alignment = candidate.alignment

    # Semantic match and explicit coverage carry the most weight because
    # evidence should reflect meaning, not just token overlap.
    components: list[tuple[float, float]] = [
        (alignment.semantic_match, 0.32),
        (alignment.coverage, 0.28),
        (alignment.topic_match, 0.10),
        (alignment.intent_match, 0.10),
    ]

    if requirement.require_target_alignment:
        components.append((max(alignment.program_match, alignment.entity_match), 0.10))
    else:
        components.append((max(alignment.program_match, alignment.entity_match), 0.05))

    if requirement.require_attribute_alignment:
        components.append((alignment.attribute_match, 0.15))
    else:
        components.append((alignment.attribute_match, 0.05))

    if requirement.require_scope_alignment:
        components.append((alignment.scope_match, 0.10))
    else:
        components.append((alignment.scope_match, 0.05))

    total_weight = sum(weight for _, weight in components)
    if total_weight <= 0:
        return 0.0

    return _clamp01(
        sum(value * weight for value, weight in components) / total_weight
    )


def _required_alignment_threshold(requirement: RetrievalRequirement) -> float:
    if requirement.mode == "exact":
        return EXACT_REQUIRED_ALIGNMENT
    if requirement.mode == "focused":
        return FOCUSED_REQUIRED_ALIGNMENT
    if requirement.mode == "standard":
        return STANDARD_REQUIRED_ALIGNMENT
    return 0.0


def _coverage_threshold(requirement: RetrievalRequirement) -> float:
    if requirement.mode == "exact":
        return EXACT_MIN_COVERAGE
    if requirement.mode == "focused":
        return FOCUSED_MIN_COVERAGE
    if requirement.mode == "standard":
        return STANDARD_MIN_COVERAGE
    return 0.0


def _required_alignment_checks(
    candidate: RetrievalCandidate,
    requirement: RetrievalRequirement,
) -> tuple[bool, list[str]]:
    """Return whether explicit retrieval requirements are met."""

    alignment = candidate.alignment
    reasons: list[str] = []
    threshold = _required_alignment_threshold(requirement)
    coverage_threshold = _coverage_threshold(requirement)

    if requirement.require_target_alignment:
        target_value = max(alignment.program_match, alignment.entity_match)
        if target_value < threshold:
            reasons.append("required target alignment is weak")

    if requirement.require_attribute_alignment:
        if alignment.attribute_match < threshold:
            reasons.append("required attribute alignment is weak")

    if requirement.require_scope_alignment:
        if alignment.scope_match < threshold:
            reasons.append("required scope alignment is weak")

    if alignment.coverage < coverage_threshold:
        reasons.append("evidence coverage is below the mode threshold")

    return not reasons, reasons


def _item_for(
    candidate: RetrievalCandidate,
    requirement: RetrievalRequirement,
    *,
    candidates_are_verified: bool = False,
) -> EvidenceItem:
    if not candidate.quality.usable:
        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=0.0,
            status="insufficient",
            reasons=("candidate content quality is not usable",),
        )

    if not _has_content(candidate):
        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=0.0,
            status="insufficient",
            reasons=("candidate contains no source content",),
        )

    if candidate.alignment.conflicts:
        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=0.0,
            status="conflicted",
            reasons=(
                "explicit semantic conflict: "
                + ", ".join(candidate.alignment.conflicts),
            ),
        )

    alignment_score = _alignment_score(candidate, requirement)
    quality_score = _quality_score(candidate)

    # Verification owns target/entity/scope compatibility. Once that contract
    # is explicitly asserted, evidence scoring must not be diluted by sparse
    # alignment metadata on compact chunks. The score becomes intrinsic
    # evidence quality; relevance has already been established upstream.
    if candidates_are_verified:
        support_score = quality_score
    else:
        support_score = _clamp01(
            (0.78 * alignment_score) + (0.22 * quality_score)
        )

    # Candidate verification is the authoritative qualification step in the
    # end-to-end pipeline. When the caller explicitly tells us candidates have
    # already passed that step, do not re-apply a second hard alignment gate
    # here. This preserves recall while keeping explicit conflicts fail-closed.
    if candidates_are_verified:
        requirements_met, requirement_reasons = True, []
    else:
        requirements_met, requirement_reasons = _required_alignment_checks(
            candidate,
            requirement,
        )

    if requirement.reject_explicit_conflict and candidate.alignment.conflicts:
        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=0.0,
            status="conflicted",
            reasons=requirement_reasons or ("explicit candidate conflict",),
        )

    if not requirements_met:
        status: EvidenceStatus = "partial" if support_score >= PARTIAL_THRESHOLD else "insufficient"
        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=support_score,
            status=status,
            reasons=tuple(requirement_reasons),
        )

    if candidates_are_verified:
        if support_score >= SUPPORTED_THRESHOLD:
            verified_status: EvidenceStatus = "supported"
        elif support_score >= PARTIAL_THRESHOLD and requirement.allow_partial_evidence:
            verified_status = "partial"
        else:
            verified_status = "insufficient"

        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=support_score,
            status=verified_status,
            reasons=(
                "candidate passed upstream verification; evidence uses intrinsic content quality only",
            ),
        )

    if support_score >= SUPPORTED_THRESHOLD:
        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=support_score,
            status="supported",
            reasons=("semantic alignment and coverage are sufficient",),
        )

    if support_score >= PARTIAL_THRESHOLD and requirement.allow_partial_evidence:
        return EvidenceItem(
            candidate_id=candidate.document_id,
            source=candidate.source,
            score=support_score,
            status="partial",
            reasons=("evidence is relevant but not strongly complete",),
        )

    return EvidenceItem(
        candidate_id=candidate.document_id,
        source=candidate.source,
        score=support_score,
        status="insufficient",
        reasons=("support score is below the evidence threshold",),
    )


def _dedupe_candidates(
    candidates: Iterable[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Preserve first occurrence of each stable candidate identity."""
    selected: list[RetrievalCandidate] = []
    seen: set[str] = set()

    for candidate in candidates:
        key = candidate.document_id
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)

    return selected


def _select_supported(
    items: Sequence[EvidenceItem],
    *,
    max_documents: int,
    max_per_source: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    source_counts: dict[str, int] = {}

    # Preserve upstream ranking order. Evidence is a qualification layer,
    # not a second ranking system.
    for item in items:
        if item.status not in {"supported", "partial"}:
            continue

        count = source_counts.get(item.source, 0)
        if count >= max_per_source:
            continue

        selected.append(item.candidate_id)
        source_counts[item.source] = count + 1

        if len(selected) >= max_documents:
            break

    return tuple(selected)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_evidence(
    candidates: Sequence[RetrievalCandidate] | Iterable[RetrievalCandidate],
    *,
    query: Query | None = None,
    requirement: RetrievalRequirement | None = None,
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    max_per_source: int = DEFAULT_MAX_PER_SOURCE,
    candidates_are_verified: bool = False,
) -> EvidenceAssessment:
    """Assess whether ranked candidates are sufficient for answering.

    Pipeline contract:
        retrieval -> verification -> evidence -> coverage -> packaging

    The input is expected to already be ordered by retrieval/ranking, so this
    function never re-ranks. When ``candidates_are_verified`` is true, upstream
    verification is authoritative for relevance and target/scope compatibility;
    this stage only removes unusable/conflicted candidates and determines
    whether at least one qualified evidence item is available.

    When called without verification authority, the function remains conservative
    and enforces the explicit retrieval requirement locally. This keeps the
    public function safe for isolated tests and legacy callers without making
    the production graph pay for duplicate qualification.
    """
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if max_per_source <= 0:
        raise ValueError("max_per_source must be positive")

    materialized = _dedupe_candidates(candidates)
    requirement = _requirement_for(query, requirement)

    if not materialized:
        return EvidenceAssessment(
            status="insufficient",
            score=0.0,
            relevant_documents=0,
            strong_documents=0,
            partial_documents=0,
            conflicted_documents=0,
            reasons=("no retrieval candidates were provided",),
            qualification_mode=("verified" if candidates_are_verified else "independent"),
        )

    items = tuple(
        _item_for(
            candidate,
            requirement,
            candidates_are_verified=candidates_are_verified,
        )
        for candidate in materialized
    )

    strong_items = [item for item in items if item.status == "supported"]
    partial_items = [item for item in items if item.status == "partial"]
    conflicted_items = [item for item in items if item.status == "conflicted"]

    supportive_scores = [
        item.score
        for item in items
        if item.status in {"supported", "partial"}
    ]

    top_score = max(supportive_scores, default=0.0)
    # A second supporting document improves confidence but cannot compensate
    # for a very weak primary candidate.
    secondary_bonus = min(0.15, max(0, len(supportive_scores) - 1) * 0.05)
    aggregate_score = _clamp01(top_score + secondary_bonus)

    is_list = _is_list_question(query)

    # -------------------------------------------------------------------
    # Contract boundary: verified candidates are already relevance-qualified.
    # Evidence must not re-apply target/attribute/scope gates here. It only
    # rejects unusable/conflicted evidence and decides whether any qualified
    # evidence is available for the next layer. This is critical for compact
    # chunks whose semantic alignment fields can be sparse even when the
    # upstream verifier has already established the correct target/scope.
    # -------------------------------------------------------------------
    if candidates_are_verified:
        if strong_items:
            status: EvidenceStatus = "supported"
        elif partial_items and requirement.allow_partial_evidence:
            status = "partial"
        elif conflicted_items and not supportive_scores:
            status = "conflicted"
        else:
            status = "insufficient"

    # Exact/focused modes are conservative for *unverified* callers: a strong
    # count of irrelevant or partial documents cannot substitute for missing
    # required alignment.
    elif requirement.mode == "exact":
        if strong_items:
            status: EvidenceStatus = "supported"
        elif partial_items and requirement.allow_partial_evidence:
            status = "partial"
        elif conflicted_items and not supportive_scores:
            status = "conflicted"
        else:
            status = "insufficient"

    elif is_list or requirement.mode == "broad":
        if strong_items:
            status = "supported"
        elif len(partial_items) >= LIST_MIN_SUPPORTING_DOCUMENTS:
            status = "supported"
        elif partial_items and partial_items[0].score >= LIST_PARTIAL_SINGLE_DOCUMENT:
            status = "partial"
        elif conflicted_items and not supportive_scores:
            status = "conflicted"
        else:
            status = "insufficient"

    else:
        if strong_items:
            status = "supported"
        elif partial_items and requirement.allow_partial_evidence:
            status = "partial"
        elif conflicted_items and not supportive_scores:
            status = "conflicted"
        else:
            status = "insufficient"

    selected = _select_supported(
        items,
        max_documents=max_documents,
        max_per_source=max_per_source,
    )

    reasons: list[str] = []
    if status == "supported":
        reasons.append("retrieved evidence is sufficient to proceed to grounding")
    elif status == "partial":
        reasons.append("evidence supports only part of the requested information")
    elif status == "conflicted":
        reasons.append("retrieved candidates contain explicit semantic conflicts")
    else:
        reasons.append("retrieved evidence is insufficient for a grounded answer")

    return EvidenceAssessment(
        status=status,
        score=aggregate_score,
        relevant_documents=len(strong_items) + len(partial_items),
        strong_documents=len(strong_items),
        partial_documents=len(partial_items),
        conflicted_documents=len(conflicted_items),
        selected_candidate_ids=selected,
        items=items,
        reasons=tuple(reasons),
        qualification_mode=("verified" if candidates_are_verified else "independent"),
    )


def assess_verified_evidence(
    candidates: Sequence[RetrievalCandidate] | Iterable[RetrievalCandidate],
    *,
    query: Query | None = None,
    requirement: RetrievalRequirement | None = None,
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    max_per_source: int = DEFAULT_MAX_PER_SOURCE,
) -> EvidenceAssessment:
    """Assess candidates after the verifier has established relevance.

    This explicit entry point is preferred by orchestration and evaluation
    code because it makes the verification/evidence boundary impossible to
    miss at the call site.
    """
    return assess_evidence(
        candidates,
        query=query,
        requirement=requirement,
        max_documents=max_documents,
        max_per_source=max_per_source,
        candidates_are_verified=True,
    )


def is_evidence_sufficient(
    candidates: Sequence[RetrievalCandidate] | Iterable[RetrievalCandidate],
    *,
    query: Query | None = None,
    requirement: RetrievalRequirement | None = None,
    candidates_are_verified: bool = False,
) -> bool:
    """Return True only when evidence status is ``supported``.

    ``candidates_are_verified`` must be set by the orchestration layer after
    candidate verification. In that mode this function intentionally does not
    re-run relevance gates; verification is the single source of truth for
    target/scope compatibility.
    """
    return assess_evidence(
        candidates,
        query=query,
        requirement=requirement,
        candidates_are_verified=candidates_are_verified,
    ).status == "supported"


__all__ = [
    "EvidenceAssessment",
    "EvidenceItem",
    "EvidenceStatus",
    "assess_evidence",
    "assess_verified_evidence",
    "is_evidence_sufficient",
]
