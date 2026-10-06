"""Production candidate verification for the reusable RAG core.

Contract
--------
Verification is a SECURITY + RELEVANCE boundary.

It answers one question:
    "Is this retrieved candidate safe and relevant enough to participate in
     ranking and final-context construction?"

It does *not* try to prove that one chunk contains the entire answer.
Multiple verified chunks are allowed to jointly answer a requirements,
list, descriptive, or process question.

Hard rejection is reserved for:
    - explicit semantic conflicts;
    - explicit scope mismatches when the query requires scope;
    - clearly unusable / clearly irrelevant candidates;
    - strict factual-relation queries whose requested relation is genuinely
      absent from the candidate.

Normal descriptive / requirements questions are intentionally recall-friendly:
partial factual support is accepted when the candidate is meaningfully tied to
the request. Ranking and the final-context builder decide which approved
information is most useful to the answer model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ai_platform.core.query.models import SemanticQueryFrame
from ai_platform.core.retrieval.contracts import RetrievalCandidate
from ai_platform.core.retrieval.evidence_matcher import (
    EvidenceAnalysis,
    analyze_candidate,
    requires_scope,
)


VerificationStatus = Literal["verified", "rejected", "uncertain"]

# These markers identify an actual relationship/comparison question. A direct
# lookup such as a fee, percentage, GATE requirement, or years of experience
# does not need candidate-local relation proof; the fact can live in one of
# several complementary verified chunks.
_STRICT_RELATION_MARKERS = (
    "included in",
    "included within",
    "part of",
    "separate from",
    "separate",
    "exempt from",
    "exemption",
    "waiver",
    "difference between",
    "versus",
    " vs ",
    "compare",
    "comparison",
)


def _clean_query(frame: SemanticQueryFrame) -> str:
    return " ".join(
        str(
            getattr(frame, "original_query", "")
            or getattr(frame, "semantic_query", "")
            or ""
        ).casefold().split()
    )


def _strict_relation_query(frame: SemanticQueryFrame) -> bool:
    """Return True only for explicit relational/comparison questions."""
    query = f" {_clean_query(frame)} "
    return any(marker in query for marker in _STRICT_RELATION_MARKERS)


def _target_required(frame: SemanticQueryFrame) -> bool:
    requirement = getattr(frame, "requirement", None)
    explicit = getattr(requirement, "require_target_alignment", None)
    if explicit is not None:
        return bool(explicit) and bool(getattr(frame, "target", None))
    return bool(getattr(frame, "target", None))


def _attribute_required(frame: SemanticQueryFrame) -> bool:
    requirement = getattr(frame, "requirement", None)
    explicit = getattr(requirement, "require_attribute_alignment", None)
    if explicit is not None:
        return bool(explicit)
    return bool(getattr(frame, "facets", ()) or getattr(frame, "request_type", None))


def _scope_ok(frame: SemanticQueryFrame, analysis: EvidenceAnalysis) -> bool:
    if analysis.conflict:
        return False
    if not requires_scope(frame):
        return True
    # Zero is treated as neutral by the underlying matcher. A positive score
    # must meet the explicit-scope threshold.
    return analysis.scope_score >= 0.50


def _candidate_relevance(
    analysis: EvidenceAnalysis,
    *,
    target_required: bool,
    attribute_required: bool,
) -> float:
    """Bounded relevance score used only for the verification decision."""
    # Target and attribute are useful signals, but neither is required to be
    # complete for a normal descriptive/requirements request.
    components = [
        (analysis.query_overlap, 0.30),
        (analysis.semantic_score, 0.25),
        (analysis.target_score, 0.25 if target_required else 0.15),
        (analysis.attribute_score, 0.15 if attribute_required else 0.10),
        (analysis.entity_score, 0.05),
    ]
    weight = sum(w for _, w in components)
    return max(
        0.0,
        min(1.0, sum(score * w for score, w in components) / max(weight, 1e-9)),
    )


def _has_meaningful_signal(
    analysis: EvidenceAnalysis,
    *,
    target_required: bool,
    attribute_required: bool,
) -> bool:
    """Accept partial evidence when at least one strong local signal exists."""
    strong_local = max(
        analysis.target_score,
        analysis.attribute_score if attribute_required else 0.0,
        analysis.entity_score,
        analysis.query_overlap,
    )

    # Hard evidence is enough on its own for a verified candidate when there is
    # no explicit conflict. This is intentionally generous because the final
    # answer may legitimately need several separate chunks.
    if analysis.hard_evidence:
        return True

    # A target match is valuable, but a single candidate is allowed to carry
    # only one requested facet. This is critical for multi-part answers where
    # the program/entity heading may live in one chunk and the requested fact
    # (for example work experience or a percentage) lives in the next chunk.
    if target_required and analysis.target_score >= 0.45:
        return analysis.query_overlap >= 0.10 or analysis.semantic_score >= 0.30

    if attribute_required and analysis.attribute_score >= 0.45:
        return analysis.query_overlap >= 0.20 or analysis.semantic_score >= 0.45

    # When a candidate does not repeat the target explicitly, allow it when its
    # requested attribute is strongly grounded and the semantic/lexical signal
    # is still meaningful. Explicit conflicts remain a hard rejection above.
    if target_required and attribute_required and analysis.attribute_score >= 0.45:
        return analysis.query_overlap >= 0.20 or analysis.semantic_score >= 0.45

    return (
        strong_local >= 0.35
        and (analysis.query_overlap >= 0.15 or analysis.semantic_score >= 0.35)
    )


def _decision_reasons(
    frame: SemanticQueryFrame,
    analysis: EvidenceAnalysis,
    *,
    relevance: float,
    scope_ok: bool,
    strict_relation: bool,
) -> tuple[str, ...]:
    reasons = list(analysis.reasons)

    if scope_ok:
        reasons.append("scope_compatible")
    else:
        reasons.append("scope_mismatch")

    if strict_relation:
        reasons.append("strict_relation_request")

    reasons.append(
        "partial_evidence_allowed"
        if not strict_relation
        else "strict_relation_requires_local_proof"
    )

    reasons.append(f"verification_relevance={relevance:.3f}")
    return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    candidate: RetrievalCandidate
    status: VerificationStatus
    accepted: bool
    target_grounded: bool
    attribute_grounded: bool
    scope_compatible: bool
    semantic_compatible: bool
    conflict_detected: bool
    coverage: float
    score: float
    reasons: tuple[str, ...]
    evidence: EvidenceAnalysis | None = None

    def to_dict(self) -> dict:
        payload = {
            "candidate_id": self.candidate.document_id,
            "status": self.status,
            "accepted": self.accepted,
            "target_grounded": self.target_grounded,
            "attribute_grounded": self.attribute_grounded,
            "scope_compatible": self.scope_compatible,
            "semantic_compatible": self.semantic_compatible,
            "conflict_detected": self.conflict_detected,
            "coverage": self.coverage,
            "score": self.score,
            "reasons": list(self.reasons),
        }
        if self.evidence is not None:
            payload["evidence"] = self.evidence.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class VerificationBatch:
    verified: tuple[RetrievalCandidate, ...]
    uncertain: tuple[RetrievalCandidate, ...]
    rejected: tuple[RetrievalCandidate, ...]
    decisions: tuple[VerificationDecision, ...]

    def to_dict(self) -> dict:
        return {
            "verified_ids": [x.document_id for x in self.verified],
            "uncertain_ids": [x.document_id for x in self.uncertain],
            "rejected_ids": [x.document_id for x in self.rejected],
            "decisions": [x.to_dict() for x in self.decisions],
        }


def verify_candidate(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
) -> VerificationDecision:
    """Verify one candidate without requiring it to contain the whole answer."""
    analysis = analyze_candidate(frame, candidate)
    target_required = _target_required(frame)
    attribute_required = _attribute_required(frame)
    scope_ok = _scope_ok(frame, analysis)
    strict_relation = _strict_relation_query(frame)
    relevance = _candidate_relevance(
        analysis,
        target_required=target_required,
        attribute_required=attribute_required,
    )

    target_grounded = analysis.target_score >= 0.45
    attribute_grounded = analysis.attribute_score >= 0.45
    semantic_compatible = analysis.semantic_score >= 0.35
    conflict_detected = bool(analysis.conflict)

    reasons = _decision_reasons(
        frame,
        analysis,
        relevance=relevance,
        scope_ok=scope_ok,
        strict_relation=strict_relation,
    )

    # ------------------------------------------------------------
    # Hard safety failures
    # ------------------------------------------------------------
    if conflict_detected:
        return VerificationDecision(
            candidate=candidate,
            status="rejected",
            accepted=False,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=False,
            semantic_compatible=semantic_compatible,
            conflict_detected=True,
            coverage=analysis.factual_score,
            score=0.0,
            reasons=tuple(dict.fromkeys((*reasons, "explicit_conflict"))),
            evidence=analysis,
        )

    if not scope_ok:
        return VerificationDecision(
            candidate=candidate,
            status="rejected",
            accepted=False,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=False,
            semantic_compatible=semantic_compatible,
            conflict_detected=False,
            coverage=analysis.factual_score,
            score=0.0,
            reasons=reasons,
            evidence=analysis,
        )

    # ------------------------------------------------------------
    # Strict factual relations remain strict.
    # ------------------------------------------------------------
    # STAGE2_V4_EXACT_HARD_GATE
    # Exact requirements are proof requests: generic relevance cannot
    # replace required target / attribute / local relation evidence.
    requirement = getattr(frame, "requirement", None)
    exact_request = str(
        getattr(requirement, "mode", "") or ""
    ).casefold() == "exact"

    if exact_request:
        relation_supported = analysis.relation_score >= 0.70

        if target_required and not target_grounded:
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=False,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_ok,
                semantic_compatible=semantic_compatible,
                conflict_detected=False,
                coverage=analysis.factual_score,
                score=analysis.factual_score,
                reasons=tuple(dict.fromkeys((*reasons, "required_target_not_grounded"))),
                evidence=analysis,
            )

        if attribute_required and not attribute_grounded:
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=False,
                scope_compatible=scope_ok,
                semantic_compatible=semantic_compatible,
                conflict_detected=False,
                coverage=analysis.factual_score,
                score=analysis.factual_score,
                reasons=tuple(dict.fromkeys((*reasons, "required_attribute_not_grounded"))),
                evidence=analysis,
            )

        if target_required and attribute_required and not relation_supported:
            return VerificationDecision(
                candidate=candidate,
                status="rejected",
                accepted=False,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=scope_ok,
                semantic_compatible=semantic_compatible,
                conflict_detected=False,
                coverage=analysis.factual_score,
                score=analysis.factual_score,
                reasons=tuple(dict.fromkeys((*reasons, "required_target_attribute_relation_not_grounded"))),
                evidence=analysis,
            )

    if strict_relation:
        relation_supported = analysis.relation_score >= 0.70
        target_signal = target_grounded or not target_required
        attribute_signal = attribute_grounded or not attribute_required
        strong_direct_support = (
            relation_supported
            and target_signal
            and attribute_signal
        )

        if strong_direct_support:
            return VerificationDecision(
                candidate=candidate,
                status="verified",
                accepted=True,
                target_grounded=target_grounded,
                attribute_grounded=attribute_grounded,
                scope_compatible=True,
                semantic_compatible=semantic_compatible,
                conflict_detected=False,
                coverage=max(analysis.factual_score, analysis.relation_score),
                score=max(relevance, analysis.relation_score),
                reasons=reasons,
                evidence=analysis,
            )

        # For an explicit relation request, generic semantic similarity is not
        # enough. This is the one place where candidate-local proof matters.
        return VerificationDecision(
            candidate=candidate,
            status="rejected",
            accepted=False,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=True,
            semantic_compatible=semantic_compatible,
            conflict_detected=False,
            coverage=analysis.factual_score,
            score=analysis.factual_score,
            reasons=tuple(dict.fromkeys((*reasons, "relation_not_locally_grounded"))),
            evidence=analysis,
        )

    # ------------------------------------------------------------
    # Normal descriptive / requirements / process requests.
    # ------------------------------------------------------------
    meaningful = _has_meaningful_signal(
        analysis,
        target_required=target_required,
        attribute_required=attribute_required,
    )

    if meaningful and relevance >= 0.28:
        return VerificationDecision(
            candidate=candidate,
            status="verified",
            accepted=True,
            target_grounded=target_grounded,
            attribute_grounded=attribute_grounded,
            scope_compatible=True,
            semantic_compatible=semantic_compatible,
            conflict_detected=False,
            coverage=max(analysis.factual_score, relevance),
            score=relevance,
            reasons=reasons,
            evidence=analysis,
        )

    # Borderline candidates are rejected rather than parked in an uncertain
    # queue. This keeps the production boundary deterministic: ranked_candidates
    # always means "approved for generation context".
    return VerificationDecision(
        candidate=candidate,
        status="rejected",
        accepted=False,
        target_grounded=target_grounded,
        attribute_grounded=attribute_grounded,
        scope_compatible=True,
        semantic_compatible=semantic_compatible,
        conflict_detected=False,
        coverage=analysis.factual_score,
        score=relevance,
        reasons=tuple(dict.fromkeys((*reasons, "insufficient_relevance"))),
        evidence=analysis,
    )


def verify_candidates(
    frame: SemanticQueryFrame,
    candidates: Sequence[RetrievalCandidate],
) -> VerificationBatch:
    verified: list[RetrievalCandidate] = []
    rejected: list[RetrievalCandidate] = []
    decisions: list[VerificationDecision] = []

    for candidate in candidates:
        decision = verify_candidate(frame, candidate)
        decisions.append(decision)
        if decision.accepted:
            verified.append(candidate)
        else:
            rejected.append(candidate)

    return VerificationBatch(
        verified=tuple(verified),
        # Production V3 intentionally has no soft-uncertain queue. Borderline
        # candidates are rejected; approved candidates are safe to rank.
        uncertain=(),
        rejected=tuple(rejected),
        decisions=tuple(decisions),
    )


def select_verified_candidates(
    frame: SemanticQueryFrame,
    candidates: Sequence[RetrievalCandidate],
) -> tuple[RetrievalCandidate, ...]:
    return verify_candidates(frame, candidates).verified


def explain_verification(
    frame: SemanticQueryFrame,
    candidate: RetrievalCandidate,
) -> dict:
    return verify_candidate(frame, candidate).to_dict()


__all__ = [
    "VerificationStatus",
    "VerificationDecision",
    "VerificationBatch",
    "verify_candidate",
    "verify_candidates",
    "select_verified_candidates",
    "explain_verification",
]
