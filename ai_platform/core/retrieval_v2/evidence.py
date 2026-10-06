"""
Retrieval V2 — Evidence Filtering

Purpose
-------
Convert retrieved documents into clean, question-relevant evidence.

Core principle
--------------
A retrieved document is NOT automatically evidence.

The pipeline is:

    retrieved document
        ↓
    local evidence units
        ↓
    scope / target filtering
        ↓
    local relevance scoring
        ↓
    deduplication
        ↓
    final evidence set

This module is:
- deterministic
- institution-agnostic
- LLM-free
- metadata-aware
- conservative

It does NOT attempt to prove truth.
It decides whether a local piece of retrieved content is useful
for the user's question.
"""

from __future__ import annotations

import re
from typing import Iterable

from .evidence_units import build_all_evidence_units
from .models import (
    EvidenceDecision,
    EvidenceScore,
    EvidenceSet,
    EvidenceUnit,
    EvidenceUnitCandidate,
    QuerySpec,
    RankedCandidate,
    RetrievalIntent,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_evidence(
    query: QuerySpec,
    candidates: Iterable[RankedCandidate],
    *,
    max_items: int = 8,
    minimum_score: float = 0.45,
) -> EvidenceSet:
    """
    Convert ranked retrieval candidates into final evidence.

    Important:
    Filtering happens at LOCAL EVIDENCE UNIT level, not whole-document level.

    This prevents a document containing:

        M.Tech information
        ...
        unrelated information
        ...
        Ph.D. eligibility

    from being treated as evidence for:

        M.Tech eligibility
    """

    units = build_all_evidence_units(candidates)

    selected: list[EvidenceUnit] = []
    seen_text: set[str] = set()

    for unit in units:
        decision = score_evidence_unit(
            query,
            unit,
        )

        if not decision.keep:
            continue

        if decision.score.final < minimum_score:
            continue

        normalized_text = _normalize_text(unit.text)

        if not normalized_text:
            continue

        if normalized_text in seen_text:
            continue

        seen_text.add(normalized_text)

        selected.append(
            EvidenceUnit(
                document_id=unit.document_id,
                text=unit.text,
                source=unit.source,
                title=unit.title,
                metadata=unit.metadata,
                score=decision.score.final,
                matched_intents=_matched_intent_indexes(
                    query,
                    unit,
                ),
            )
        )

        if len(selected) >= max_items:
            break

    return EvidenceSet(
        items=tuple(selected),
    )


def score_evidence_unit(
    query: QuerySpec,
    unit: EvidenceUnitCandidate,
) -> EvidenceDecision:
    """
    Score one LOCAL evidence unit against the query.

    This is the central V2 evidence contract.

    A candidate must have at least one intent that is locally supported.

    Explicit metadata mismatches are hard failures.
    """

    metadata = _as_metadata(unit.metadata)

    scope_score = _scope_score(
        query,
        metadata,
    )

    if scope_score == 0.0:
        return _reject(
            scope=scope_score,
            reason="scope_mismatch",
        )

    best = _best_intent_match(
        query,
        unit,
        metadata,
    )

    if best is None:
        return _reject(
            scope=scope_score,
            reason="no_intent_match",
        )

    intent_index, target_score, facet_score, entity_score = best

    semantic_score = _clamp(unit.semantic_score)
    lexical_score = _clamp(unit.lexical_score)

    final = (
        0.30 * semantic_score
        + 0.15 * lexical_score
        + 0.25 * target_score
        + 0.20 * facet_score
        + 0.05 * entity_score
        + 0.05 * scope_score
    )

    reasons: list[str] = [
        f"intent_{intent_index}",
    ]

    if target_score > 0:
        reasons.append("target_match")

    if facet_score > 0:
        reasons.append("facet_match")

    if entity_score > 0:
        reasons.append("entity_match")

    if semantic_score > 0:
        reasons.append("semantic_signal")

    if lexical_score > 0:
        reasons.append("lexical_signal")

    return EvidenceDecision(
        keep=final >= 0.45,
        score=EvidenceScore(
            target=target_score,
            facet=facet_score,
            entity=entity_score,
            semantic=semantic_score,
            lexical=lexical_score,
            scope=scope_score,
            final=min(1.0, final),
        ),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Intent matching
# ---------------------------------------------------------------------------


def _best_intent_match(
    query: QuerySpec,
    unit: EvidenceUnitCandidate,
    metadata: dict[str, object],
) -> tuple[int, float, float, float] | None:
    """
    Find the strongest intent supported by this LOCAL evidence unit.

    Returns:

        (intent_index, target_score, facet_score, entity_score)

    Important:
    An intent is not considered matched merely because one facet appears
    somewhere in the unit. If an explicit target exists, the target must
    also be locally compatible.
    """

    best: tuple[int, float, float, float] | None = None
    best_value = -1.0

    for index, intent in enumerate(query.intents):
        target_score = _target_match(
            intent,
            metadata,
            unit.text,
        )

        facet_score = _facet_match(
            intent,
            metadata,
            unit.text,
        )

        entity_score = _entity_match(
            intent,
            metadata,
            unit.text,
        )

        # ---------------------------------------------------------------
        # Explicit target protection
        # ---------------------------------------------------------------
        #
        # If metadata explicitly says this is another target/program,
        # do not allow a shared facet such as "eligibility" to make it
        # relevant to the requested target.
        #
        # Example:
        #
        # query target = mtech
        # candidate program = phd
        #
        # → reject this intent.
        #
        if intent.target:
            explicit_target = _explicit_target(metadata)

            if (
                explicit_target
                and not _target_values_match(
                    intent.target,
                    explicit_target,
                )
            ):
                continue

        # If the query has both target and facet, require BOTH locally.
        if intent.target and intent.facets:
            if target_score <= 0.0 or facet_score <= 0.0:
                continue

        # Target-only intent.
        elif intent.target:
            if target_score <= 0.0:
                continue

        # Facet-only intent.
        elif intent.facets:
            if facet_score <= 0.0:
                continue

        value = (
            0.50 * target_score
            + 0.35 * facet_score
            + 0.15 * entity_score
        )

        if value > best_value:
            best_value = value
            best = (
                index,
                target_score,
                facet_score,
                entity_score,
            )

    return best


# ---------------------------------------------------------------------------
# Target matching
# ---------------------------------------------------------------------------


def _target_match(
    intent: RetrievalIntent,
    metadata: dict[str, object],
    text: str,
) -> float:
    if not intent.target:
        return 0.0

    target = _normalize_value(
        intent.target,
    )

    explicit_target = _explicit_target(
        metadata,
    )

    if explicit_target:
        if _target_values_match(
            target,
            explicit_target,
        ):
            return 1.0

        return 0.0

    # No explicit metadata:
    # allow local textual evidence as a weaker signal.
    if _contains_term(
        text,
        target,
    ):
        return 0.70

    return 0.0


def _explicit_target(
    metadata: dict[str, object],
) -> str | None:
    """
    Find an explicit target/program value in metadata.

    This is intentionally generic.
    """

    for key in (
        "target",
        "program",
        "program_id",
        "target_id",
    ):
        value = metadata.get(key)

        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            values = [
                _normalize_value(item)
                for item in value
            ]

            values = [
                item
                for item in values
                if item
            ]

            if values:
                return values[0]

        else:
            normalized = _normalize_value(value)

            if normalized:
                return normalized

    return None


def _target_values_match(
    requested: str,
    candidate: str,
) -> bool:
    requested = _normalize_value(requested)
    candidate = _normalize_value(candidate)

    if requested == candidate:
        return True

    aliases = {
        "m.tech": {"mtech", "m.tech", "master of technology"},
        "mtech": {"mtech", "m.tech", "master of technology"},
        "b.tech": {"btech", "b.tech", "bachelor of technology"},
        "btech": {"btech", "b.tech", "bachelor of technology"},
        "ph.d": {"phd", "ph.d", "doctor of philosophy"},
        "phd": {"phd", "ph.d", "doctor of philosophy"},
        "m.sc": {"msc", "m.sc", "master of science"},
        "msc": {"msc", "m.sc", "master of science"},
    }

    requested_values = aliases.get(
        requested,
        {requested},
    )

    candidate_values = aliases.get(
        candidate,
        {candidate},
    )

    return bool(
        requested_values.intersection(
            candidate_values,
        )
    )


# ---------------------------------------------------------------------------
# Facet matching
# ---------------------------------------------------------------------------


def _facet_match(
    intent: RetrievalIntent,
    metadata: dict[str, object],
    text: str,
) -> float:
    if not intent.facets:
        return 0.0

    metadata_values = _metadata_values(
        metadata,
        (
            "facet",
            "facets",
            "topic",
            "topics",
            "category",
        ),
    )

    for facet in intent.facets:
        normalized = _normalize_value(
            facet,
        )

        if normalized in metadata_values:
            return 1.0

        if _contains_term(
            text,
            normalized,
        ):
            return 0.65

    return 0.0


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------


def _entity_match(
    intent: RetrievalIntent,
    metadata: dict[str, object],
    text: str,
) -> float:
    if not intent.entities:
        return 0.0

    metadata_values = _metadata_values(
        metadata,
        (
            "entity",
            "entities",
            "department",
            "school",
        ),
    )

    for entity in intent.entities:
        normalized = _normalize_value(
            entity,
        )

        if normalized in metadata_values:
            return 1.0

        if _contains_term(
            text,
            normalized,
        ):
            return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def _scope_score(
    query: QuerySpec,
    metadata: dict[str, object],
) -> float:
    """
    Institution scope is a filter when explicitly available.

    Missing candidate scope is not automatically rejected at V2
    foundation stage because some existing stores may not have
    institution metadata on every chunk.
    """

    if not query.institution_id:
        return 1.0

    candidate_scope = metadata.get(
        "institution_id",
    )

    if candidate_scope is None:
        return 0.5

    return (
        1.0
        if _normalize_value(candidate_scope)
        == _normalize_value(query.institution_id)
        else 0.0
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _matched_intent_indexes(
    query: QuerySpec,
    unit: EvidenceUnitCandidate,
) -> tuple[int, ...]:
    metadata = _as_metadata(unit.metadata)

    matched: list[int] = []

    for index, intent in enumerate(query.intents):
        result = _best_intent_match_for_single(
            intent,
            unit,
            metadata,
        )

        if result:
            matched.append(index)

    return tuple(matched)


def _best_intent_match_for_single(
    intent: RetrievalIntent,
    unit: EvidenceUnitCandidate,
    metadata: dict[str, object],
) -> bool:
    if intent.target:
        explicit_target = _explicit_target(
            metadata,
        )

        if (
            explicit_target
            and not _target_values_match(
                intent.target,
                explicit_target,
            )
        ):
            return False

        target = _target_match(
            intent,
            metadata,
            unit.text,
        )

        if target <= 0:
            return False

    if intent.facets:
        facet = _facet_match(
            intent,
            metadata,
            unit.text,
        )

        if facet <= 0:
            return False

    return True


def _metadata_values(
    metadata: dict[str, object],
    keys: tuple[str, ...],
) -> set[str]:
    values: set[str] = set()

    for key in keys:
        value = metadata.get(key)

        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            for item in value:
                normalized = _normalize_value(item)

                if normalized:
                    values.add(normalized)

        else:
            normalized = _normalize_value(value)

            if normalized:
                values.add(normalized)

    return values


def _contains_term(
    text: str,
    term: str,
) -> bool:
    if not text or not term:
        return False

    # Multi-word expressions need whitespace-aware matching.
    escaped = re.escape(
        term,
    )

    return bool(
        re.search(
            rf"(?<!\w){escaped}(?!\w)",
            text.casefold(),
        )
    )


def _normalize_text(
    text: str,
) -> str:
    return " ".join(
        str(text).casefold().split(),
    )


def _normalize_value(
    value: object,
) -> str:
    return " ".join(
        str(value).strip().casefold().split(),
    )


def _as_metadata(
    metadata: object,
) -> dict[str, object]:
    if isinstance(metadata, dict):
        return metadata

    return {}


def _clamp(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def _reject(
    *,
    scope: float,
    reason: str,
) -> EvidenceDecision:
    return EvidenceDecision(
        keep=False,
        score=EvidenceScore(
            scope=scope,
            final=0.0,
        ),
        reasons=(reason,),
    )