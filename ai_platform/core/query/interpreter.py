"""
Canonical question-understanding layer for the reusable RAG core.

Responsibilities
----------------
- Understand the user's information need.
- Preserve the original user wording.
- Extract structured meaning from the question.
- Preserve explicit targets, conditions, attributes, and constraints.
- Detect broad/list/comparison/multi-part requests.
- Derive retrieval strictness from the understood question.
- Reject fabricated target/entity values.
- Fall back safely when understanding fails.

Important architecture rule
----------------------------
This module understands the QUESTION.

It does NOT:
- retrieve documents
- perform BM25 matching
- perform vector search
- use institution-specific facts
- use institution-specific vocabularies
- decide what an institution knows
- answer the user

Lexical matching belongs later in retrieval.

Question understanding is meaning-first.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
import re
from typing import Any, Protocol

from ai_platform.core.query.frame import (
    QueryFacet,
    QueryRequirement,
    SemanticQueryFrame,
)

from ai_platform.runtime.llm import query_llm


logger = logging.getLogger(__name__)


# ============================================================
# LLM interface
# ============================================================


class QueryLLM(Protocol):
    """Minimal interface required from the query-understanding model."""

    def invoke(
        self,
        prompt: str,
    ) -> Any:
        ...


# ============================================================
# Limits
# ============================================================

MAX_FACETS = 12
MAX_FIELD_ITEMS = 16
MAX_PRESERVED_TERMS = 24
MAX_SEMANTIC_QUERY_WORDS = 24


VALID_REQUIREMENT_MODES = {
    "broad",
    "standard",
    "focused",
    "exact",
}


VALID_INTERPRETATION_STATUSES = {
    "trusted",
    "review",
    "rejected",
}


# ============================================================
# Canonical question-understanding prompt
# ============================================================


QUESTION_UNDERSTANDING_PROMPT = r"""
You are the question-understanding component of a retrieval system.

Your job is ONLY to understand what the user is asking.

Do NOT answer the question.
Do NOT retrieve documents.
Do NOT use institution-specific knowledge.
Do NOT use a college-specific vocabulary or database.
Do NOT guess missing facts.

The USER QUESTION is authoritative.

Your task is to identify the user's INFORMATION NEED.

Preserve meaning, not just keywords.

You must understand:
- what the user wants to know
- what target the question refers to
- what attribute or information is requested
- what action is requested
- important conditions
- important constraints
- time or duration requirements
- category or person type
- location when explicitly given
- comparison targets
- whether the user wants a list
- whether the question contains multiple independent requests

Very important:

Do NOT replace a specific user target with a similar-looking target.

For example:

User:
"What are the admission routes for M.Sc.?"

The target is "M.Sc."

Do NOT convert it to:
"M.S."
"master's"
"M.Tech"
or another program.

Likewise, preserve distinctions such as:
- M.Sc. vs M.S.
- M.Tech vs B.Tech
- one department vs another department
- one category vs another category
- one date/year vs another date/year

If the meaning of a phrase is uncertain:
- preserve the user's original wording
- leave unsupported structured fields empty
- lower confidence
- use "review"
- do not invent a canonical replacement

Semantic understanding is allowed.

Semantic invention is not.

The semantic representation must describe the user's information need,
not an answer.

Return JSON ONLY.

Required shape:

{
  "semantic_query": "",
  "target": null,
  "request_type": null,

  "facets": [
    {
      "name": "",
      "value": "",
      "required": true,
      "importance": 1.0
    }
  ],

  "qualifiers": [],
  "conditions": [],
  "relations": [],
  "temporal_context": [],
  "comparison_targets": [],
  "preserved_terms": [],

  "is_list_question": false,
  "is_comparison_question": false,
  "is_multi_part": false,

  "confidence": 0.0,

  "needs_clarification": false,
  "clarification_reason": "",

  "language": null,

  "interpretation_status": "review"
}

Generic facet names may include:
- target
- requested_attribute
- subject
- action
- quantity
- unit
- person_type
- category
- location
- eligibility
- condition
- date
- duration
- status
- method
- requirement

Do not create a facet merely because a concept exists.
Create it only when that concept is relevant to the user's request.

Examples:

User:
"What is the price of item A?"

Understand:
- target = item A
- requested_attribute = price

User:
"How do I register for item A?"

Understand:
- target = item A
- requested_attribute = registration procedure

User:
"Can I apply after graduation?"

Understand:
- requested action = apply
- condition = after graduation

User:
"Which options are available?"

Understand:
- list/question breadth is important
- do not invent the options

For Hindi, Hinglish, slang, spelling variations, and informal wording:
understand meaning when the wording supports it, but preserve unusual
literal terms when uncertain.

The original question must remain available for retrieval.

Return JSON only.
""".strip()


# ============================================================
# Text helpers
# ============================================================


def _clean_text(
    value: object,
) -> str:
    return " ".join(
        str(
            value or ""
        ).strip().split()
    )


def _clean_items(
    values: object,
    *,
    limit: int,
) -> tuple[str, ...]:
    if not isinstance(
        values,
        (list, tuple),
    ):
        return ()

    result: list[str] = []
    seen: set[str] = set()

    for value in values:

        cleaned = _clean_text(
            value
        )

        if not cleaned:
            continue

        key = cleaned.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

        if len(result) >= limit:
            break

    return tuple(result)


def _coerce_bool(
    value: object,
    default: bool = False,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().casefold()

        if normalized in {
            "true",
            "yes",
            "1",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "0",
        }:
            return False

    return default


def _coerce_float(
    value: object,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

    return max(
        0.0,
        min(
            1.0,
            number,
        ),
    )


# ============================================================
# Literal preservation
# ============================================================


def _literal_is_grounded(
    original_query: str,
    value: str,
) -> bool:
    """
    Verify that an explicitly extracted entity/target came from the
    user's wording.

    This is NOT question understanding.

    It is a safety boundary preventing the model from inventing a
    different target after the question has already been understood.
    """

    original = _clean_text(
        original_query
    ).casefold()

    candidate = _clean_text(
        value
    ).casefold()

    if not candidate:
        return False

    if candidate in original:
        return True

    original_tokens = re.findall(
        r"\w+",
        original,
        flags=re.UNICODE,
    )

    candidate_tokens = re.findall(
        r"\w+",
        candidate,
        flags=re.UNICODE,
    )

    if not candidate_tokens:
        return False

    width = len(
        candidate_tokens
    )

    if width > len(
        original_tokens
    ):
        return False

    for index in range(
        len(original_tokens)
        - width
        + 1
    ):
        if (
            original_tokens[
                index:index + width
            ]
            == candidate_tokens
        ):
            return True

    return False


def _preserve_grounded_terms(
    original_query: str,
    data: dict[str, Any],
) -> tuple[str, ...]:
    """
    Keep only explicit terms that the model claims are important and
    that can actually be found in the user wording.
    """

    candidates: list[str] = []

    candidates.extend(
        _clean_items(
            data.get(
                "preserved_terms",
                [],
            ),
            limit=MAX_PRESERVED_TERMS,
        )
    )

    target = _clean_text(
        data.get(
            "target",
            "",
        )
    )

    if target:
        candidates.append(target)

    for raw_facet in data.get(
        "facets",
        [],
    ):

        if not isinstance(
            raw_facet,
            dict,
        ):
            continue

        name = _clean_text(
            raw_facet.get(
                "name",
                "",
            )
        ).casefold()

        if name not in {
            "target",
            "subject",
            "entity",
            "program",
            "category",
            "location",
            "person_type",
        }:
            continue

        value = _clean_text(
            raw_facet.get(
                "value",
                "",
            )
        )

        if value:
            candidates.append(
                value
            )

    result: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:

        if not _literal_is_grounded(
            original_query,
            candidate,
        ):
            continue

        key = candidate.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(candidate)

        if len(result) >= MAX_PRESERVED_TERMS:
            break

    return tuple(result)


# ============================================================
# JSON extraction
# ============================================================


def _response_to_text(
    response: Any,
) -> str:
    if response is None:
        return ""

    if isinstance(
        response,
        str,
    ):
        return response.strip()

    content = getattr(
        response,
        "content",
        None,
    )

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if content is not None:
        return str(
            content
        ).strip()

    if isinstance(
        response,
        dict,
    ):
        for key in (
            "content",
            "text",
            "response",
            "output",
        ):
            value = response.get(key)

            if isinstance(
                value,
                str,
            ):
                return value.strip()

    return str(
        response
    ).strip()


def _strip_code_fences(
    text: str,
) -> str:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    )

    return cleaned.strip()


def _extract_json_object(
    text: str,
) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(
        text
    )

    if not cleaned:
        return None

    try:
        parsed = json.loads(
            cleaned
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    start = cleaned.find(
        "{"
    )

    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(cleaned),
    ):
        char = cleaned[index]

        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
            continue

        if char == "}":
            depth -= 1

            if depth != 0:
                continue

            candidate = cleaned[
                start:index + 1
            ]

            try:
                parsed = json.loads(
                    candidate
                )
            except json.JSONDecodeError:
                return None

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

            return None

    return None


# ============================================================
# Facets
# ============================================================


def _build_facets(
    original_query: str,
    raw_facets: object,
) -> tuple[QueryFacet, ...]:
    if not isinstance(
        raw_facets,
        (list, tuple),
    ):
        return ()

    result: list[
        QueryFacet
    ] = []

    for raw in raw_facets:

        if not isinstance(
            raw,
            dict,
        ):
            continue

        name = _clean_text(
            raw.get(
                "name",
                "",
            )
        )

        value = _clean_text(
            raw.get(
                "value",
                "",
            )
        )

        if not name or not value:
            continue

        required = _coerce_bool(
            raw.get(
                "required",
                True,
            ),
            default=True,
        )

        importance = _coerce_float(
            raw.get(
                "importance",
                1.0,
            ),
            default=1.0,
        )

        # Explicit identity-bearing facets must remain grounded in the
        # original question.
        if name.casefold() in {
            "target",
            "subject",
            "entity",
            "program",
            "category",
            "location",
        }:
            if not _literal_is_grounded(
                original_query,
                value,
            ):
                continue

        try:
            facet = QueryFacet(
                name=name,
                value=value,
                required=required,
                importance=importance,
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        result.append(
            facet
        )

        if len(result) >= MAX_FACETS:
            break

    return tuple(result)


# ============================================================
# Requirement derivation
# ============================================================


def _has_facet(
    facets: tuple[QueryFacet, ...],
    name: str,
) -> bool:
    return any(
        facet.name.casefold()
        == name.casefold()
        for facet in facets
    )


def _derive_requirement(
    *,
    target: str | None,
    request_type: str | None,
    facets: tuple[QueryFacet, ...],
    is_list_question: bool,
    is_comparison_question: bool,
    is_multi_part: bool,
) -> QueryRequirement:
    """
    Decide retrieval strictness from question structure.

    This function uses generic question structure only.
    It does not know college-specific words or facts.
    """

    has_target = bool(
        target
    )

    has_attribute = _has_facet(
        facets,
        "requested_attribute",
    )

    has_scope = any(
        facet.name.casefold()
        in {
            "location",
            "category",
            "person_type",
            "scope",
        }
        for facet in facets
    )

    if is_list_question:
        return QueryRequirement(
            mode="broad",
            require_target_alignment=False,
            require_attribute_alignment=False,
            require_scope_alignment=False,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=1,
        )

    if is_comparison_question:
        return QueryRequirement(
            mode="focused",
            require_target_alignment=has_target,
            require_attribute_alignment=has_attribute,
            require_scope_alignment=has_scope,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=0,
        )

    if is_multi_part:
        return QueryRequirement(
            mode="focused",
            require_target_alignment=has_target,
            require_attribute_alignment=has_attribute,
            require_scope_alignment=has_scope,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=1,
        )

    # One target + one explicitly requested attribute is a precise
    # information slot.
    if (
        has_target
        and has_attribute
    ):
        return QueryRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            require_scope_alignment=has_scope,
            reject_explicit_conflict=True,
            allow_partial_evidence=False,
            max_unmatched_required_facets=0,
        )

    if (
        has_target
        or request_type
        or facets
    ):
        return QueryRequirement(
            mode="focused",
            require_target_alignment=has_target,
            require_attribute_alignment=has_attribute,
            require_scope_alignment=has_scope,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=1,
        )

    return QueryRequirement(
        mode="standard",
        require_target_alignment=False,
        require_attribute_alignment=False,
        require_scope_alignment=False,
        reject_explicit_conflict=True,
        allow_partial_evidence=True,
        max_unmatched_required_facets=1,
    )


# ============================================================
# Semantic frame normalization
# ============================================================


def _normalize_interpretation(
    original_query: str,
    data: dict[str, Any],
) -> SemanticQueryFrame:
    normalized_query = _clean_text(
        data.get(
            "normalized_query",
            "",
        )
    ) or original_query

    semantic_query = _clean_text(
        data.get(
            "semantic_query",
            "",
        )
    )

    # Do not allow an oversized model-generated retrieval representation.
    if (
        len(
            semantic_query.split()
        )
        > MAX_SEMANTIC_QUERY_WORDS
    ):
        semantic_query = " ".join(
            semantic_query.split()[
                :MAX_SEMANTIC_QUERY_WORDS
            ]
        )

    target = _clean_text(
        data.get(
            "target",
            "",
        )
    )

    if target and not _literal_is_grounded(
        original_query,
        target,
    ):
        target = ""

    if not target:

        for raw_facet in data.get(
            "facets",
            [],
        ):

            if not isinstance(
                raw_facet,
                dict,
            ):
                continue

            name = _clean_text(
                raw_facet.get(
                    "name",
                    "",
                )
            ).casefold()

            value = _clean_text(
                raw_facet.get(
                    "value",
                    "",
                )
            )

            if (
                name == "target"
                and value
                and _literal_is_grounded(
                    original_query,
                    value,
                )
            ):
                target = value
                break

    target = target or None

    request_type = _clean_text(
        data.get(
            "request_type",
            "",
        )
    )

    request_type = (
        request_type
        or None
    )

    facets = _build_facets(
        original_query,
        data.get(
            "facets",
            [],
        ),
    )

    qualifiers = _clean_items(
        data.get(
            "qualifiers",
            [],
        ),
        limit=MAX_FIELD_ITEMS,
    )

    conditions = _clean_items(
        data.get(
            "conditions",
            [],
        ),
        limit=MAX_FIELD_ITEMS,
    )

    relations = _clean_items(
        data.get(
            "relations",
            [],
        ),
        limit=MAX_FIELD_ITEMS,
    )

    temporal_context = _clean_items(
        data.get(
            "temporal_context",
            [],
        ),
        limit=MAX_FIELD_ITEMS,
    )

    comparison_targets = _clean_items(
        data.get(
            "comparison_targets",
            [],
        ),
        limit=MAX_FIELD_ITEMS,
    )

    preserved_terms = (
        _preserve_grounded_terms(
            original_query,
            data,
        )
    )

    is_list_question = _coerce_bool(
        data.get(
            "is_list_question",
            False,
        )
    )

    is_comparison_question = _coerce_bool(
        data.get(
            "is_comparison_question",
            False,
        )
    )

    is_multi_part = _coerce_bool(
        data.get(
            "is_multi_part",
            False,
        )
    )

    confidence = _coerce_float(
        data.get(
            "confidence",
            0.0,
        )
    )

    needs_clarification = _coerce_bool(
        data.get(
            "needs_clarification",
            False,
        )
    )

    clarification_reason = _clean_text(
        data.get(
            "clarification_reason",
            "",
        )
    )

    language = _clean_text(
        data.get(
            "language",
            "",
        )
    )

    status = _clean_text(
        data.get(
            "interpretation_status",
            "review",
        )
    ).casefold()

    if status not in (
        VALID_INTERPRETATION_STATUSES
    ):
        status = "review"

    if confidence < 0.50:
        status = "review"

    if needs_clarification:
        status = "review"

    requirement = _derive_requirement(
        target=target,
        request_type=request_type,
        facets=facets,
        is_list_question=is_list_question,
        is_comparison_question=is_comparison_question,
        is_multi_part=is_multi_part,
    )

    return SemanticQueryFrame(
        original_query=original_query,
        normalized_query=normalized_query,
        semantic_query=semantic_query,
        target=target,
        request_type=request_type,
        facets=facets,
        qualifiers=qualifiers,
        conditions=conditions,
        relations=relations,
        temporal_context=temporal_context,
        comparison_targets=comparison_targets,
        preserved_terms=preserved_terms,
        is_list_question=is_list_question,
        is_comparison_question=is_comparison_question,
        is_multi_part=is_multi_part,
        requirement=requirement,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_reason=clarification_reason,
        language=language or None,
        interpretation_status=status,
    )


# ============================================================
# Safe fallback
# ============================================================


def fallback_query_frame(
    original_query: str,
) -> SemanticQueryFrame:
    """
    Safe fallback.

    The original user wording remains available to retrieval.
    """

    return SemanticQueryFrame(
        original_query=original_query,
        normalized_query=original_query,
        semantic_query="",
        target=None,
        request_type=None,
        facets=(),
        qualifiers=(),
        conditions=(),
        relations=(),
        temporal_context=(),
        comparison_targets=(),
        preserved_terms=(),
        is_list_question=False,
        is_comparison_question=False,
        is_multi_part=False,
        requirement=QueryRequirement(
            mode="standard",
            require_target_alignment=False,
            require_attribute_alignment=False,
            require_scope_alignment=False,
            reject_explicit_conflict=True,
            allow_partial_evidence=True,
            max_unmatched_required_facets=1,
        ),
        confidence=0.0,
        needs_clarification=False,
        clarification_reason="",
        language=None,
        interpretation_status="review",
    )


# ============================================================
# Main question understanding
# ============================================================


def interpret_query(
    query: str,
    *,
    llm: QueryLLM | None = None,
) -> SemanticQueryFrame:
    """
    Understand one user question.

    The model interprets meaning.
    Python validates the interpretation.
    Retrieval happens elsewhere.
    """

    original_query = _clean_text(
        query
    )

    if not original_query:
        raise ValueError(
            "query cannot be empty."
        )

    active_llm = (
        llm
        or query_llm
    )

    prompt = (
        QUESTION_UNDERSTANDING_PROMPT
        + "\n\nUSER QUESTION:\n"
        + original_query
        + "\n\nJSON:"
    )

    try:

        response = active_llm.invoke(
            prompt
        )

        response_text = (
            _response_to_text(
                response
            )
        )

        parsed = _extract_json_object(
            response_text
        )

    except Exception as exc:

        logger.warning(
            "Question understanding failed: %s",
            exc,
        )

        return fallback_query_frame(
            original_query
        )

    if not parsed:
        return fallback_query_frame(
            original_query
        )

    try:

        return _normalize_interpretation(
            original_query,
            parsed,
        )

    except Exception as exc:

        logger.warning(
            "Question understanding normalization failed: %s",
            exc,
        )

        return fallback_query_frame(
            original_query
        )


# ============================================================
# Diagnostics
# ============================================================


def explain_query_frame(
    frame: SemanticQueryFrame,
) -> dict[str, Any]:
    """
    Return a compact representation for debugging/evaluation.
    """

    if not isinstance(
        frame,
        SemanticQueryFrame,
    ):
        raise TypeError(
            "frame must be a SemanticQueryFrame."
        )

    return {
        "original_query": frame.original_query,
        "normalized_query": frame.normalized_query,
        "semantic_query": frame.semantic_query,
        "target": frame.target,
        "request_type": frame.request_type,
        "facets": [
            facet.to_dict()
            for facet in frame.facets
        ],
        "qualifiers": list(
            frame.qualifiers
        ),
        "conditions": list(
            frame.conditions
        ),
        "relations": list(
            frame.relations
        ),
        "temporal_context": list(
            frame.temporal_context
        ),
        "comparison_targets": list(
            frame.comparison_targets
        ),
        "preserved_terms": list(
            frame.preserved_terms
        ),
        "requirement": (
            frame.requirement.to_dict()
        ),
        "confidence": frame.confidence,
        "needs_clarification": (
            frame.needs_clarification
        ),
        "clarification_reason": (
            frame.clarification_reason
        ),
        "language": frame.language,
        "interpretation_status": (
            frame.interpretation_status
        ),
        "retrieval_text": frame.retrieval_text,
    }


__all__ = [
    "interpret_query",
    "fallback_query_frame",
    "explain_query_frame",
    "QUESTION_UNDERSTANDING_PROMPT",
]