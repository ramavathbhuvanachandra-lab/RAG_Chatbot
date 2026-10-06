"""
Retrieval V2 — Query Preparation

Converts a raw user query into a small QuerySpec.

Important:
This module does not contain institution-specific vocabularies.

Institution/program/topic mappings can be supplied later through
configuration or semantic registries.
"""

from __future__ import annotations

import re

from .models import QuerySpec, RetrievalIntent


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """
    Normalize whitespace while preserving the user's words.

    We deliberately do not aggressively stem or rewrite the query.
    Dense retrieval should see the natural language question.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    return _WHITESPACE_RE.sub(" ", query)


def build_query_spec(
    query: str,
    *,
    institution_id: str | None = None,
    language: str | None = None,
) -> QuerySpec:
    """
    Build the initial query specification.

    At V2 foundation stage, intent extraction remains conservative.
    Retrieval should never depend on aggressive query rewriting.
    """

    normalized = normalize_query(query)

    intent = RetrievalIntent(
        keywords=tuple(_keyword_candidates(normalized)),
    )

    return QuerySpec(
        raw_query=normalized,
        intents=(intent,),
        institution_id=institution_id,
        language=language,
    )


def with_intents(
    spec: QuerySpec,
    intents: list[RetrievalIntent] | tuple[RetrievalIntent, ...],
) -> QuerySpec:
    """
    Return a QuerySpec with explicitly resolved intents.

    This gives us a clean seam for a future semantic/query planner
    without changing the retrieval pipeline.
    """

    normalized_intents = tuple(
        intent.normalized()
        for intent in intents
        if intent is not None
    )

    return QuerySpec(
        raw_query=spec.raw_query,
        intents=normalized_intents,
        institution_id=spec.institution_id,
        language=spec.language,
        request_type=spec.request_type,
        metadata=spec.metadata,
    )


def _keyword_candidates(query: str) -> list[str]:
    """
    Lightweight lexical representation.

    Keep meaningful tokens and avoid punctuation-only fragments.
    """

    tokens = re.findall(r"\b[\w.-]+\b", query.casefold())

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "for",
        "how",
        "i",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "what",
        "where",
        "which",
        "who",
        "with",
    }

    return [
        token
        for token in tokens
        if token not in stopwords and len(token) > 1
    ]