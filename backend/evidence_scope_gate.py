"""
IIT Jodhpur V1 — Phase 8E
Answer Evidence Scope Gate

Purpose
-------
Select the final answer evidence from candidates that have already passed
retrieval/reranking/scope verification.

Rules
-----
1. Broad/non-scoped queries keep the incoming order.
2. Program-specific queries require the requested program AND requested
   topic to be represented by the source path.
3. Non-program scoped queries may establish scope from either source path
   or document content.
4. Among compatible candidates, direct content relevance determines which
   documents enter the final context budget.
5. If no compatible scoped candidate exists, preserve the incoming evidence
   instead of guessing.

This gate does not introduce institution-specific filenames or hardcoded
topic lists.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from backend.retriever import (
    detect_programs,
    detect_entities,
    detect_topics,
    get_source,
    normalize_text,
)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "did", "do", "does", "for", "from", "get", "give", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "related", "the",
    "their", "there", "this", "to", "what", "which", "who", "why", "with",
    "would", "you", "your", "about", "into", "than", "then", "that",
    "these", "those", "was", "were", "will", "have", "has", "had",
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _as_set(value: Any) -> set[str]:
    """Normalize detector output so list/set/tuple outputs are all safe."""
    if value is None:
        return set()

    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]

    return {
        str(item).casefold()
        for item in values
        if str(item).strip()
    }


def _program_signals(text: str) -> set[str]:
    return _as_set(
        detect_programs(
            _string(text)
        )
    )


def _scope_signals(text: str) -> set[str]:
    value = _string(text)
    if not value:
        return set()

    return (
        _program_signals(value)
        | _as_set(
            detect_entities(value)
        )
    )


def _topic_signals(text: str) -> set[str]:
    """Normalize the existing generic topic detector output."""
    value = _string(text)
    if not value:
        return set()

    return _as_set(
        detect_topics(value)
    )


def _source_topics(document: Any) -> set[str]:
    source = get_source(document)

    source_topics = _topic_signals(source)
    if source_topics:
        return source_topics

    return _topic_signals(
        getattr(document, "page_content", "")
    )


def _content_tokens(text: str) -> set[str]:
    value = normalize_text(text)

    return {
        token
        for token in value.split()
        if len(token) > 2 and token not in _STOPWORDS
    }


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    for token in normalize_text(query).split():
        if (
            len(token) <= 2
            or token in _STOPWORDS
            or token in seen
        ):
            continue

        seen.add(token)
        terms.append(token)

    return terms


def _query_term_weights(
    query: str,
    documents: Sequence[Any],
) -> dict[str, float]:
    """
    Give rarer query terms more weight inside the already-compatible
    candidate set. This lets a concrete term such as "robotics" recover a
    lower-ranked but directly relevant document.
    """
    terms = _query_terms(query)
    if not terms:
        return {}

    total_documents = max(
        len(documents),
        1,
    )

    document_frequencies = {
        term: 0
        for term in terms
    }

    for document in documents:
        tokens = _content_tokens(
            getattr(
                document,
                "page_content",
                "",
            )
        )

        tokens |= _content_tokens(
            get_source(document)
        )

        for term in terms:
            if term in tokens:
                document_frequencies[term] += 1

    return {
        term: 1.0
        + math.log(
            (total_documents + 1.0)
            / (frequency + 1.0)
        )
        for term, frequency in document_frequencies.items()
    }


def _relevance_score(
    query: str,
    document: Any,
    term_weights: dict[str, float],
) -> float:
    content = _content_tokens(
        getattr(
            document,
            "page_content",
            "",
        )
    )

    source = _content_tokens(
        get_source(document)
    )

    if not term_weights:
        return 0.0

    total_weight = (
        sum(term_weights.values())
        or 1.0
    )

    content_weight = sum(
        weight
        for term, weight in term_weights.items()
        if term in content
    )

    source_only_weight = sum(
        weight
        for term, weight in term_weights.items()
        if term in source
        and term not in content
    )

    score = (
        content_weight
        / total_weight
    )

    score += (
        0.15
        * source_only_weight
        / total_weight
    )

    normalized_query = normalize_text(query)
    normalized_content = normalize_text(
        getattr(
            document,
            "page_content",
            "",
        )
    )

    query_terms = _query_terms(query)

    if len(query_terms) >= 2:
        bigram_hits = 0

        for first, second in zip(
            query_terms,
            query_terms[1:],
        ):
            if re.search(
                rf"\b{re.escape(first)}\s+{re.escape(second)}\b",
                normalized_content,
            ):
                bigram_hits += 1

        score += (
            0.05
            * bigram_hits
        )

    if (
        normalized_query
        and normalized_query in normalized_content
    ):
        score += 0.10

    return score


def _rank_by_answer_relevance(
    query: str,
    documents: Sequence[Any],
) -> list[Any]:
    docs = list(documents or [])
    if not docs:
        return []

    weights = _query_term_weights(
        query,
        docs,
    )

    scored = []

    for index, document in enumerate(docs):
        scored.append(
            (
                _relevance_score(
                    query,
                    document,
                    weights,
                ),
                -index,
                document,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    return [
        item[2]
        for item in scored
    ]


def select_answer_evidence(
    *,
    query: str,
    documents: Sequence[Any],
    max_documents: int = 5,
) -> list[Any]:
    """
    Select final evidence for answer generation.

    The selector operates only on documents that have already survived the
    earlier retrieval/relevance stages. It performs narrow scope protection
    where the question explicitly names a program/entity and uses content
    relevance to prevent a final top-k cut from hiding strong evidence.
    """
    if max_documents <= 0:
        return []

    docs = list(documents or [])
    if not docs:
        return []

    query = _string(query)

    query_programs = _program_signals(query)
    query_signals = _scope_signals(query)
    query_topics = _topic_signals(query)

    # ---------------------------------------------------------
    # Broad / genuinely non-scoped query
    # ---------------------------------------------------------
    #
    # Do not invent a second ranking policy for a broad query. The earlier
    # retrieval stages already produced an evidence order.
    #
    if not query_signals:
        return docs[:max_documents]

    matching: list[Any] = []

    for document in docs:
        source = get_source(document)
        content = getattr(
            document,
            "page_content",
            "",
        ) or ""

        source_scope = _scope_signals(source)
        content_scope = _scope_signals(content)

        source_programs = _program_signals(source)

        # -----------------------------------------------------
        # Explicit program query
        # -----------------------------------------------------
        #
        # Program questions stay strict. The source path must identify both
        # the requested program and the requested topic. This prevents an
        # M.Sc. program-overview source from satisfying an M.Sc. admission
        # question.
        #
        if query_programs:
            program_match = bool(
                query_programs
                & source_programs
            )

            if not program_match:
                continue

            source_topics = _source_topics(document)

            if query_topics and not (
                query_topics & source_topics
            ):
                continue

            matching.append(document)
            continue

        # -----------------------------------------------------
        # Non-program scoped query
        # -----------------------------------------------------
        #
        # Scope may be established by source metadata OR content. This is
        # important because a valid chunk can live in an unexpected folder.
        #
        if not (
            query_signals
            & (
                source_scope
                | content_scope
            )
        ):
            continue

        source_topics = _source_topics(document)
        content_topics = _topic_signals(content)

        candidate_topics = (
            source_topics
            | content_topics
        )

        # Reject an explicitly incompatible topic, but do not require topic
        # agreement when the candidate does not expose a detectable topic.
        if (
            query_topics
            and candidate_topics
            and not (query_topics & candidate_topics)
        ):
            continue

        matching.append(document)

    if matching:
        ranked = _rank_by_answer_relevance(
            query,
            matching,
        )
        return ranked[:max_documents]

    # No compatible scoped evidence exists. Preserve the pre-existing
    # evidence instead of guessing.
    return docs[:max_documents]