"""
IIT Jodhpur V1 — Phase 8E
Answer Evidence Scope Gate

Final production rule
---------------------
For a narrow query, a document is promoted into the answer evidence only
when both of these are true:

    1. The document explicitly matches the requested program/entity.
    2. When the query expresses a topic/scope, the source also matches that
       requested topic/scope.

This prevents a program-overview document from being selected for an
admission question merely because both mention the same program.

No institution-specific filenames, IDs, or folder blacklists are used.
"""

from __future__ import annotations

from typing import Any, Sequence

from backend.retriever import (
    detect_programs,
    detect_entities,
    detect_query_scope,
    get_source,
)


def _string(value: Any) -> str:
    """Return a safe normalized string."""
    return str(
        value or ""
    ).strip()


def _scope_signals(text: str) -> set[str]:
    """Extract configured program/entity signals from text."""
    value = _string(
        text
    )

    if not value:
        return set()

    return {
        str(item).casefold()
        for item in (
            set(
                detect_programs(value)
            )
            |
            set(
                detect_entities(value)
            )
        )
        if str(item).strip()
    }


def _topic_signal(text: str) -> str | None:
    """Detect the generic query/source topic represented by text."""
    value = _string(
        text
    )

    if not value:
        return None

    return detect_query_scope(
        value
    )


def _source_topic(
    document: Any,
) -> str | None:
    """
    Prefer source/path topic information, with content as a fallback.

    The source path is the stronger signal because document content may
    mention several topics.
    """
    source = get_source(
        document
    )

    source_topic = _topic_signal(
        source
    )

    if source_topic:
        return source_topic

    return _topic_signal(
        getattr(
            document,
            "page_content",
            "",
        )
    )


def select_answer_evidence(
    *,
    query: str,
    documents: Sequence[Any],
    max_documents: int = 5,
) -> list[Any]:
    """
    Select final evidence for answer generation.

    Broad/non-scoped queries remain unchanged.

    For scoped queries:
        entity/program match is mandatory;
        topic match is mandatory when the query has a detectable topic and
        the candidate exposes a detectable topic.

    If no sufficiently scoped candidate exists, return the pre-existing
    evidence unchanged rather than guessing.
    """
    if max_documents <= 0:
        return []

    docs = list(
        documents or []
    )

    if not docs:
        return []

    query = _string(
        query
    )

    query_signals = _scope_signals(
        query
    )

    # Do not narrow broad/non-entity queries.
    if not query_signals:
        return docs[
            :max_documents
        ]

    query_topic = _topic_signal(
        query
    )

    matching: list[Any] = []

    for document in docs:
        source_signals = _scope_signals(
            get_source(
                document
            )
        )

        # Program/entity scope is the first gate.
        if not (
            query_signals
            &
            source_signals
        ):
            continue

        source_topic = _source_topic(
            document
        )

        # If both query and source expose a topic, they must agree.
        # This prevents M.Sc. program-overview evidence from being promoted
        # for an M.Sc. admission question.
        if (
            query_topic
            and source_topic
            and source_topic != query_topic
        ):
            continue

        matching.append(
            document
        )

    if matching:
        return matching[
            :max_documents
        ]

    # No sufficiently scoped evidence exists. Preserve the existing evidence
    # rather than inventing a relationship.
    return docs[
        :max_documents
    ]
