"""
Retrieval V2 — Evidence Unit Construction

Purpose
-------
Convert retrieved documents into small, local evidence units before
relevance filtering.

Core invariant
--------------
A retrieved document is NOT automatically treated as one evidence unit.

Relevance must be evaluated on local text so that unrelated information
far apart in the same document cannot be joined together.

Example:

    M.Tech programs are listed here.

    ... many unrelated sections ...

    Ph.D. eligibility requirements are listed separately.

These statements must not become a single piece of evidence merely because
they occur in the same retrieved document.
"""

from __future__ import annotations

import re

from .models import (
    EvidenceUnitCandidate,
    RankedCandidate,
)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_PATTERN = re.compile(
    r"""
    (?<=[.!?])\s+
    |
    \n+
    """,
    re.VERBOSE,
)


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentences while preserving character offsets.

    Returns
    -------
    list[tuple[str, int, int]]
        (sentence_text, start_char, end_char)
    """

    text = text.strip()

    if not text:
        return []

    units: list[tuple[str, int, int]] = []

    cursor = 0

    for match in _SENTENCE_PATTERN.finditer(text):
        start = cursor
        end = match.start()

        sentence = text[start:end].strip()

        if sentence:
            actual_start = start

            while actual_start < end and text[actual_start].isspace():
                actual_start += 1

            actual_end = end

            while actual_end > actual_start and text[actual_end - 1].isspace():
                actual_end -= 1

            units.append(
                (
                    text[actual_start:actual_end],
                    actual_start,
                    actual_end,
                )
            )

        cursor = match.end()

    # Final sentence.
    if cursor < len(text):
        sentence = text[cursor:].strip()

        if sentence:
            actual_start = cursor

            while (
                actual_start < len(text)
                and text[actual_start].isspace()
            ):
                actual_start += 1

            actual_end = len(text)

            while (
                actual_end > actual_start
                and text[actual_end - 1].isspace()
            ):
                actual_end -= 1

            units.append(
                (
                    text[actual_start:actual_end],
                    actual_start,
                    actual_end,
                )
            )

    return units


# ---------------------------------------------------------------------------
# Evidence-unit construction
# ---------------------------------------------------------------------------


def build_evidence_units(
    candidate: RankedCandidate,
    *,
    max_sentences: int = 3,
    sentence_stride: int = 1,
) -> list[EvidenceUnitCandidate]:
    """
    Convert one ranked document into local evidence units.

    Parameters
    ----------
    candidate:
        Ranked retrieval candidate.

    max_sentences:
        Maximum number of sentences in one evidence unit.

    sentence_stride:
        Number of sentences by which the window advances.

    Important design decision
    -------------------------
    Evidence units are deliberately local.

    We do NOT create one giant unit spanning the entire document.

    We also avoid a very large overlap because excessive overlap can
    accidentally combine unrelated claims that happen to be near each
    other.
    """

    if max_sentences <= 0:
        raise ValueError("max_sentences must be greater than zero")

    if sentence_stride <= 0:
        raise ValueError("sentence_stride must be greater than zero")

    document = candidate.document
    text = document.text.strip()

    if not text:
        return []

    sentences = _split_sentences(text)

    if not sentences:
        return []

    # ------------------------------------------------------------------
    # Very short documents
    # ------------------------------------------------------------------

    if len(sentences) <= max_sentences:
        combined_text = " ".join(
            sentence[0]
            for sentence in sentences
        ).strip()

        return [
            EvidenceUnitCandidate(
                document_id=document.document_id,
                text=combined_text,
                source=document.source,
                title=document.title,
                metadata=document.metadata,
                unit_index=0,
                start_char=sentences[0][1],
                end_char=sentences[-1][2],
                semantic_score=candidate.semantic_score,
                lexical_score=candidate.lexical_score,
            )
        ]

    # ------------------------------------------------------------------
    # Sliding local windows
    # ------------------------------------------------------------------

    evidence_units: list[EvidenceUnitCandidate] = []

    unit_index = 0

    for start_index in range(
        0,
        len(sentences),
        sentence_stride,
    ):
        end_index = min(
            start_index + max_sentences,
            len(sentences),
        )

        selected = sentences[start_index:end_index]

        if not selected:
            continue

        start_char = selected[0][1]
        end_char = selected[-1][2]

        unit_text = " ".join(
            sentence[0]
            for sentence in selected
        ).strip()

        if not unit_text:
            continue

        evidence_units.append(
            EvidenceUnitCandidate(
                document_id=document.document_id,
                text=unit_text,
                source=document.source,
                title=document.title,
                metadata=document.metadata,
                unit_index=unit_index,
                start_char=start_char,
                end_char=end_char,
                semantic_score=candidate.semantic_score,
                lexical_score=candidate.lexical_score,
            )
        )

        unit_index += 1

        # Once the final sentence has been included, stop.
        if end_index >= len(sentences):
            break

    return evidence_units


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def build_all_evidence_units(
    candidates: list[RankedCandidate] | tuple[RankedCandidate, ...],
    *,
    max_sentences: int = 3,
    sentence_stride: int = 1,
) -> list[EvidenceUnitCandidate]:
    """
    Expand all ranked retrieval candidates into local evidence units.

    Candidate ordering is preserved.
    """

    evidence_units: list[EvidenceUnitCandidate] = []

    for candidate in candidates:
        evidence_units.extend(
            build_evidence_units(
                candidate,
                max_sentences=max_sentences,
                sentence_stride=sentence_stride,
            )
        )

    return evidence_units