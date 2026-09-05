"""
IIT Jodhpur V1 — Local Context Expansion

Purpose
-------
Recover a small amount of useful nearby context for retrieved
documents when the original chunk appears incomplete.

Design principles
-----------------
- Deterministic only.
- No LLM calls.
- No retrieval redesign.
- No college-specific filenames, headings, or chunk numbers.
- Never cross source boundaries.
- Preserve the original retrieved chunk text.
- Consider at most one previous and one next same-source chunk.
- Reject obvious navigation / metadata noise.
- Detect generic structural section boundaries.
- Preserve useful text before a new structural section begins.
- Use continuation-aware scoring so legitimate chunk continuations
  are not rejected merely because lexical overlap is low.
"""

import re
from typing import List, Optional

from langchain_core.documents import Document

from backend.retriever import (
    chunks,
    get_source,
    normalize_text,
    detect_programs,
    detect_topics,
    detect_entities,
)


# =========================================================
# Configuration
# =========================================================

MAX_PREVIOUS_CONTEXT = 1
MAX_NEXT_CONTEXT = 1

MIN_CONTEXT_SCORE = 0.18
MIN_CONTENT_TOKENS = 8
MIN_BOUNDARY_PREFIX_CHARS = 80

TAIL_TOKENS = 40
HEAD_TOKENS = 40


# =========================================================
# Noise Detection
# =========================================================

NOISE_MARKERS = {
    "original source urls",
    "retrieval representation",
    "chunk id",
    "rrf score",
    "source path",
    "retrieval rank",
    "command 6",
}


def is_context_noise(
    document: Document,
) -> bool:
    """
    Detect obvious navigation, metadata, or retrieval-noise chunks.

    Normalization is used only for detection.
    Original evidence text is never modified here.
    """

    normalized = normalize_text(
        document.page_content
    )

    if not normalized:
        return True

    for marker in NOISE_MARKERS:
        if marker in normalized:
            return True

    if normalized.count("http") >= 3:
        return True

    tokens = [
        token
        for token in normalized.split()
        if len(token) > 2
    ]

    if len(tokens) < MIN_CONTENT_TOKENS:
        return True

    return False


# =========================================================
# Token Helpers
# =========================================================

def _tokens(
    text: str,
):
    """
    Return normalized tokens for matching only.
    """

    return [
        token
        for token in normalize_text(
            text
        ).split()
        if len(token) > 2
    ]


def _token_set(
    text: str,
):
    return set(
        _tokens(text)
    )


# =========================================================
# Structural Boundary Detection
# =========================================================

def _is_numbered_section_start(
    text: str,
) -> bool:
    """
    Detect hierarchical numbered headings such as:

        2. Admission
        20.2 Admission
        20.2.9 Admission to Ph.D.

    This is structural rather than institution-specific.
    """

    return bool(
        re.match(
            r"^\d+(?:\.\d+)+\s+",
            text,
        )
    )


def _is_markdown_heading(
    text: str,
) -> bool:
    """
    Detect markdown-style headings.
    """

    return bool(
        re.match(
            r"^#{1,6}\s+",
            text,
        )
    )


def _is_named_structural_heading(
    text: str,
) -> bool:
    """
    Detect generic explicit structural headings.
    """

    heading_starts = (
        "section ",
        "chapter ",
        "appendix ",
        "part ",
        "unit ",
        "module ",
    )

    lowered = text.lower()

    return any(
        lowered.startswith(
            prefix
        )
        for prefix in heading_starts
    )


def _is_all_caps_heading(
    text: str,
) -> bool:
    """
    Detect short all-caps heading-like chunks.
    """

    if len(text) < 8:
        return False

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if not letters:
        return False

    uppercase_letters = [
        char
        for char in letters
        if char.isupper()
    ]

    return (
        len(uppercase_letters)
        / len(letters)
        >= 0.90
        and len(text.split()) <= 12
    )


def _is_section_heading(
    text: str,
) -> bool:
    """
    Detect whether an entire neighboring chunk starts with a
    structural heading.
    """

    cleaned = re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()

    if not cleaned:
        return False

    prefix = cleaned[:300]

    return (
        _is_numbered_section_start(
            prefix
        )
        or
        _is_markdown_heading(
            prefix
        )
        or
        _is_named_structural_heading(
            prefix
        )
        or
        _is_all_caps_heading(
            prefix
        )
    )


# =========================================================
# Internal Boundary Detection
# =========================================================

def _find_internal_numbered_boundary(
    text: str,
) -> Optional[int]:
    """
    Find a hierarchical numbered section that begins later inside
    a chunk.

    Only the position is returned. The original text remains intact.
    """

    pattern = re.compile(
        r"(?<!\w)"
        r"\d+(?:\.\d+)+"
        r"\s+"
    )

    for match in pattern.finditer(
        text
    ):
        position = match.start()

        if position < MIN_BOUNDARY_PREFIX_CHARS:
            continue

        return position

    return None


def _find_internal_named_boundary(
    text: str,
) -> Optional[int]:
    """
    Find an explicit generic structural heading later in a chunk.
    """

    pattern = re.compile(
        r"(?<!\w)"
        r"(section|chapter|appendix|part|unit|module)"
        r"\s+[A-Za-z0-9]",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(
        text
    ):
        position = match.start()

        if position < MIN_BOUNDARY_PREFIX_CHARS:
            continue

        return position

    return None


def _find_internal_boundary(
    text: str,
) -> Optional[int]:
    """
    Find the earliest clear structural boundary.
    """

    positions = []

    numbered = _find_internal_numbered_boundary(
        text
    )

    if numbered is not None:
        positions.append(
            numbered
        )

    named = _find_internal_named_boundary(
        text
    )

    if named is not None:
        positions.append(
            named
        )

    if not positions:
        return None

    return min(
        positions
    )


# =========================================================
# Preserve Evidence Text
# =========================================================

def _clean_preserve_text(
    text: str,
) -> str:
    """
    Collapse whitespace while preserving factual content exactly
    enough for answer generation.

    Symbols such as %, ₹, dates, codes, URLs, and decimal values
    remain intact.
    """

    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()


def _extract_boundary_safe_content(
    text: str,
) -> Optional[str]:
    """
    Preserve useful content before a new structural section.

    Examples of information that must remain untouched:

        70%
        6.0/10
        ₹450/-
        four-year
        course codes
        URLs
    """

    cleaned = _clean_preserve_text(
        text
    )

    if not cleaned:
        return None

    # A neighbor that starts with a new section is not local
    # continuation of the anchor.
    if _is_section_heading(
        cleaned
    ):
        return None

    boundary_position = (
        _find_internal_boundary(
            cleaned
        )
    )

    if boundary_position is None:
        return cleaned

    prefix = cleaned[
        :boundary_position
    ].strip()

    if len(prefix) < MIN_BOUNDARY_PREFIX_CHARS:
        return None

    return prefix


# =========================================================
# Continuation Signals
# =========================================================

def _tail_head_overlap(
    anchor_text: str,
    neighbor_text: str,
) -> float:
    """
    Measure lexical continuity between the end of the anchor and
    the beginning of the neighboring content.

    This is important for chunks where the next chunk continues a
    sentence/list but does not repeat many words from the full anchor.
    """

    anchor_tokens = _tokens(
        anchor_text
    )

    neighbor_tokens = _tokens(
        neighbor_text
    )

    if not anchor_tokens or not neighbor_tokens:
        return 0.0

    anchor_tail = set(
        anchor_tokens[
            -TAIL_TOKENS:
        ]
    )

    neighbor_head = set(
        neighbor_tokens[
            :HEAD_TOKENS
        ]
    )

    if not anchor_tail or not neighbor_head:
        return 0.0

    return (
        len(
            anchor_tail
            & neighbor_head
        )
        / max(
            len(anchor_tail),
            1,
        )
    )


def _sentence_continuation_signal(
    anchor_text: str,
    neighbor_text: str,
) -> float:
    """
    Detect lightweight sentence/list continuation patterns.

    This is deliberately generic.
    """

    anchor_clean = (
        str(anchor_text or "")
        .strip()
    )

    neighbor_clean = (
        str(neighbor_text or "")
        .strip()
    )

    if not anchor_clean or not neighbor_clean:
        return 0.0

    score = 0.0

    # The anchor ending in a comma/colon/semicolon strongly suggests
    # structural continuation.
    if anchor_clean.endswith(
        (
            ",",
            ":",
            ";",
        )
    ):
        score += 0.12

    # Bullet/list continuation.
    if re.match(
        r"^(?:[-*•]|\(?[a-zA-Z0-9]+\))\s+",
        neighbor_clean,
    ):
        score += 0.08

    # Natural sentence continuation with a common grammatical start.
    continuation_starts = (
        "the applicant",
        "the applicants",
        "candidates",
        "candidate",
        "students",
        "student",
        "the department",
        "the institute",
        "the program",
        "the programme",
    )

    lowered = neighbor_clean.lower()

    if any(
        lowered.startswith(
            prefix
        )
        for prefix in continuation_starts
    ):
        score += 0.05

    return min(
        score,
        0.20,
    )


# =========================================================
# Neighbor Continuity Score
# =========================================================

def _neighbor_score(
    anchor: Document,
    neighbor_text: str,
) -> float:
    """
    Estimate whether neighboring content continues the anchor.

    Signals:

        1. tail/head lexical continuity
        2. broader lexical overlap
        3. program continuity
        4. topic continuity
        5. entity continuity
        6. sentence/list continuation
    """

    anchor_content = normalize_text(
        anchor.page_content
    )

    neighbor_content = normalize_text(
        neighbor_text
    )

    anchor_tokens = _token_set(
        anchor_content
    )

    neighbor_tokens = _token_set(
        neighbor_content
    )

    if not anchor_tokens or not neighbor_tokens:
        return 0.0

    score = 0.0

    # -----------------------------------------------------
    # 1. Tail/head continuity
    # -----------------------------------------------------

    score += (
        _tail_head_overlap(
            anchor.page_content,
            neighbor_text,
        )
        * 0.45
    )

    # -----------------------------------------------------
    # 2. Broader lexical overlap
    # -----------------------------------------------------

    lexical_overlap = (
        len(
            anchor_tokens
            & neighbor_tokens
        )
        / max(
            len(anchor_tokens),
            1,
        )
    )

    score += (
        lexical_overlap
        * 0.20
    )

    # -----------------------------------------------------
    # 3. Program continuity
    # -----------------------------------------------------

    anchor_programs = detect_programs(
        anchor_content
    )

    neighbor_programs = detect_programs(
        neighbor_content
    )

    if (
        anchor_programs
        & neighbor_programs
    ):
        score += 0.10

    # -----------------------------------------------------
    # 4. Topic continuity
    # -----------------------------------------------------

    anchor_topics = detect_topics(
        anchor_content
    )

    neighbor_topics = detect_topics(
        neighbor_content
    )

    if (
        anchor_topics
        & neighbor_topics
    ):
        score += 0.10

    # -----------------------------------------------------
    # 5. Entity continuity
    # -----------------------------------------------------

    anchor_entities = detect_entities(
        anchor_content
    )

    neighbor_entities = detect_entities(
        neighbor_content
    )

    if (
        anchor_entities
        & neighbor_entities
    ):
        score += 0.05

    # -----------------------------------------------------
    # 6. Sentence / list continuation
    # -----------------------------------------------------

    score += (
        _sentence_continuation_signal(
            anchor.page_content,
            neighbor_text,
        )
    )

    return min(
        score,
        1.0,
    )


# =========================================================
# Canonical Chunk Matching
# =========================================================

def _find_chunk_index(
    document: Document,
):
    """
    Locate a retrieved document in the canonical ingestion sequence.

    Chroma may reconstruct Document objects, so object identity is
    not sufficient.
    """

    target_source = get_source(
        document
    )

    target_content = normalize_text(
        document.page_content
    )

    if not target_content:
        return None

    target_tokens = _token_set(
        target_content
    )

    best_index = None
    best_overlap = 0.0

    for index, candidate in enumerate(
        chunks
    ):

        if (
            get_source(candidate)
            != target_source
        ):
            continue

        candidate_content = normalize_text(
            candidate.page_content
        )

        if not candidate_content:
            continue

        if candidate_content == target_content:
            return index

        candidate_tokens = _token_set(
            candidate_content
        )

        if not candidate_tokens:
            continue

        overlap = (
            len(
                target_tokens
                & candidate_tokens
            )
            / max(
                len(target_tokens),
                1,
            )
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    return best_index


# =========================================================
# Candidate Preparation
# =========================================================

def _prepare_neighbor(
    anchor: Document,
    candidate: Document,
):
    """
    Return safe neighboring content when it is sufficiently
    related to the anchor.
    """

    if (
        get_source(anchor)
        != get_source(candidate)
    ):
        return None

    safe_content = (
        _extract_boundary_safe_content(
            candidate.page_content
        )
    )

    if not safe_content:
        return None

    score = _neighbor_score(
        anchor,
        safe_content,
    )

    if score < MIN_CONTEXT_SCORE:
        return None

    return safe_content


# =========================================================
# Expand One Document
# =========================================================

def expand_document_context(
    document: Document,
) -> List[Document]:
    """
    Expand one retrieved document with at most one previous and one
    next same-source neighbor.

    The original retrieved document remains unchanged.
    """

    index = _find_chunk_index(
        document
    )

    if index is None:
        return [
            document
        ]

    selected = []

    # -----------------------------------------------------
    # Previous neighbor
    # -----------------------------------------------------

    if (
        index
        >= MAX_PREVIOUS_CONTEXT
    ):

        previous = chunks[
            index - 1
        ]

        previous_content = (
            _prepare_neighbor(
                document,
                previous,
            )
        )

        if previous_content:

            selected.append(
                Document(
                    page_content=previous_content,
                    metadata=dict(
                        previous.metadata
                    ),
                )
            )

    # -----------------------------------------------------
    # Anchor
    # -----------------------------------------------------

    selected.append(
        document
    )

    # -----------------------------------------------------
    # Next neighbor
    # -----------------------------------------------------

    if (
        index + MAX_NEXT_CONTEXT
        < len(chunks)
    ):

        following = chunks[
            index + 1
        ]

        following_content = (
            _prepare_neighbor(
                document,
                following,
            )
        )

        if following_content:

            selected.append(
                Document(
                    page_content=following_content,
                    metadata=dict(
                        following.metadata
                    ),
                )
            )

    return selected


# =========================================================
# Expand Retrieved Documents
# =========================================================

def expand_local_context(
    documents: List[Document],
) -> List[Document]:
    """
    Expand retrieved documents conservatively.

    Duplicates are removed while preserving order.
    """

    if not documents:
        return []

    expanded = []
    seen = set()

    for document in documents:

        candidates = (
            expand_document_context(
                document
            )
        )

        for candidate in candidates:

            source = get_source(
                candidate
            )

            content = normalize_text(
                candidate.page_content
            )

            key = (
                source,
                content,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            expanded.append(
                candidate
            )

    return expanded