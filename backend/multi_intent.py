from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from backend.retriever import (
    detect_entities,
    detect_topics,
    normalize_text,
)


# =========================================================
# Configuration
# =========================================================

MULTI_INTENT_CONNECTORS = (
    " and ",
    " also ",
    " as well as ",
)

QUESTION_PREFIXES = (
    "what ",
    "which ",
    "who ",
    "where ",
    "when ",
    "why ",
    "how ",
    "is ",
    "are ",
    "can ",
    "does ",
    "do ",
    "will ",
    "would ",
    "could ",
)

# These coordinated expressions represent one semantic request.
SINGLE_INTENT_PHRASES = (
    "fees and charges",
    "costs and charges",
    "research themes and areas",
    "research areas and themes",
    "rules and regulations",
    "facilities and amenities",
    "programs and programmes",
    "programmes and programs",
    "eligibility and admission requirements",
    "admission requirements and eligibility",
)

# Broad request-head terms used only to recognize concise independent
# request fragments. These are not institution-specific.
REQUEST_HEADS = {
    "fee",
    "fees",
    "charge",
    "charges",
    "cost",
    "costs",
    "eligibility",
    "requirement",
    "requirements",
    "admission",
    "admissions",
    "program",
    "programs",
    "programme",
    "programmes",
    "degree",
    "degrees",
    "research",
    "area",
    "areas",
    "theme",
    "themes",
    "facility",
    "facilities",
    "amenity",
    "amenities",
    "department",
    "departments",
    "school",
    "schools",
    "centre",
    "centres",
    "center",
    "centers",
    "hostel",
    "hostels",
    "accommodation",
    "placement",
    "placements",
    "scholarship",
    "scholarships",
    "faculty",
    "professor",
    "professors",
    "course",
    "courses",
    "campus",
    "rules",
    "rule",
    "policy",
    "policies",
    "infrastructure",
    "facilities",
}

ANAPHORIC_TERMS = (
    "there",
    "here",
    "that department",
    "this department",
    "that school",
    "this school",
    "that program",
    "this program",
    "that programme",
    "this programme",
    "that center",
    "this center",
    "that centre",
    "this centre",
)

MIN_CLAUSE_LENGTH = 4


# =========================================================
# Data model
# =========================================================

@dataclass(frozen=True)
class IntentUnit:
    """
    One independently answerable user request.
    """

    question: str
    topics: frozenset[str]
    entities: frozenset[str]


# =========================================================
# Basic text helpers
# =========================================================

def _normalized(
    text: str,
) -> str:
    """
    Normalize whitespace/text using the project's existing normalizer.
    """

    return normalize_text(
        text or ""
    ).strip()


def _clean_clause(
    clause: str,
) -> str:
    """
    Normalize whitespace and ensure a question mark is present.
    """

    clause = str(
        clause or ""
    ).strip()

    clause = re.sub(
        r"\s+",
        " ",
        clause,
    )

    clause = clause.strip(
        " ,;:-"
    )

    if not clause:
        return ""

    if clause[-1] not in ".?!":
        clause += "?"

    return clause


def _has_question_shape(
    clause: str,
) -> bool:
    """
    Detect an explicit interrogative structure.

    A conversational/personal preamble may appear before the actual
    question. Use the raw clause for sentence-boundary detection so
    punctuation inside program names such as ``Ph.D.`` does not prevent
    recognizing the following question.
    """

    raw = str(
        clause or ""
    ).strip()

    normalized = _normalized(
        raw
    )

    if not normalized:
        return False

    if normalized.endswith("?"):
        return True

    if normalized.startswith(
        QUESTION_PREFIXES
    ):
        return True

    # A conversational preamble may be followed by the actual question.
    # Check each sentence-like segment using the original punctuation.
    segments = re.split(
        r"(?<=[.!?])\s+",
        raw,
    )

    for segment in segments[1:]:
        segment_normalized = _normalized(
            segment
        )

        if segment_normalized.startswith(
            QUESTION_PREFIXES
        ):
            return True

    # Robust fallback for cases where an upstream normalizer has removed
    # punctuation from an embedded question boundary.
    question_boundary_pattern = re.compile(
        r"(?:^|[.!?]\s+)(?:"
        + "|".join(
            re.escape(prefix)
            for prefix in QUESTION_PREFIXES
        )
        + r")",
        flags=re.IGNORECASE,
    )

    return bool(
        question_boundary_pattern.search(
            raw
        )
    )


def _contains_question_prefix(
    clause: str,
) -> bool:
    """
    Detect whether the clause begins with a question word/auxiliary.
    """

    normalized = _normalized(
        clause
    )

    return normalized.startswith(
        QUESTION_PREFIXES
    )


def _extract_embedded_question(
    clause: str,
) -> str:
    """
    Remove conversational/personal preamble before an embedded question.

    Example:
        "I have 74% in B.Tech. Can I apply for Ph.D.?"

    becomes:
        "Can I apply for Ph.D.?"

    This is used only for intent decomposition. The original user
    situation remains available to the Phase-5 situation pipeline.
    """

    raw = str(
        clause or ""
    ).strip()

    if not raw:
        return ""

    segments = [
        segment.strip()
        for segment in re.split(
            r"(?<=[.!?])\s+",
            raw,
        )
        if segment.strip()
    ]

    if len(segments) <= 1:
        return raw

    for index, segment in enumerate(
        segments
    ):
        normalized = _normalized(
            segment
        )

        if normalized.startswith(
            QUESTION_PREFIXES
        ):
            return " ".join(
                segments[index:]
            )

    return raw


def _request_heads(
    clause: str,
) -> set[str]:
    """
    Find broad request-head words in a clause.
    """

    normalized = _normalized(
        clause
    )

    tokens = set(
        re.findall(
            r"\b[a-z0-9]+\b",
            normalized,
        )
    )

    return (
        tokens
        &
        REQUEST_HEADS
    )


def _semantic_signal_count(
    clause: str,
) -> int:
    """
    Use the existing project's semantic detectors as secondary signals.
    """

    return (
        len(
            detect_topics(
                clause
            )
        )
        +
        len(
            detect_entities(
                clause
            )
        )
    )


def _contains_anaphor(
    clause: str,
) -> bool:
    """
    Detect simple references whose meaning depends on earlier context.
    """

    normalized = _normalized(
        clause
    )

    return any(
        term in normalized
        for term in ANAPHORIC_TERMS
    )


def _looks_like_single_intent_phrase(
    question: str,
) -> bool:
    """
    Protect common coordinated phrases.
    """

    normalized = _normalized(
        question
    )

    return any(
        phrase in normalized
        for phrase in SINGLE_INTENT_PHRASES
    )


# =========================================================
# Scope extraction
# =========================================================

def _extract_scope_phrase(
    clause: str,
) -> str | None:
    """
    Extract a high-confidence trailing scope phrase.

    Examples:
        "... in Electrical Engineering"
        "... at the School of Mathematics"
        "... through the School of Artificial Intelligence and Data Science"
        "... from the hostel"
    """

    raw = str(
        clause or ""
    ).strip()

    patterns = (
        r"(\bin\s+.+?)$",
        r"(\bat\s+.+?)$",
        r"(\bthrough\s+.+?)$",
        r"(\bwithin\s+.+?)$",
        r"(\bfrom\s+.+?)$",
        r"(\bfor\s+.+?)$",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            raw,
            flags=re.IGNORECASE,
        )

        if match:
            value = match.group(
                1
            ).strip(
                " ."
            )

            if value:
                return value

    return None


# =========================================================
# Anaphora resolution
# =========================================================

def _resolve_anaphora(
    first_clause: str,
    second_clause: str,
) -> str:
    """
    Resolve a simple second-clause reference using the explicit scope of
    the first clause.

    Only high-confidence references are replaced.
    """

    if not _contains_anaphor(
        second_clause
    ):
        return second_clause

    scope = _extract_scope_phrase(
        first_clause
    )

    if not scope:
        return second_clause

    resolved = second_clause

    # Prefer longer phrases before shorter terms.
    replacement_terms = (
        "that department",
        "this department",
        "that school",
        "this school",
        "that program",
        "this program",
        "that programme",
        "this programme",
        "that center",
        "this center",
        "that centre",
        "this centre",
        "there",
        "here",
    )

    for term in replacement_terms:

        pattern = re.compile(
            rf"\b{re.escape(term)}\b",
            flags=re.IGNORECASE,
        )

        if pattern.search(
            resolved
        ):
            resolved = pattern.sub(
                scope,
                resolved,
                count=1,
            )
            break

    return resolved


# =========================================================
# Candidate generation
# =========================================================

def _candidate_splits(
    question: str,
):
    """
    Generate every possible connector boundary.

    Each candidate carries its connector position so candidates can be
    ranked by confidence later.
    """

    raw = str(
        question or ""
    ).strip()

    for connector in MULTI_INTENT_CONNECTORS:

        pattern = re.compile(
            re.escape(
                connector
            ),
            flags=re.IGNORECASE,
        )

        for match in pattern.finditer(
            raw
        ):

            left = raw[
                :match.start()
            ]

            right = raw[
                match.end():
            ]

            yield {
                "left": left,
                "right": right,
                "position": match.start(),
                "connector": connector.strip(),
            }


# =========================================================
# Boundary scoring
# =========================================================

def _boundary_score(
    question: str,
    left: str,
    right: str,
) -> int:
    """
    Score a candidate intent boundary.

    Higher confidence:
        - second clause explicitly starts as a question
        - first clause is already question-shaped
        - both sides have independent request heads
        - complete input is a question

    Lower confidence:
        - arbitrary conjunction in a declarative sentence
    """

    raw = str(
        question or ""
    ).strip()

    clean_left = _clean_clause(
        left
    )

    clean_right = _clean_clause(
        right
    )

    normalized_raw = _normalized(
        raw
    )

    score = 0

    left_question = _has_question_shape(
        left
    )

    right_question = _has_question_shape(
        right
    )

    left_prefix = _contains_question_prefix(
        left
    )

    right_prefix = _contains_question_prefix(
        right
    )

    left_heads = _request_heads(
        left
    )

    right_heads = _request_heads(
        right
    )

    left_signals = _semantic_signal_count(
        left
    )

    right_signals = _semantic_signal_count(
        right
    )

    # -----------------------------------------------------
    # Strongest evidence: explicit second question.
    # -----------------------------------------------------

    if right_prefix:
        score += 8

    if right_question:
        score += 5

    if left_question:
        score += 4

    # -----------------------------------------------------
    # Independent request heads.
    # -----------------------------------------------------

    if left_heads:
        score += 2

    if right_heads:
        score += 3

    if (
        left_heads
        and
        right_heads
    ):
        score += 5

    # -----------------------------------------------------
    # Existing semantic detectors provide secondary evidence.
    # -----------------------------------------------------

    if left_signals:
        score += 1

    if right_signals:
        score += 1

    # -----------------------------------------------------
    # Entire user input is explicitly a question.
    # Useful for concise forms such as:
    #
    #     Hostel fees and Ph.D. eligibility?
    # -----------------------------------------------------

    if normalized_raw.endswith(
        "?"
    ):
        score += 2

    # -----------------------------------------------------
    # Anaphoric second clause is a strong indication of a new intent
    # when the first clause contains explicit scope.
    # -----------------------------------------------------

    if (
        _contains_anaphor(
            right
        )
        and
        _extract_scope_phrase(
            left
        )
    ):
        score += 5

    return score


# =========================================================
# Boundary validation
# =========================================================

def _is_valid_boundary(
    question: str,
    left: str,
    right: str,
) -> bool:
    """
    Determine whether the candidate represents two independent
    requests.

    Critical rule:
        Ordinary statements must not be split merely because they contain
        "and".
    """

    left = _clean_clause(
        left
    )

    right = _clean_clause(
        right
    )

    if (
        len(left)
        < MIN_CLAUSE_LENGTH
        or
        len(right)
        < MIN_CLAUSE_LENGTH
    ):
        return False

    combined = (
        left.rstrip(
            "?!."
        )
        + " and "
        + right.lstrip(
            " ?!."
        )
    )

    if _looks_like_single_intent_phrase(
        combined
    ):
        return False

    left_heads = _request_heads(
        left
    )

    right_heads = _request_heads(
        right
    )

    left_question = _has_question_shape(
        left
    )

    right_question = _has_question_shape(
        right
    )

    right_prefix = _contains_question_prefix(
        right
    )

    # -----------------------------------------------------
    # Explicit question + explicit question.
    # -----------------------------------------------------

    if (
        left_question
        and
        right_question
        and
        (
            bool(left_heads)
            or
            _semantic_signal_count(left) >= 1
        )
        and
        (
            bool(right_heads)
            or
            _semantic_signal_count(right) >= 1
        )
    ):
        return True

    # -----------------------------------------------------
    # Question + second question phrase.
    #
    # Example:
    # "... Electrical Engineering and what programs..."
    # -----------------------------------------------------

    if (
        left_question
        and
        right_prefix
    ):
        return True

    # -----------------------------------------------------
    # Anaphoric second question.
    # -----------------------------------------------------

    if (
        left_question
        and
        _contains_anaphor(
            right
        )
        and
        _extract_scope_phrase(
            left
        )
    ):
        return True

    # -----------------------------------------------------
    # Concise noun-phrase multi-intent question.
    #
    # Require the COMPLETE USER INPUT to be a question.
    # This prevents:
    #
    #     Tell me about the hostel and campus.
    #
    # from being split.
    # -----------------------------------------------------

    if (
        str(question or "").strip().endswith("?")
        and
        left_heads
        and
        right_heads
    ):
        return True

    return False


# =========================================================
# Explicit multi-question list handling
# =========================================================

def _strip_leading_conjunction(
    clause: str,
) -> str:
    """
    Remove a leading conjunction from an independent request.
    """

    return re.sub(
        r"^(?:and|also|as well as)\s+",
        "",
        str(clause or "").strip(),
        flags=re.IGNORECASE,
    )


def _split_explicit_question_list(
    question: str,
) -> list[str] | None:
    """
    Recognize a 3+ request list only when every comma-separated part is
    itself question-shaped.

    This is intentionally narrower than generic comma splitting.
    """

    raw = str(
        question or ""
    ).strip()

    if not raw.endswith("?"):
        return None

    parts = [
        _strip_leading_conjunction(
            part
        )
        for part in raw[
            :-1
        ].split(",")
        if part.strip()
    ]

    if len(parts) < 3:
        return None

    if not all(
        _has_question_shape(
            part
        )
        for part in parts
    ):
        return None

    return [
        _clean_clause(
            part
        )
        for part in parts
    ]


def _looks_like_compact_noun_list(
    question: str,
) -> bool:
    """
    Detect a single question that merely enumerates attributes.

    Example:
        "What are the hostel facilities, fees, and rules?"

    This prevents a broad request from becoming:
        facilities?
        fees?
        rules?
    """

    raw = str(
        question or ""
    ).strip()

    if (
        not raw.endswith("?")
        or ","
        not in raw
    ):
        return False

    parts = [
        part.strip()
        for part in raw[
            :-1
        ].split(",")
        if part.strip()
    ]

    if len(parts) < 3:
        return False

    # If each part starts a fresh interrogative, it may be a true
    # multi-question list. Otherwise treat it as one broad request.
    return not any(
        _contains_question_prefix(
            part
        )
        for part in parts[1:]
    )


def _make_intent_unit(
    question: str,
) -> IntentUnit:
    """
    Build an IntentUnit using the existing project's detectors.
    """

    cleaned = _clean_clause(
        question
    )

    return IntentUnit(
        question=cleaned,
        topics=frozenset(
            detect_topics(
                cleaned
            )
        ),
        entities=frozenset(
            detect_entities(
                cleaned
            )
        ),
    )


# =========================================================
# Public decomposition
# =========================================================

def decompose_multi_intent(
    question: str,
) -> List[IntentUnit]:
    """
    Decompose a user question into independent intent units.

    Conservative strategy:
        1. Protect known single-intent phrases.
        2. Protect compact broad comma-lists.
        3. Recognize explicit 3+ question lists.
        4. Generate connector boundaries.
        5. Reject unsafe boundaries.
        6. Select the highest-confidence boundary.
        7. Resolve simple anaphora.
        8. Return one unit if no safe split exists.
    """

    raw = str(
        question or ""
    ).strip()

    if not raw:
        return []

    # -----------------------------------------------------
    # Conservative protection for broad comma-separated noun lists.
    # -----------------------------------------------------

    if _looks_like_compact_noun_list(
        raw
    ):
        return [
            _make_intent_unit(
                raw
            )
        ]

    # -----------------------------------------------------
    # Protect known coordinated one-intent expressions.
    # -----------------------------------------------------

    if _looks_like_single_intent_phrase(
        raw
    ):

        return [
            _make_intent_unit(
                raw
            )
        ]

    # -----------------------------------------------------
    # Explicit three-or-more question list.
    # -----------------------------------------------------

    explicit_list = _split_explicit_question_list(
        raw
    )

    if explicit_list is not None:
        return [
            _make_intent_unit(
                part
            )
            for part in explicit_list
        ]

    # -----------------------------------------------------
    # Evaluate all possible connector boundaries.
    # -----------------------------------------------------

    candidates = []

    for candidate in _candidate_splits(
        raw
    ):

        left = candidate[
            "left"
        ]

        right = candidate[
            "right"
        ]

        if not _is_valid_boundary(
            raw,
            left,
            right,
        ):
            continue

        score = _boundary_score(
            raw,
            left,
            right,
        )

        candidates.append(
            (
                score,
                candidate[
                    "position"
                ],
                left,
                right,
            )
        )

    # -----------------------------------------------------
    # Select the strongest boundary.
    #
    # Tie-breaking prefers the later boundary. This matters for:
    #
    #   School of Artificial Intelligence and Data Science
    #   and what research...
    #
    # where the first "and" belongs inside the school name.
    # -----------------------------------------------------

    if candidates:

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        (
            _score,
            _position,
            left,
            right,
        ) = candidates[0]

        left = _clean_clause(
            left
        )

        # Drop conversational preamble from the decomposed intent.
        # User-specific facts remain available through Phase 5 situation
        # understanding.
        embedded_left = _extract_embedded_question(
            left
        )

        if embedded_left != left:
            left = _clean_clause(
                embedded_left
            )

        right = _clean_clause(
            right
        )

        right = _resolve_anaphora(
            left,
            right,
        )

        right = _clean_clause(
            right
        )

        return [
            _make_intent_unit(
                left
            ),
            _make_intent_unit(
                right
            ),
        ]

    # -----------------------------------------------------
    # No safe decomposition.
    # -----------------------------------------------------

    return [
        _make_intent_unit(
            raw
        )
    ]


# =========================================================
# Convenience helpers
# =========================================================

def is_multi_intent(
    question: str,
) -> bool:
    """
    Return True only when safe decomposition yields multiple intents.
    """

    return (
        len(
            decompose_multi_intent(
                question
            )
        )
        > 1
    )


def intent_questions(
    question: str,
) -> List[str]:
    """
    Return only the decomposed question strings.
    """

    return [
        unit.question
        for unit in decompose_multi_intent(
            question
        )
    ]