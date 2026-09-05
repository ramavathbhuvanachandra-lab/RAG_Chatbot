"""
IIT Jodhpur V1 — Evidence Sufficiency

Purpose
-------
Determine whether retrieved evidence is sufficiently aligned with the
user's question before answer generation.

Design principles
-----------------
- Deterministic and lightweight.
- No additional LLM call.
- Generic across institutions and corpora.
- Question-type aware.
- Claim-scope aware.
- Numeric/unit aware.
- Temporal aware.
- Conservative under incomplete evidence.
- A single strong, directly aligned document may be sufficient even for
  a list-style question when that document clearly contains the requested
  information.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.retriever import (
    normalize_text,
    detect_programs,
    detect_topics,
    detect_entities,
)

from backend.evidence_coverage import (
    detect_question_type,
)

from backend.claim_scope import (
    ADMISSION_SCOPE,
    detect_claim_scopes,
    detect_admission_modes,
    has_scope_conflict,
    scopes_compatible,
    admission_modes_compatible,
)

from backend.evidence_temporal_numeric import (
    quantitative_temporal_compatible,
)


# =========================================================
# Configuration
# =========================================================

LIST_MIN_DOCUMENT_SCORE = 0.08
LIST_MIN_SUPPORTING_DOCUMENTS = 2
LIST_MIN_AGGREGATE_SCORE = 0.30

# A list question can be fully supported by one highly aligned document
# when that document directly answers the requested scope.
LIST_STRONG_SINGLE_DOCUMENT_SCORE = 0.45

REQUIREMENTS_MIN_DOCUMENT_SCORE = 0.30
QUANTITATIVE_MIN_DOCUMENT_SCORE = 0.30
DESCRIPTIVE_MIN_DOCUMENT_SCORE = 0.30

DIRECT_REQUIREMENT_SUPPORT = 0.35


# =========================================================
# Quantitative evidence
# =========================================================

QUANTITATIVE_PATTERNS = [
    r"₹\s*\d",
    r"\$\s*\d",
    r"€\s*\d",
    r"£\s*\d",
    r"\b\d+(?:\.\d+)?\s*%",
    r"\b\d+(?:\.\d+)?\s*(?:cgpa|cpi)\b",
    r"\b\d+(?:\.\d+)?\s*(?:days?|months?|years?)\b",
    r"\b\d+(?:\.\d+)?\b",
]


def _contains_quantitative_evidence(
    text: str,
) -> bool:

    normalized = normalize_text(
        text
    )

    return any(
        re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )
        for pattern in QUANTITATIVE_PATTERNS
    )


# =========================================================
# Requirement evidence
# =========================================================

REQUIREMENT_MARKERS = (
    "eligibility",
    "eligible",
    "qualification",
    "qualifications",
    "requirement",
    "requirements",
    "criteria",
    "degree",
    "marks",
    "percentage",
    "cgpa",
    "cpi",
    "gate",
    "experience",
    "admission",
)


def _contains_requirement_evidence(
    text: str,
) -> bool:

    normalized = normalize_text(
        text
    )

    return any(
        marker in normalized
        for marker in REQUIREMENT_MARKERS
    )


DIRECT_ADMISSION_PATTERNS = [
    r"\bmust have\b",
    r"\bminimum\b",
    r"\bminimum four year\b",
    r"\bfour year\b.*\bbachelor\b",
    r"\bbachelor\b.*\bdegree\b",
    r"\bmaster\b.*\bdegree\b",
    r"\bat least\b.*\bmarks\b",
    r"\bat least\b.*\b%",
    r"\bcgpa\b",
    r"\bcpi\b",
]


def _contains_direct_admission_evidence(
    text: str,
) -> bool:

    normalized = normalize_text(
        text
    )

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in DIRECT_ADMISSION_PATTERNS
    )


# =========================================================
# Claim-scope compatibility
# =========================================================

def _claim_scope_compatible(
    query: str,
    document,
) -> bool:

    document_text = document.page_content

    if has_scope_conflict(
        query,
        document_text,
    ):
        return False

    query_scopes = detect_claim_scopes(
        query
    )

    document_scopes = detect_claim_scopes(
        document_text
    )

    query_modes = detect_admission_modes(
        query
    )

    document_modes = detect_admission_modes(
        document_text
    )

    if query_scopes:

        if not scopes_compatible(
            query_scopes,
            document_scopes,
        ):

            if document_scopes:
                return False

    if not admission_modes_compatible(
        query_modes,
        document_modes,
    ):
        return False

    return True


# =========================================================
# Single-document support
# =========================================================

def _score_document_support(
    query: str,
    document,
) -> float:

    if not _claim_scope_compatible(
        query,
        document,
    ):
        return 0.0

    normalized_query = normalize_text(
        query
    )

    content = normalize_text(
        document.page_content
    )

    if not content:
        return 0.0

    query_tokens = {
        token
        for token in normalized_query.split()
        if len(token) > 2
    }

    content_tokens = {
        token
        for token in content.split()
        if len(token) > 2
    }

    score = 0.0

    # -----------------------------------------------------
    # Query/content overlap
    # -----------------------------------------------------

    if query_tokens:

        overlap = (
            len(
                query_tokens
                &
                content_tokens
            )
            /
            len(
                query_tokens
            )
        )

        score += (
            overlap
            * 0.50
        )

    # -----------------------------------------------------
    # Program alignment
    # -----------------------------------------------------

    query_programs = detect_programs(
        query
    )

    document_programs = detect_programs(
        content
    )

    if (
        query_programs
        and
        query_programs
        &
        document_programs
    ):
        score += 0.25

    # -----------------------------------------------------
    # Topic alignment
    # -----------------------------------------------------

    query_topics = detect_topics(
        query
    )

    document_topics = detect_topics(
        content
    )

    if (
        query_topics
        and
        query_topics
        &
        document_topics
    ):
        score += 0.15

    # -----------------------------------------------------
    # Entity alignment
    # -----------------------------------------------------

    query_entities = detect_entities(
        query
    )

    document_entities = detect_entities(
        content
    )

    if (
        query_entities
        and
        query_entities
        &
        document_entities
    ):
        score += 0.10

    # -----------------------------------------------------
    # Direct admission evidence
    # -----------------------------------------------------

    if (
        ADMISSION_SCOPE
        in
        detect_claim_scopes(
            query
        )
        and
        _contains_direct_admission_evidence(
            content
        )
    ):
        score += (
            DIRECT_REQUIREMENT_SUPPORT
        )

    return min(
        score,
        1.0,
    )


# =========================================================
# List assessment
# =========================================================

def _assess_list_support(
    query: str,
    documents: List[Any],
    scores: List[float],
) -> Dict[str, Any]:
    """
    Assess list-style questions.

    Normal behavior:
        Require multiple reasonably supporting documents so that broad
        enumeration questions do not become falsely "supported" from
        one weak fragment.

    Strong-document exception:
        A single highly aligned document is sufficient when its support
        score is high enough to indicate that it directly addresses the
        requested list/scope.

    This preserves conservative behavior for weak evidence while avoiding
    false "insufficient" results when one authoritative document already
    contains the complete answer.
    """

    supporting_scores = [
        score
        for document, score in zip(
            documents,
            scores,
        )
        if (
            score
            >= LIST_MIN_DOCUMENT_SCORE
            and
            _claim_scope_compatible(
                query,
                document,
            )
        )
    ]

    strongest_scores = sorted(
        supporting_scores,
        reverse=True,
    )[:5]

    aggregate_score = sum(
        strongest_scores
    )

    relevant_documents = len(
        supporting_scores
    )

    best_score = max(
        scores,
        default=0.0,
    )

    # -----------------------------------------------------
    # Standard multi-document support.
    # -----------------------------------------------------

    multi_document_support = (
        relevant_documents
        >= LIST_MIN_SUPPORTING_DOCUMENTS
        and
        aggregate_score
        >= LIST_MIN_AGGREGATE_SCORE
    )

    # -----------------------------------------------------
    # Strong single-document support.
    #
    # Example:
    #
    # "What research areas are available in Electrical Engineering?"
    #
    # Evidence:
    #
    # "The Department of Electrical Engineering research areas
    #  include MIMO communications, control systems, signal
    #  processing, VLSI, and cyber-physical systems."
    #
    # One document is clearly sufficient here.
    # -----------------------------------------------------

    strong_single_document_support = (
        relevant_documents >= 1
        and
        best_score
        >= LIST_STRONG_SINGLE_DOCUMENT_SCORE
    )

    supported = (
        multi_document_support
        or
        strong_single_document_support
    )

    return {
        "status": (
            "supported"
            if supported
            else "insufficient"
        ),
        "score": round(
            min(
                max(
                    best_score,
                    aggregate_score,
                ),
                1.0,
            ),
            3,
        ),
        "relevant_documents": (
            relevant_documents
        ),
    }


# =========================================================
# Quantitative assessment
# =========================================================

def _assess_quantitative_support(
    query: str,
    documents: List[Any],
    scores: List[float],
) -> Dict[str, Any]:

    qualifying_scores = []

    for document, score in zip(
        documents,
        scores,
    ):

        if score < QUANTITATIVE_MIN_DOCUMENT_SCORE:
            continue

        if not _claim_scope_compatible(
            query,
            document,
        ):
            continue

        if not _contains_quantitative_evidence(
            document.page_content
        ):
            continue

        if not quantitative_temporal_compatible(
            query,
            document.page_content,
        ):
            continue

        qualifying_scores.append(
            score
        )

    best_score = max(
        scores,
        default=0.0,
    )

    if qualifying_scores:

        return {
            "status": "supported",
            "score": round(
                max(
                    qualifying_scores
                ),
                3,
            ),
            "relevant_documents": len(
                qualifying_scores
            ),
        }

    return {
        "status": "insufficient",
        "score": round(
            min(
                best_score,
                0.299,
            ),
            3,
        ),
        "relevant_documents": 0,
    }


# =========================================================
# Requirements assessment
# =========================================================

def _assess_requirements_support(
    query: str,
    documents: List[Any],
    scores: List[float],
) -> Dict[str, Any]:

    qualifying_scores = []

    for document, score in zip(
        documents,
        scores,
    ):

        if score < REQUIREMENTS_MIN_DOCUMENT_SCORE:
            continue

        if not _claim_scope_compatible(
            query,
            document,
        ):
            continue

        if not _contains_requirement_evidence(
            document.page_content
        ):
            continue

        adjusted_score = score

        if _contains_direct_admission_evidence(
            document.page_content
        ):
            adjusted_score += (
                DIRECT_REQUIREMENT_SUPPORT
            )

        qualifying_scores.append(
            min(
                adjusted_score,
                1.0,
            )
        )

    if qualifying_scores:

        return {
            "status": "supported",
            "score": round(
                max(
                    qualifying_scores
                ),
                3,
            ),
            "relevant_documents": len(
                qualifying_scores
            ),
        }

    return {
        "status": "insufficient",
        "score": round(
            max(
                scores,
                default=0.0,
            ),
            3,
        ),
        "relevant_documents": 0,
    }


# =========================================================
# Descriptive assessment
# =========================================================

def _assess_descriptive_support(
    query: str,
    documents: List[Any],
    scores: List[float],
) -> Dict[str, Any]:

    qualifying_scores = [
        score
        for document, score in zip(
            documents,
            scores,
        )
        if (
            score
            >= DESCRIPTIVE_MIN_DOCUMENT_SCORE
            and
            _claim_scope_compatible(
                query,
                document,
            )
        )
    ]

    best_score = max(
        scores,
        default=0.0,
    )

    if qualifying_scores:

        return {
            "status": "supported",
            "score": round(
                best_score,
                3,
            ),
            "relevant_documents": len(
                qualifying_scores
            ),
        }

    return {
        "status": "insufficient",
        "score": round(
            best_score,
            3,
        ),
        "relevant_documents": 0,
    }


# =========================================================
# Main assessment
# =========================================================

def assess_evidence_sufficiency(
    query: str,
    documents,
) -> Dict[str, Any]:

    if not documents:

        return {
            "status": "insufficient",
            "score": 0.0,
            "relevant_documents": 0,
        }

    question_type = detect_question_type(
        query
    )

    scores = [
        _score_document_support(
            query=query,
            document=document,
        )
        for document in documents
    ]

    if question_type == "list":

        return _assess_list_support(
            query=query,
            documents=documents,
            scores=scores,
        )

    if question_type == "quantitative":

        return _assess_quantitative_support(
            query=query,
            documents=documents,
            scores=scores,
        )

    if question_type == "requirements":

        return _assess_requirements_support(
            query=query,
            documents=documents,
            scores=scores,
        )

    return _assess_descriptive_support(
        query=query,
        documents=documents,
        scores=scores,
    )