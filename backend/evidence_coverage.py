"""
IIT Jodhpur V1 — Evidence Coverage

Purpose
-------
Determine whether retrieved evidence sufficiently covers a user's
question for the detected question type.

Phase 4 responsibilities
------------------------
- Question-type detection.
- Focused evidence coverage.
- Numeric evidence validation.
- Requirement evidence validation.
- Broad-list relevance and breadth assessment.

Core distinction
----------------
Relevant evidence is not automatically complete evidence.

For list questions:

    relevant evidence
        ->
    breadth assessment
        ->
    supported / partially_supported / insufficient

Design principles
-----------------
- Deterministic.
- Lightweight.
- No LLM call.
- Institution-agnostic.
- No hardcoded institution counts.
- Specialized question types use specialized evidence signals.
- Existing focused-question behavior remains conservative.
"""

from __future__ import annotations

import re
from typing import Any, Dict


from backend.retriever import (
    normalize_text,
    detect_programs,
    detect_topics,
    detect_entities,
)


# =========================================================
# Configuration
# =========================================================

STRONG_DOCUMENT_SCORE = 0.55
PARTIAL_DOCUMENT_SCORE = 0.30

# Retained for compatibility with existing code.
MIN_LIST_CONTENT_CHARS = 450

# ---------------------------------------------------------
# List coverage
# ---------------------------------------------------------

LIST_RELEVANT_SCORE = 0.20
LIST_STRONG_SCORE = 0.50

LIST_SINGLE_INVENTORY_ITEMS = 5

LIST_SUPPORTED_DOCUMENTS = 3
LIST_SUPPORTED_ITEMS = 7
LIST_SUPPORTED_CUES = 2

LIST_PARTIAL_ITEMS = 2

# ---------------------------------------------------------
# Requirement coverage
# ---------------------------------------------------------

REQUIREMENT_RELEVANT_SCORE = 0.20


# =========================================================
# Question Type Detection
# =========================================================

def detect_question_type(
    question: str,
) -> str:
    """
    Detect the broad question type.

    Priority:

        quantitative
        requirements
        list
        descriptive
    """

    normalized = normalize_text(
        question
    )

    # -----------------------------------------------------
    # Quantitative
    # -----------------------------------------------------

    quantitative_patterns = [
        r"\bhow much\b",
        r"\bhow many\b",

        r"\bwhat is (?:the )?(?:fee|fees)\b",
        r"\bwhat are (?:the )?(?:fee|fees)\b",
        r"\bwhat is .*?\bfee\b",
        r"\bwhat are .*?\bfees\b",

        r"\bwhat is (?:the )?cost\b",
        r"\bwhat are (?:the )?costs\b",
        r"\bwhat is .*?\bcost\b",
        r"\bwhat are .*?\bcosts\b",

        r"\bwhat is (?:the )?charge\b",
        r"\bwhat are (?:the )?charges\b",
        r"\bwhat is .*?\bcharge\b",
        r"\bwhat are .*?\bcharges\b",

        r"\bwhat is (?:the )?price\b",
        r"\bwhat are (?:the )?prices\b",
        r"\bwhat is .*?\bprice\b",
        r"\bwhat are .*?\bprices\b",

        r"\bwhat is (?:the )?amount\b",
        r"\bwhat are (?:the )?amounts\b",

        r"\bhow long\b",
        r"\bwhat is (?:the )?duration\b",
        r"\bwhat are (?:the )?durations\b",
        r"\bwhat is .*?\bduration\b",

        r"\bhow many years\b",
        r"\bhow many months\b",
        r"\bhow many days\b",

        r"\bwhat percentage\b",
        r"\bwhat percent\b",
        r"\bwhat cgpa\b",
        r"\bwhat cpi\b",
        r"\bwhat score\b",
        r"\bwhat rank\b",
        r"\bwhat number\b",
    ]

    if any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in quantitative_patterns
    ):
        return "quantitative"

    # -----------------------------------------------------
    # Requirements / eligibility
    # -----------------------------------------------------

    requirement_patterns = [
        r"\beligibility\b",
        r"\beligible\b",
        r"\brequirement\b",
        r"\brequirements\b",
        r"\bqualification\b",
        r"\bqualifications\b",
        r"\bcriteria\b",

        r"\bwhat do i need\b",
        r"\bwhat is required\b",
        r"\bwhat are required\b",
        r"\bwhat qualification do i need\b",
        r"\bwhat qualifications do i need\b",
        r"\bwhat do .* need\b",

        r"\bcan .* apply\b",
        r"\bwho can apply\b",
        r"\bwhat conditions\b",
        r"\bwhat are the conditions\b",
    ]

    if any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in requirement_patterns
    ):
        return "requirements"

    # -----------------------------------------------------
    # Lists
    # -----------------------------------------------------

    list_patterns = [
        r"\bwhat programs\b",
        r"\bwhich programs\b",

        r"\bwhat programmes\b",
        r"\bwhich programmes\b",

        r"\bwhat research areas\b",
        r"\bwhich research areas\b",

        r"\bwhat research themes\b",
        r"\bwhich research themes\b",

        r"\bwhat areas\b",
        r"\bwhich areas\b",

        r"\bwhat facilities\b",
        r"\bwhich facilities\b",

        r"\bwhat courses\b",
        r"\bwhich courses\b",

        r"\bwhat departments\b",
        r"\bwhich departments\b",

        r"\bwhat schools\b",
        r"\bwhich schools\b",

        r"\bwhat centers\b",
        r"\bwhich centers\b",

        r"\bwhat centres\b",
        r"\bwhich centres\b",

        r"\bwhat are .* available\b",
        r"\bwhich .* are available\b",

        r"\bwhat kinds? of\b",
        r"\bwhat types? of\b",

        r"\bwhat options\b",
        r"\bwhich options\b",

        r"\blist\b",
    ]

    if any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in list_patterns
    ):
        return "list"

    return "descriptive"


# =========================================================
# Query Anchors
# =========================================================

STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "how",
    "who",
    "why",
    "are",
    "is",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "at",
    "do",
    "does",
    "can",
    "could",
    "would",
    "should",
    "there",
    "available",
    "please",
    "tell",
    "me",
    "offer",
    "offers",
    "offered",
    "institute",
    "college",
    "university",
}


def _query_anchors(
    question: str,
):
    """
    Extract lightweight lexical anchors.
    """

    normalized = normalize_text(
        question
    )

    return {
        token
        for token in normalized.split()
        if (
            len(token) > 2
            and token not in STOPWORDS
        )
    }


# =========================================================
# Generic Evidence Detection
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


REQUIREMENT_MARKERS = (
    "eligibility",
    "eligible",
    "qualification",
    "qualifications",
    "requirement",
    "requirements",
    "criteria",
    "degree",
    "degrees",
    "marks",
    "percentage",
    "cgpa",
    "cpi",
    "gate",
    "net",
    "jrf",
    "experience",
    "admission",
)


def _contains_quantitative_evidence(
    text: str,
) -> bool:
    """
    Return True when explicit numerical evidence is present.
    """

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


def _contains_requirement_evidence(
    text: str,
) -> bool:
    """
    Return True when direct qualification/admission evidence exists.
    """

    normalized = normalize_text(
        text
    )

    return any(
        marker in normalized
        for marker in REQUIREMENT_MARKERS
    )


# =========================================================
# Generic Document Support
# =========================================================

def _document_token_overlap(
    question: str,
    content: str,
) -> float:
    """
    Estimate lexical overlap between query and content.
    """

    query_anchors = _query_anchors(
        question
    )

    content_tokens = {
        token
        for token in normalize_text(
            content
        ).split()
        if len(token) > 2
    }

    if not query_anchors:
        return 0.0

    return (
        len(
            query_anchors
            &
            content_tokens
        )
        /
        len(query_anchors)
    )


def _document_support_score(
    query: str,
    document,
) -> float:
    """
    Generic support score for focused question types.
    """

    content = normalize_text(
        document.page_content
        or ""
    )

    if not content:
        return 0.0

    score = (
        _document_token_overlap(
            query,
            content,
        )
        * 0.50
    )

    query_programs = detect_programs(
        query
    )

    document_programs = detect_programs(
        content
    )

    if (
        query_programs
        and
        (
            query_programs
            &
            document_programs
        )
    ):
        score += 0.25

    query_topics = detect_topics(
        query
    )

    document_topics = detect_topics(
        content
    )

    if (
        query_topics
        and
        (
            query_topics
            &
            document_topics
        )
    ):
        score += 0.15

    query_entities = detect_entities(
        query
    )

    document_entities = detect_entities(
        content
    )

    if (
        query_entities
        and
        (
            query_entities
            &
            document_entities
        )
    ):
        score += 0.10

    return min(
        score,
        1.0,
    )


# =========================================================
# Requirement Relevance
# =========================================================

def _requirement_relevance_score(
    query: str,
    document,
) -> float:
    """
    Dedicated relevance score for requirement questions.

    Requirement questions often use abstract words such as "eligibility"
    while source documents use concrete qualification evidence such as
    degrees, marks, CGPA, GATE, NET, and experience.
    """

    content = normalize_text(
        document.page_content
        or ""
    )

    if not content:
        return 0.0

    score = 0.0

    lexical = _document_token_overlap(
        query,
        content,
    )

    score += (
        lexical
        * 0.30
    )

    if _contains_requirement_evidence(
        content
    ):
        score += 0.40

    qualification_signals = (
        "master",
        "bachelor",
        "four-year",
        "degree",
        "marks",
        "percentage",
        "cgpa",
        "cpi",
        "gate",
        "net",
        "jrf",
        "experience",
        "admission",
        "eligible",
        "eligibility",
        "criteria",
        "qualification",
        "requirements",
    )

    matches = sum(
        1
        for signal in qualification_signals
        if signal in content
    )

    if matches >= 1:
        score += 0.15

    if matches >= 3:
        score += 0.10

    return min(
        score,
        1.0,
    )


# =========================================================
# List Scope Detection
# =========================================================

LIST_SCOPE_PATTERNS = {
    "programs": (
        r"\bprogram(?:s|mes)?\b",
        r"\bacademic programs?\b",
        r"\bacademic programmes?\b",
        r"\bdegrees?\b",
    ),
    "research": (
        r"\bresearch areas?\b",
        r"\bresearch themes?\b",
        r"\bresearch\b",
    ),
    "facilities": (
        r"\bfacilities?\b",
        r"\bamenities?\b",
        r"\binfrastructure\b",
    ),
    "departments": (
        r"\bdepartments?\b",
    ),
    "schools": (
        r"\bschools?\b",
    ),
    "centers": (
        r"\bcent(?:er|re)s?\b",
    ),
    "courses": (
        r"\bcourses?\b",
    ),
    "options": (
        r"\boptions?\b",
        r"\btypes?\b",
        r"\bkinds?\b",
    ),
}


# These are concrete program markers. They allow us to recognize program
# evidence even when the source text does not literally contain the word
# "programs".
PROGRAM_EVIDENCE_MARKERS = (
    r"\bb\.?\s*tech\.?\b",
    r"\bm\.?\s*tech\.?\b",
    r"\bm\.?\s*sc\.?\b",
    r"\bb\.?\s*sc\.?\b",
    r"\bph\.?\s*d\.?\b",
    r"\bmba\b",
    r"\bmsc\b",
    r"\bbsc\b",
    r"\bdoctoral\b",
    r"\bpostgraduate\b",
    r"\bundergraduate\b",
    r"\bdegree program\b",
)


def _detect_list_scopes(
    text: str,
) -> set[str]:
    """
    Detect generic list scopes.

    Program scope receives an additional concrete-marker check so that
    evidence such as "B.Tech in Electrical Engineering" is recognized as
    program evidence even when the word "programs" is absent.
    """

    normalized = normalize_text(
        text
    )

    scopes = set()

    for scope, patterns in LIST_SCOPE_PATTERNS.items():

        if any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in patterns
        ):
            scopes.add(
                scope
            )

    # -----------------------------------------------------
    # Concrete program evidence.
    # -----------------------------------------------------

    if any(
        re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )
        for pattern in PROGRAM_EVIDENCE_MARKERS
    ):
        scopes.add(
            "programs"
        )

    return scopes


# =========================================================
# List Inventory Signals
# =========================================================

LIST_INVENTORY_CUES = (
    "offers",
    "offer",
    "includes",
    "include",
    "available",
    "following",
    "list",
    "comprises",
    "consists of",
    "programs:",
    "programmes:",
    "departments:",
    "schools:",
    "centers:",
    "centres:",
    "courses:",
    "facilities:",
    "research areas:",
    "research themes:",
)


def _list_inventory_cue_count(
    content: str,
) -> int:
    """
    Count generic enumeration cues.
    """

    normalized = normalize_text(
        content
    )

    return sum(
        normalized.count(
            cue
        )
        for cue in LIST_INVENTORY_CUES
    )


def _enumeration_item_count(
    content: str,
) -> int:
    """
    Estimate the number of list-like entries.

    This is a heuristic breadth signal, not proof of global completeness.
    """

    raw = str(
        content
        or ""
    )

    normalized = normalize_text(
        raw
    )

    count = 0

    # -----------------------------------------------------
    # Bullets / numbered entries.
    # -----------------------------------------------------

    count += len(
        re.findall(
            r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+",
            raw,
        )
    )

    count += len(
        re.findall(
            r"\b\d+[.)]\s+",
            raw,
        )
    )

    # -----------------------------------------------------
    # Semicolon-delimited inventory.
    # -----------------------------------------------------

    semicolon_items = [
        item.strip()
        for item in raw.split(";")
        if item.strip()
    ]

    if len(
        semicolon_items
    ) >= 2:
        count += len(
            semicolon_items
        )

    # -----------------------------------------------------
    # Comma / conjunction inventory.
    # -----------------------------------------------------

    if any(
        cue in normalized
        for cue in LIST_INVENTORY_CUES
    ):

        comma_items = [
            item.strip()
            for item in re.split(
                r",|\band\b",
                raw,
                flags=re.IGNORECASE,
            )
            if item.strip()
        ]

        if len(
            comma_items
        ) >= 3:
            count += len(
                comma_items
            )

    # -----------------------------------------------------
    # Tables.
    # -----------------------------------------------------

    table_rows = [
        line
        for line in raw.splitlines()
        if "|" in line
    ]

    if len(
        table_rows
    ) >= 3:
        count += len(
            table_rows
        )

    # -----------------------------------------------------
    # Concrete program markers.
    #
    # A source such as:
    #
    #   B.Tech in Electrical Engineering and
    #   B.Tech in Computer Science
    #
    # is inherently list-like program evidence even without explicit
    # bullets or semicolons.
    # -----------------------------------------------------

    program_matches = []

    for pattern in PROGRAM_EVIDENCE_MARKERS:

        program_matches.extend(
            re.finditer(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )
        )

    count += len(
        program_matches
    )

    return count


def _list_document_relevance(
    query: str,
    document,
) -> float:
    """
    Dedicated relevance score for list questions.

    Scope matching is the primary gate. Lexical overlap and enumeration
    signals then increase confidence.
    """

    content = document.page_content or ""

    if not content.strip():
        return 0.0

    query_scopes = _detect_list_scopes(
        query
    )

    document_scopes = _detect_list_scopes(
        content
    )

    # -----------------------------------------------------
    # Scope gate.
    # -----------------------------------------------------

    if query_scopes and not (
        query_scopes
        &
        document_scopes
    ):
        return 0.0

    lexical = _document_token_overlap(
        query,
        content,
    )

    score = (
        lexical
        * 0.40
    )

    if (
        query_scopes
        &
        document_scopes
    ):
        score += 0.35

    cue_count = _list_inventory_cue_count(
        content
    )

    if cue_count >= 1:
        score += 0.10

    if cue_count >= 2:
        score += 0.05

    item_count = _enumeration_item_count(
        content
    )

    if item_count >= 2:
        score += 0.05

    if item_count >= 5:
        score += 0.05

    return min(
        score,
        1.0,
    )


# =========================================================
# Duplicate Resistance
# =========================================================

def _deduplicate_documents(
    documents,
):
    """
    Remove exact textual duplicates.
    """

    seen = set()
    unique = []

    for document in documents:

        content = normalize_text(
            getattr(
                document,
                "page_content",
                "",
            )
        )

        if not content:
            continue

        if content in seen:
            continue

        seen.add(
            content
        )

        unique.append(
            document
        )

    return unique


# =========================================================
# Broad List Coverage
# =========================================================

def _assess_list_coverage(
    query: str,
    documents,
) -> Dict[str, Any]:
    """
    Assess list evidence separately from focused-question support.

    Supported means the retrieved evidence is sufficiently broad and
    coherent to answer confidently.

    It does not make a claim that the entire institutional corpus is
    globally exhaustive.
    """

    unique_documents = _deduplicate_documents(
        documents
    )

    if not unique_documents:

        return {
            "status": "insufficient",
            "list_completeness": "insufficient",
            "relevant_list_documents": 0,
            "list_inventory_items": 0,
            "list_inventory_cues": 0,
        }

    scored_documents = [
        (
            document,
            _list_document_relevance(
                query=query,
                document=document,
            ),
        )
        for document in unique_documents
    ]

    relevant_documents = [
        document
        for document, score in scored_documents
        if score >= LIST_RELEVANT_SCORE
    ]

    if not relevant_documents:

        return {
            "status": "insufficient",
            "list_completeness": "insufficient",
            "relevant_list_documents": 0,
            "list_inventory_items": 0,
            "list_inventory_cues": 0,
        }

    inventory_items = sum(
        _enumeration_item_count(
            document.page_content
            or ""
        )
        for document in relevant_documents
    )

    inventory_cues = sum(
        _list_inventory_cue_count(
            document.page_content
            or ""
        )
        for document in relevant_documents
    )

    strong_relevant_documents = sum(
        1
        for document, score in scored_documents
        if (
            document in relevant_documents
            and
            score >= LIST_STRONG_SCORE
        )
    )

    # -----------------------------------------------------
    # Strong single inventory.
    # -----------------------------------------------------

    for document, score in scored_documents:

        if score < LIST_STRONG_SCORE:
            continue

        item_count = _enumeration_item_count(
            document.page_content
            or ""
        )

        cue_count = _list_inventory_cue_count(
            document.page_content
            or ""
        )

        if (
            item_count
            >= LIST_SINGLE_INVENTORY_ITEMS
            and
            cue_count >= 1
        ):

            return {
                "status": "supported",
                "list_completeness": "supported",
                "relevant_list_documents": len(
                    relevant_documents
                ),
                "list_inventory_items": inventory_items,
                "list_inventory_cues": inventory_cues,
            }

    # -----------------------------------------------------
    # Multiple coherent documents.
    # -----------------------------------------------------

    if (
        len(
            relevant_documents
        )
        >= LIST_SUPPORTED_DOCUMENTS
        and
        inventory_items
        >= LIST_SUPPORTED_ITEMS
        and
        inventory_cues
        >= LIST_SUPPORTED_CUES
        and
        (
            strong_relevant_documents >= 1
            or
            inventory_items
            >= LIST_SUPPORTED_ITEMS
        )
    ):

        return {
            "status": "supported",
            "list_completeness": "supported",
            "relevant_list_documents": len(
                relevant_documents
            ),
            "list_inventory_items": inventory_items,
            "list_inventory_cues": inventory_cues,
        }

    # -----------------------------------------------------
    # Partial list evidence.
    # -----------------------------------------------------

    if (
        len(
            relevant_documents
        )
        >= 1
        and
        (
            inventory_items >= LIST_PARTIAL_ITEMS
            or
            inventory_cues >= 1
        )
    ):

        return {
            "status": "partially_supported",
            "list_completeness": "partial",
            "relevant_list_documents": len(
                relevant_documents
            ),
            "list_inventory_items": inventory_items,
            "list_inventory_cues": inventory_cues,
        }

    return {
        "status": "insufficient",
        "list_completeness": "insufficient",
        "relevant_list_documents": len(
            relevant_documents
        ),
        "list_inventory_items": inventory_items,
        "list_inventory_cues": inventory_cues,
    }


# =========================================================
# Main Coverage Assessment
# =========================================================

def assess_evidence_coverage(
    query: str,
    documents,
) -> Dict[str, Any]:
    """
    Determine evidence coverage for the detected question type.
    """

    question_type = detect_question_type(
        query
    )

    # -----------------------------------------------------
    # Empty evidence.
    # -----------------------------------------------------

    if not documents:

        return {
            "status": "insufficient",
            "question_type": question_type,
            "strong_documents": 0,
            "partial_documents": 0,
            "combined_characters": 0,
        }

    # -----------------------------------------------------
    # LIST QUESTIONS
    # -----------------------------------------------------

    if question_type == "list":

        generic_scores = [
            _document_support_score(
                query=query,
                document=document,
            )
            for document in documents
        ]

        strong_documents = sum(
            1
            for score in generic_scores
            if score >= STRONG_DOCUMENT_SCORE
        )

        partial_documents = sum(
            1
            for score in generic_scores
            if score >= PARTIAL_DOCUMENT_SCORE
        )

        combined_characters = sum(
            len(
                document.page_content
                or ""
            )
            for document in documents
        )

        list_result = _assess_list_coverage(
            query=query,
            documents=documents,
        )

        return {
            "status": list_result[
                "status"
            ],
            "question_type": question_type,
            "strong_documents": strong_documents,
            "partial_documents": partial_documents,
            "combined_characters": combined_characters,
            "list_completeness": list_result[
                "list_completeness"
            ],
            "relevant_list_documents": list_result[
                "relevant_list_documents"
            ],
            "list_inventory_items": list_result[
                "list_inventory_items"
            ],
            "list_inventory_cues": list_result[
                "list_inventory_cues"
            ],
        }

    # -----------------------------------------------------
    # REQUIREMENTS
    # -----------------------------------------------------

    if question_type == "requirements":

        requirement_scores = [
            _requirement_relevance_score(
                query=query,
                document=document,
            )
            for document in documents
        ]

        strong_documents = sum(
            1
            for score in requirement_scores
            if score >= STRONG_DOCUMENT_SCORE
        )

        partial_documents = sum(
            1
            for score in requirement_scores
            if score >= REQUIREMENT_RELEVANT_SCORE
        )

        combined_characters = sum(
            len(
                document.page_content
                or ""
            )
            for document in documents
        )

        requirement_support = any(
            (
                score
                >=
                REQUIREMENT_RELEVANT_SCORE
                and
                _contains_requirement_evidence(
                    document.page_content
                    or ""
                )
            )
            for score, document in zip(
                requirement_scores,
                documents,
            )
        )

        if requirement_support:

            status = "supported"

        elif partial_documents >= 1:

            status = "partially_supported"

        else:

            status = "insufficient"

        return {
            "status": status,
            "question_type": question_type,
            "strong_documents": strong_documents,
            "partial_documents": partial_documents,
            "combined_characters": combined_characters,
        }

    # -----------------------------------------------------
    # FOCUSED QUESTIONS
    # -----------------------------------------------------

    scores = [
        _document_support_score(
            query=query,
            document=document,
        )
        for document in documents
    ]

    strong_documents = sum(
        1
        for score in scores
        if score >= STRONG_DOCUMENT_SCORE
    )

    partial_documents = sum(
        1
        for score in scores
        if score >= PARTIAL_DOCUMENT_SCORE
    )

    combined_characters = sum(
        len(
            document.page_content
            or ""
        )
        for document in documents
    )

    # -----------------------------------------------------
    # QUANTITATIVE
    # -----------------------------------------------------

    if question_type == "quantitative":

        numeric_support = any(
            (
                score
                >=
                PARTIAL_DOCUMENT_SCORE
                and
                _contains_quantitative_evidence(
                    document.page_content
                    or ""
                )
            )
            for score, document in zip(
                scores,
                documents,
            )
        )

        if numeric_support:

            status = "supported"

        elif partial_documents >= 1:

            status = "partially_supported"

        else:

            status = "insufficient"

    # -----------------------------------------------------
    # DESCRIPTIVE
    # -----------------------------------------------------

    else:

        if strong_documents >= 1:

            status = "supported"

        elif partial_documents >= 1:

            status = "partially_supported"

        else:

            status = "insufficient"

    return {
        "status": status,
        "question_type": question_type,
        "strong_documents": strong_documents,
        "partial_documents": partial_documents,
        "combined_characters": combined_characters,
    }
