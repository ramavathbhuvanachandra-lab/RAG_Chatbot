"""
Generic Claim-Scope Safety Layer

Purpose
-------
Provide a small deterministic safety layer for retrieved evidence.

This module is NOT responsible for:
    - understanding the user's topic
    - discovering research areas
    - deciding semantic relevance
    - generating retrieval queries
    - replacing the query interpreter
    - replacing the reranker

The canonical flow is:

    user question
        ↓
    query understanding
        ↓
    retrieval
        ↓
    reranking
        ↓
    claim-scope safety check
        ↓
    evidence sufficiency / coverage
        ↓
    answer generation

This module only removes evidence when there is a concrete,
explicit incompatibility between the requested claim and the
retrieved evidence.

Important invariants
--------------------
1. Broad queries may use narrow evidence.
2. Narrow evidence is NOT automatically a conflict.
3. Explicitly different named organizations are a conflict.
4. Explicitly incompatible admission modes are a conflict.
5. Financial-assistance-only evidence does not satisfy an admission claim.
6. No institution-specific organization names are hardcoded.
7. No topic-specific retrieval vocabulary is used to determine relevance.
8. Filtering is document-local: one conflicting document is removed,
   while compatible evidence remains untouched.
"""

from __future__ import annotations

import re
from typing import Set

from backend.retriever import normalize_text


# =========================================================
# Generic claim scopes
# =========================================================
#
# These are generic evidence dimensions that can exist across
# different institutions.
#
# They are NOT institution-specific topics.
#

ADMISSION_SCOPE = "admission"
FINANCIAL_ASSISTANCE_SCOPE = "financial_assistance"
APPLICATION_SCOPE = "application"
FEES_SCOPE = "fees"

# Backward-compatible names used by older callers.
HOSTEL_SCOPE = "hostel"
RESEARCH_SCOPE = "research"
PROGRAM_SCOPE = "program"
FACULTY_SCOPE = "faculty"
FACILITIES_SCOPE = "facilities"


# =========================================================
# Organizational classes
# =========================================================

INSTITUTE_SCOPE = "institute"
SCHOOL_SCOPE = "school"
DEPARTMENT_SCOPE = "department"
CENTRE_SCOPE = "centre"
PROGRAM_ENTITY_SCOPE = "program_entity"


# =========================================================
# Admission modes
# =========================================================

REGULAR_MODE = "regular"
PART_TIME_MODE = "part_time"
SPONSORED_MODE = "sponsored"
EXTERNAL_MODE = "external"


# =========================================================
# Generic claim vocabulary
# =========================================================
#
# Keep this list small.
#
# The retriever/query interpreter remains responsible for topic
# understanding. These terms exist only for explicit safety
# relationships such as admission vs financial assistance.
#

CLAIM_SCOPE_TERMS = {
    ADMISSION_SCOPE: {
        "admission",
        "admissions",
        "eligibility",
        "eligible",
        "qualification",
        "qualifications",
        "criteria",
        "requirement",
        "requirements",
        "apply",
        "applying",
    },

    FINANCIAL_ASSISTANCE_SCOPE: {
        "financial assistance",
        "financial aid",
        "fellowship",
        "fellowships",
        "stipend",
        "stipends",
        "funding",
        "funded",
        "scholarship",
        "scholarships",
    },

    APPLICATION_SCOPE: {
        "application",
        "application form",
        "application process",
        "how to apply",
        "submission",
        "submit",
    },

    FEES_SCOPE: {
        "fee",
        "fees",
        "cost",
        "costs",
        "charge",
        "charges",
        "rent",
        "tuition",
        "payment",
    },
}


# =========================================================
# Admission markers
# =========================================================

ADMISSION_CLAIM_MARKERS = {
    "must have",
    "minimum",
    "minimum four-year",
    "four-year degree",
    "bachelor's degree",
    "bachelor degree",
    "master's degree",
    "master degree",
    "marks",
    "percentage",
    "cgpa",
    "cpi",
    "qualifying degree",
    "qualifying qualification",
    "equivalent",
    "eligible",
    "eligibility",
}


# =========================================================
# Financial-assistance markers
# =========================================================

FINANCIAL_ASSISTANCE_MARKERS = {
    "financial assistance",
    "financial aid",
    "fellowship",
    "stipend",
    "funding",
    "scholarship",
}


# =========================================================
# Fee markers
# =========================================================

FEE_CLAIM_MARKERS = {
    "fee",
    "fees",
    "cost",
    "costs",
    "charge",
    "charges",
    "rent",
    "tuition",
}


# =========================================================
# Organizational vocabulary
# =========================================================

ORGANIZATIONAL_TERMS = {
    INSTITUTE_SCOPE: {
        "institute-wide",
        "institute wide",
        "across the institute",
        "at the institute",
        "institute",
        "university-wide",
        "university wide",
        "across the university",
        "at the university",
        "university",
    },

    SCHOOL_SCOPE: {
        "school",
        "schools",
    },

    DEPARTMENT_SCOPE: {
        "department",
        "departments",
    },

    CENTRE_SCOPE: {
        "centre",
        "centres",
        "center",
        "centers",
    },

    PROGRAM_ENTITY_SCOPE: {
        "program",
        "programs",
        "programme",
        "programmes",
    },
}


# =========================================================
# Admission-mode vocabulary
# =========================================================

MODE_TERMS = {
    REGULAR_MODE: {
        "regular",
        "full-time",
        "full time",
    },

    PART_TIME_MODE: {
        "part-time",
        "part time",
    },

    SPONSORED_MODE: {
        "sponsored",
        "sponsor",
    },

    EXTERNAL_MODE: {
        "external",
    },
}


# =========================================================
# Normalization helpers
# =========================================================

def _normalized(
    text: str,
) -> str:
    """
    Normalize text through the shared retrieval normalizer.
    """

    return normalize_text(
        text
    )


def _contains_term(
    normalized_text: str,
    term: str,
) -> bool:
    """
    Match complete tokens/phrases rather than arbitrary substrings.
    """

    normalized_term = _normalized(
        term
    )

    if not normalized_term:
        return False

    text_tokens = normalized_text.split()
    term_tokens = normalized_term.split()

    if len(term_tokens) == 1:
        return (
            term_tokens[0]
            in text_tokens
        )

    width = len(
        term_tokens
    )

    for index in range(
        len(text_tokens) - width + 1
    ):
        if (
            text_tokens[
                index:
                index + width
            ]
            == term_tokens
        ):
            return True

    return False


def _contains_any_term(
    normalized_text: str,
    terms: Set[str],
) -> bool:
    """
    Return True when any complete term/phrase is present.
    """

    return any(
        _contains_term(
            normalized_text,
            term,
        )
        for term in terms
    )


# =========================================================
# Claim-scope detection
# =========================================================

def detect_claim_scopes(
    text: str,
) -> Set[str]:
    """
    Detect generic claim dimensions.

    This function intentionally does NOT attempt to identify
    arbitrary subjects such as robotics, economics, chemistry,
    electrical engineering, etc.

    The semantic query interpreter/retriever owns that responsibility.
    """

    normalized = _normalized(
        text
    )

    scopes: Set[str] = set()

    for scope, terms in CLAIM_SCOPE_TERMS.items():
        if _contains_any_term(
            normalized,
            terms,
        ):
            scopes.add(
                scope
            )

    if _contains_any_term(
        normalized,
        ADMISSION_CLAIM_MARKERS,
    ):
        scopes.add(
            ADMISSION_SCOPE
        )

    if _contains_any_term(
        normalized,
        FINANCIAL_ASSISTANCE_MARKERS,
    ):
        scopes.add(
            FINANCIAL_ASSISTANCE_SCOPE
        )

    if _contains_any_term(
        normalized,
        FEE_CLAIM_MARKERS,
    ):
        scopes.add(
            FEES_SCOPE
        )

    return scopes


# =========================================================
# Organizational class detection
# =========================================================

def detect_organizational_scopes(
    text: str,
) -> Set[str]:
    """
    Detect organizational classes only.

    Examples:

        department
        school
        centre
        institute-wide
    """

    normalized = _normalized(
        text
    )

    scopes: Set[str] = set()

    for scope, terms in ORGANIZATIONAL_TERMS.items():
        if _contains_any_term(
            normalized,
            terms,
        ):
            scopes.add(
                scope
            )

    return scopes


# =========================================================
# Organization-name extraction
# =========================================================

_ORGANIZATIONAL_STOPWORDS = {
    "research",
    "researches",
    "admission",
    "admissions",
    "eligibility",
    "eligible",
    "requirement",
    "requirements",
    "program",
    "programs",
    "programme",
    "programmes",
    "course",
    "courses",
    "faculty",
    "facilities",
    "offers",
    "offer",
    "provides",
    "provide",
    "pursues",
    "pursue",
    "has",
    "have",
    "includes",
    "include",
    "with",
    "for",
    "is",
    "are",
}


def _clean_organization_name(
    value: str,
) -> str:
    """
    Clean an extracted organization name.
    """

    normalized = _normalized(
        value
    )

    if not normalized:
        return ""

    tokens = normalized.split()

    cleaned = []

    for token in tokens:
        if token in _ORGANIZATIONAL_STOPWORDS:
            break

        cleaned.append(
            token
        )

    return " ".join(
        cleaned
    ).strip()


def _extract_explicit_organization_names(
    text: str,
) -> Set[str]:
    """
    Extract organization identities from explicit forms:

        School of X
        Department of X
        Centre of X
        Center of X

    This is deliberately structural and generic.
    """

    normalized = _normalized(
        text
    )

    if not normalized:
        return set()

    names: Set[str] = set()

    pattern = re.compile(
        r"\b(?:school|department|centre|center)"
        r"\s+of\s+"
        r"(.+?)"
        r"(?=\s+(?:research|researches|"
        r"admission|admissions|eligibility|eligible|"
        r"requirements?|program(?:s|mes)?|course(?:s)?|"
        r"faculty|facilities|offers?|provides?|"
        r"pursues?|has|have|includes?|include|"
        r"with|for|is|are)\b|$)",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(
        normalized
    ):
        name = _clean_organization_name(
            match.group(
                1
            )
        )

        if name:
            names.add(
                name
            )

    return names


def _extract_contextual_organization_names(
    text: str,
) -> Set[str]:
    """
    Extract an organization referenced without explicitly saying
    school/department/centre.

    Examples:

        research in Electrical Engineering
        requirements in Computer Science
        programs at Data Science

    This is only used to compare an already-narrow document
    against a query that names the organization naturally.
    """

    normalized = _normalized(
        text
    )

    if not normalized:
        return set()

    names: Set[str] = set()

    pattern = re.compile(
        r"\b(?:in|at|from|within|under|through)"
        r"\s+"
        r"(?:the\s+)?"
        r"(.+?)"
        r"(?=\s+(?:research|researches|"
        r"admission|admissions|eligibility|"
        r"requirements?|program(?:s|mes)?|"
        r"course(?:s)?|faculty|facilities|"
        r"offers?|provides?|pursues?|has|have|"
        r"includes?|include|with|for|is|are)\b|$)",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(
        normalized
    ):
        name = _clean_organization_name(
            match.group(
                1
            )
        )

        if name:
            names.add(
                name
            )

    return names


def _organization_names_match(
    first_name: str,
    second_name: str,
) -> bool:
    """
    Compare organization identities conservatively.

    Exact normalized matches are preferred.
    A complete shorter phrase may also match a longer identity.
    """

    first = _normalized(
        first_name
    )

    second = _normalized(
        second_name
    )

    if not first or not second:
        return False

    if first == second:
        return True

    first_tokens = first.split()
    second_tokens = second.split()

    if len(first_tokens) > len(
        second_tokens
    ):
        first_tokens, second_tokens = (
            second_tokens,
            first_tokens,
        )

    width = len(
        first_tokens
    )

    for index in range(
        len(second_tokens) - width + 1
    ):
        if (
            second_tokens[
                index:
                index + width
            ]
            == first_tokens
        ):
            return True

    return False


def _organization_identity_compatible(
    query: str,
    document_text: str,
) -> bool:
    """
    Determine whether the explicitly narrow document organization
    matches the organization named by the query.

    Broad queries are allowed to use narrow documents.

    This function is therefore only used when the query contains
    an organization identity.
    """

    document_names = (
        _extract_explicit_organization_names(
            document_text
        )
    )

    if not document_names:
        return True

    query_names = (
        _extract_explicit_organization_names(
            query
        )
    )

    if not query_names:
        query_names = (
            _extract_contextual_organization_names(
                query
            )
        )

    if not query_names:
        return True

    return any(
        _organization_names_match(
            query_name,
            document_name,
        )
        for query_name in query_names
        for document_name in document_names
    )


# =========================================================
# Admission mode detection
# =========================================================

def detect_admission_modes(
    text: str,
) -> Set[str]:
    """
    Detect explicit admission modes.
    """

    normalized = _normalized(
        text
    )

    modes: Set[str] = set()

    for mode, terms in MODE_TERMS.items():
        if _contains_any_term(
            normalized,
            terms,
        ):
            modes.add(
                mode
            )

    return modes


# =========================================================
# Generic scope compatibility
# =========================================================

def scopes_compatible(
    query_scopes: Set[str],
    document_scopes: Set[str],
) -> bool:
    """
    Determine broad claim compatibility.

    This helper is intentionally permissive:
    absence of a scope in a document does not automatically
    make the document invalid.
    """

    if not query_scopes:
        return True

    if not document_scopes:
        return True

    return bool(
        query_scopes
        &
        document_scopes
    )


# =========================================================
# Admission-mode compatibility
# =========================================================

def admission_modes_compatible(
    query_modes: Set[str],
    document_modes: Set[str],
) -> bool:
    """
    Determine admission-mode compatibility.

    Missing mode information remains permissive.
    """

    if not query_modes:
        return True

    if not document_modes:
        return True

    return bool(
        query_modes
        &
        document_modes
    )


# =========================================================
# Organizational conflict detection
# =========================================================

def organizational_scope_conflict(
    query: str,
    document_text: str,
) -> bool:
    """
    Reject only explicit organizational identity conflicts.

    Important:

    A broad query such as:

        "What research areas are available?"

    is NOT in conflict with:

        "Department of Electrical Engineering research areas..."

    That department document is allowed as candidate evidence.

    A narrow query such as:

        "What research areas are available in Electrical Engineering?"

    is also compatible with:

        "Department of Electrical Engineering research areas..."

    But:

        "School of Artificial Intelligence and Data Science"

    must not be satisfied by evidence explicitly scoped to:

        "School of Electrical Engineering"
    """

    query_org = detect_organizational_scopes(
        query
    )

    document_org = detect_organizational_scopes(
        document_text
    )

    narrow_scopes = {
        SCHOOL_SCOPE,
        DEPARTMENT_SCOPE,
        CENTRE_SCOPE,
    }

    document_narrow_scopes = (
        document_org
        &
        narrow_scopes
    )

    # No narrow organizational scope in the evidence.
    if not document_narrow_scopes:
        return False

    # -----------------------------------------------------
    # Query explicitly names a narrow organizational class.
    # -----------------------------------------------------

    query_narrow_scopes = (
        query_org
        &
        narrow_scopes
    )

    if query_narrow_scopes:

        query_names = (
            _extract_explicit_organization_names(
                query
            )
        )

        document_names = (
            _extract_explicit_organization_names(
                document_text
            )
        )

        # When both sides expose explicit identities,
        # reject mismatches.
        if query_names and document_names:
            return not any(
                _organization_names_match(
                    query_name,
                    document_name,
                )
                for query_name in query_names
                for document_name in document_names
            )

        # If identities cannot be extracted safely,
        # do not invent a conflict.
        return False

    # -----------------------------------------------------
    # Query names an organization naturally but does not
    # explicitly say "department"/"school".
    # -----------------------------------------------------

    query_names = (
        _extract_explicit_organization_names(
            query
        )
    )

    if not query_names:
        query_names = (
            _extract_contextual_organization_names(
                query
            )
        )

    # -----------------------------------------------------
    # Broad query:
    #
    # No explicit organization means the narrow document
    # remains a valid candidate. Coverage later decides
    # whether it is enough.
    # -----------------------------------------------------

    if not query_names:
        return False

    document_names = (
        _extract_explicit_organization_names(
            document_text
        )
    )

    if not document_names:
        return False

    return not any(
        _organization_names_match(
            query_name,
            document_name,
        )
        for query_name in query_names
        for document_name in document_names
    )


# =========================================================
# Main claim-scope conflict detection
# =========================================================

def has_scope_conflict(
    query: str,
    document_text: str,
) -> bool:
    """
    Detect only explicit conflicts that should justify removing
    the individual document.

    Relevance decisions belong to retrieval/reranking.
    """

    query_normalized = _normalized(
        query
    )

    document_normalized = _normalized(
        document_text
    )

    # -----------------------------------------------------
    # 1. Organizational identity conflict
    # -----------------------------------------------------

    if organizational_scope_conflict(
        query,
        document_text,
    ):
        return True

    # -----------------------------------------------------
    # 2. Admission vs financial assistance
    #
    # Financial assistance may be useful in other contexts,
    # but it does not satisfy a direct admission/eligibility
    # claim when it contains no admission claim itself.
    # -----------------------------------------------------

    query_scopes = detect_claim_scopes(
        query_normalized
    )

    document_scopes = detect_claim_scopes(
        document_normalized
    )

    if (
        ADMISSION_SCOPE
        in query_scopes
        and
        FINANCIAL_ASSISTANCE_SCOPE
        in document_scopes
        and
        ADMISSION_SCOPE
        not in document_scopes
    ):
        return True

    # -----------------------------------------------------
    # 3. Explicit admission-mode conflict
    # -----------------------------------------------------

    query_modes = detect_admission_modes(
        query_normalized
    )

    document_modes = detect_admission_modes(
        document_normalized
    )

    alternate_modes = {
        PART_TIME_MODE,
        SPONSORED_MODE,
        EXTERNAL_MODE,
    }

    if (
        REGULAR_MODE
        in query_modes
        and
        bool(
            document_modes
            &
            alternate_modes
        )
        and
        REGULAR_MODE
        not in document_modes
    ):
        return True

    return False