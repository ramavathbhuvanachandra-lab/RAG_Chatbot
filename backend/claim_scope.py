"""
Phase 4/5 — Claim Scope Detection

Purpose
-------
Identify semantic claim scope, organizational scope, and admission-mode
compatibility between user queries and retrieved evidence.

The module is institution-agnostic.

Important invariants
--------------------
1. Matching is token/phrase aware.
2. Substrings must not create false matches.
3. Explicit semantic conflicts reject evidence.
4. Narrow organizational evidence may support a query when the query
   identifies the same organization by name, even without explicitly
   saying "department", "school", or "centre".
5. Narrow organizational evidence must not satisfy a broader query
   automatically.
6. Different named organizations must not be mixed.
7. Admission modes remain scope-sensitive.
8. No IITJ-specific organization names are hardcoded.
"""

from __future__ import annotations

import re
from typing import Set

from backend.retriever import normalize_text


# =========================================================
# Claim scopes
# =========================================================

ADMISSION_SCOPE = "admission"
FINANCIAL_ASSISTANCE_SCOPE = "financial_assistance"
APPLICATION_SCOPE = "application"
FEES_SCOPE = "fees"
HOSTEL_SCOPE = "hostel"
RESEARCH_SCOPE = "research"
PROGRAM_SCOPE = "program"
FACULTY_SCOPE = "faculty"
FACILITIES_SCOPE = "facilities"


# =========================================================
# Organizational scopes
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
# Claim scope terms
# =========================================================

SCOPE_TERMS = {
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

    HOSTEL_SCOPE: {
        "hostel",
        "hostels",
        "accommodation",
        "room",
        "rooms",
        "residence",
        "residential",
    },

    RESEARCH_SCOPE: {
        "research",
        "research area",
        "research areas",
        "research theme",
        "research themes",
        "research group",
        "research groups",
        "research work",
    },

    PROGRAM_SCOPE: {
        "program",
        "programs",
        "programme",
        "programmes",
        "degree",
        "degrees",
        "course",
        "courses",
    },

    FACULTY_SCOPE: {
        "faculty",
        "professor",
        "professors",
        "faculty member",
        "faculty members",
    },

    FACILITIES_SCOPE: {
        "facility",
        "facilities",
        "amenity",
        "amenities",
        "infrastructure",
    },
}


# =========================================================
# Organizational terms
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
# Admission mode terms
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
# Direct claim markers
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


FINANCIAL_ASSISTANCE_MARKERS = {
    "financial assistance",
    "financial aid",
    "fellowship",
    "stipend",
    "funding",
    "scholarship",
}


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
    Match a complete token or phrase.

    Prevents false substring matches such as:

        fee  != feet
        room != classroom
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
    Return True when any complete term or phrase exists.
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
    Detect broad semantic claim scopes.

    Multiple scopes may legitimately be returned.
    """

    normalized = _normalized(
        text
    )

    scopes: Set[str] = set()

    for scope, terms in SCOPE_TERMS.items():

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
# Organizational scope detection
# =========================================================

def detect_organizational_scopes(
    text: str,
) -> Set[str]:
    """
    Detect explicit organizational classes.

    Examples:

        department
        school
        centre
        institute-wide

    Absence of an organizational class does not imply a conflict.
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
# Named organizational identity
# =========================================================

# These are generic claim boundary words.
# They are NOT institution-specific.
_ORGANIZATIONAL_BOUNDARY_WORDS = {
    "research",
    "researches",
    "admission",
    "admissions",
    "eligibility",
    "requirement",
    "requirements",
    "offer",
    "offers",
    "provide",
    "provides",
    "pursue",
    "pursues",
    "has",
    "have",
    "include",
    "includes",
    "with",
    "for",
}


def _clean_organizational_name(
    value: str,
) -> str:
    """
    Clean an extracted organization name.

    'and' is intentionally preserved.
    """

    normalized = _normalized(
        value
    )

    if not normalized:
        return ""

    tokens = normalized.split()

    cleaned: list[str] = []

    for token in tokens:

        if token in _ORGANIZATIONAL_BOUNDARY_WORDS:
            break

        cleaned.append(
            token
        )

    return " ".join(
        cleaned
    ).strip()


def _extract_explicit_organizational_entities(
    text: str,
) -> Set[str]:
    """
    Extract explicit organization identities only from safe structural
    forms:

        School of X
        Department of X
        Centre of X
        Center of X

    We intentionally do NOT use a broad suffix regex here. That regex can
    accidentally capture unrelated words before "school"/"department"
    inside a normal question.
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
        r"(?=\s+(?:research|researches|admission|admissions|"
        r"eligibility|requirements?|offers?|provides?|"
        r"pursues?|has|have|includes?|with|for)\b"
        r"|[,.!?;:]|$)",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(
        normalized
    ):

        name = _clean_organizational_name(
            match.group(
                1
            )
        )

        if name:
            names.add(
                name
            )

    return names


def _extract_contextual_organization_entities(
    text: str,
) -> Set[str]:
    """
    Extract organization names referenced without explicitly saying
    school/department/centre.

    Examples:

        research in Electrical Engineering
            -> electrical engineering

        requirements in Computer Science
            -> computer science

        programs at Data Science
            -> data science

    This parser is intentionally conservative.
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
        r"(?=\s+(?:research|researches|admission|admissions|"
        r"eligibility|requirements?|program|programs|programme|"
        r"programmes|course|courses|facilities?|department|"
        r"school|centre|center)\b"
        r"|[,.!?;:]|$)",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(
        normalized
    ):

        name = _clean_organizational_name(
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
    Compare two organization identities conservatively.

    Exact normalized identity is preferred.

    A shorter complete phrase may match a longer identity when the shorter
    phrase occurs as a contiguous sequence.
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


def _organizational_identity_compatible(
    query: str,
    document_text: str,
) -> bool:
    """
    Determine whether narrow organizational evidence matches the
    organization referenced by the query.

    Explicit organization names are authoritative.

    If the document is explicitly narrow and the query does not identify
    any organization, the evidence is considered too narrow.
    """

    document_names = (
        _extract_explicit_organizational_entities(
            document_text
        )
    )

    # Evidence is not explicitly narrow.
    if not document_names:
        return True

    query_names = (
        _extract_explicit_organizational_entities(
            query
        )
    )

    # A query such as:
    #
    #   research in Electrical Engineering
    #
    # identifies an organization without saying "department".
    if not query_names:

        query_names = (
            _extract_contextual_organization_entities(
                query
            )
        )

    # Narrow evidence without any organization in the query is not
    # sufficient to establish a broader claim.
    if not query_names:
        return False

    return any(
        _organization_names_match(
            query_name,
            document_name,
        )
        for query_name in query_names
        for document_name in document_names
    )


# =========================================================
# Admission-mode detection
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
# Broad claim compatibility
# =========================================================

def scopes_compatible(
    query_scopes: Set[str],
    document_scopes: Set[str],
) -> bool:
    """
    Determine broad semantic compatibility.

    Empty document scopes remain permissive because a useful evidence
    chunk may not explicitly repeat its category.
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

    If evidence does not explicitly identify a mode, it remains usable.
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
    Reject explicit organizational mismatches.

    Examples:

        Query:
            research areas in Electrical Engineering

        Evidence:
            Department of Electrical Engineering research areas

        -> compatible

        Query:
            Ph.D. requirements in School A

        Evidence:
            School B Ph.D. requirements

        -> conflict

        Query:
            What research areas are available?

        Evidence:
            Department of Electrical Engineering research areas

        -> conflict because the evidence is narrower than the query.
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

    if not document_narrow_scopes:
        return False

    query_narrow_scopes = (
        query_org
        &
        narrow_scopes
    )

    # -----------------------------------------------------
    # Query explicitly names the organizational class.
    # -----------------------------------------------------

    if query_narrow_scopes:

        if not (
            query_narrow_scopes
            &
            document_narrow_scopes
        ):
            return True

        return not (
            _organizational_identity_compatible(
                query,
                document_text,
            )
        )

    # -----------------------------------------------------
    # Query does not explicitly name the class.
    # It may still identify the organization by name.
    # -----------------------------------------------------

    return not (
        _organizational_identity_compatible(
            query,
            document_text,
        )
    )


# =========================================================
# Main scope conflict detection
# =========================================================

def has_scope_conflict(
    query: str,
    document_text: str,
) -> bool:
    """
    Detect explicit semantic conflicts between query and evidence.
    """

    query_normalized = _normalized(
        query
    )

    document_normalized = _normalized(
        document_text
    )

    query_scopes = detect_claim_scopes(
        query_normalized
    )

    document_scopes = detect_claim_scopes(
        document_normalized
    )

    query_modes = detect_admission_modes(
        query_normalized
    )

    document_modes = detect_admission_modes(
        document_normalized
    )

    # -----------------------------------------------------
    # Organizational scope
    # -----------------------------------------------------

    if organizational_scope_conflict(
        query,
        document_text,
    ):
        return True

    # -----------------------------------------------------
    # Admission vs financial assistance
    # -----------------------------------------------------

    if (
        ADMISSION_SCOPE
        in
        query_scopes
        and
        FINANCIAL_ASSISTANCE_SCOPE
        in
        document_scopes
        and
        ADMISSION_SCOPE
        not in
        document_scopes
    ):
        return True

    # -----------------------------------------------------
    # Regular vs alternate admission modes
    # -----------------------------------------------------

    alternate_modes = {
        PART_TIME_MODE,
        SPONSORED_MODE,
        EXTERNAL_MODE,
    }

    if (
        REGULAR_MODE
        in
        query_modes
        and
        bool(
            document_modes
            &
            alternate_modes
        )
        and
        REGULAR_MODE
        not in
        document_modes
    ):
        return True

    # -----------------------------------------------------
    # Fee vs unrelated evidence
    # -----------------------------------------------------

    if (
        FEES_SCOPE
        in
        query_scopes
        and
        FEES_SCOPE
        not in
        document_scopes
    ):
        return True

    return False