"""
IIT Jodhpur V1 — Phase 5.1
Student Situation Understanding

Purpose
-------
Convert a natural-language student message into a compact structured
representation for later routing/query planning.

This layer does NOT:
- retrieve college evidence
- decide institutional policy
- generate an answer
- call an LLM
- replace the existing retrieval/evidence pipeline

It captures what the student explicitly says:
- primary intent
- goal
- entities
- constraints/preferences
- user/background facts
- conversational signals

The vocabulary is deliberately small and reusable. Institution-specific
entities can be supplied at runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence


# =========================================================
# Public data contract
# =========================================================

@dataclass(frozen=True)
class StudentSituation:
    """Structured representation of one student's message."""

    raw_text: str
    intent: str
    goal: str
    entities: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    user_facts: Mapping[str, Any] = field(default_factory=dict)
    signals: tuple[str, ...] = ()
    confidence: float = 0.0
    requires_clarification: bool = False
    clarification_reason: str = ""


# =========================================================
# Reusable vocabulary
# =========================================================

RESEARCH_INTEREST_TERMS = (
    "robotics",
    "control systems",
    "power systems",
    "power electronics",
    "renewable energy",
    "signal processing",
    "embedded systems",
    "communication systems",
    "rf",
    "microwave",
    "vlsi",
    "visual computing",
    "machine learning",
    "artificial intelligence",
    "data science",
    "flexible electronics",
    "sensors",
    "neuroscience",
    "bio-imaging",
    "materials",
)

DEFAULT_ENTITY_TERMS = (
    "hostel",
    "ph.d",
    "phd",
    "m.tech",
    "mtech",
    "m.s.",
    "m.s",
    "ms by research",
    "m.sc",
    "msc",
    "b.tech",
    "btech",
    "b.sc",
    "bsc",
    "b.a",
    "ba",
    "electrical engineering",
    "computer science",
    "computer science and engineering",
    "mechanical engineering",
    "physics",
    "chemistry",
    "economics",
    "management",
    "artificial intelligence",
    "data science",
    "vlsi",
    "robotics",
    "control systems",
    "power systems",
    "power electronics",
    "communication systems",
    "visual computing",
    "machine learning",
)


# =========================================================
# Regex
# =========================================================

PERCENT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*%"
)

CGPA_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)"
    r"(?:\s*/\s*(?P<scale>\d+(?:\.\d+)?))?"
    r"\s*(?:cgpa|cpi)",
    re.IGNORECASE,
)

DURATION_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
    r"(?P<unit>days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}

BACHELOR_RE = re.compile(
    r"\b(?:"
    r"my\s+|"
    r"i\s+have\s+(?:a\s+)?|"
    r"my\s+degree\s+is\s+"
    r")?"
    r"(?:four[- ]year\s+)?"
    r"(?:bachelor(?:'s)?(?:\s+degree)?|"
    r"b\.?\s*tech|"
    r"b\.?\s*sc|"
    r"b\.?\s*a)"
    r"\b",
    re.IGNORECASE,
)

MASTER_RE = re.compile(
    r"\b(?:"
    r"my\s+|"
    r"i\s+have\s+(?:a\s+)?|"
    r"my\s+degree\s+is\s+"
    r")?"
    r"(?:master(?:'s)?(?:\s+degree)?|"
    r"m\.?\s*tech|"
    r"m\.?\s*sc)"
    r"\b",
    re.IGNORECASE,
)


# =========================================================
# Basic helpers
# =========================================================

def normalize_text(
    text: str,
) -> str:
    """Normalize case, whitespace, and dash variants."""
    value = str(text or "").strip().lower()

    value = (
        value.replace("–", "-")
        .replace("—", "-")
        .replace("\n", " ")
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _has_any(
    text: str,
    phrases: Sequence[str],
) -> bool:
    return any(
        phrase in text
        for phrase in phrases
    )


def _dedupe_preserve_order(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Deduplicate strings without changing first-seen order."""
    seen: set[str] = set()
    output: list[str] = []

    for value in values:

        value = str(value).strip()

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return tuple(output)


# =========================================================
# Intent detection
# =========================================================

def _research_interest_signal(
    text: str,
) -> bool:
    """Detect interest-led research requests."""
    interest_language = _has_any(
        text,
        (
            "my interest",
            "my research interest",
            "i'm interested",
            "i am interested",
            "interested in",
            "i want to work on",
            "i want to focus on",
            "i want to pursue",
            "looking for a research",
        ),
    )

    return (
        interest_language
        and
        _has_any(
            text,
            RESEARCH_INTEREST_TERMS,
        )
    )


def _phd_eligibility_signal(
    text: str,
) -> bool:
    """
    Recognize Ph.D. eligibility/admission situations.

    A user does not have to say the literal word "eligibility".
    These are all valid eligibility signals:
        - "Can I apply for a regular Ph.D.?"
        - "My B.Tech qualifies me for Ph.D."
        - "I have a 7.2 CGPA and am considering Ph.D. admission."
        - "I want regular Ph.D. admission."
    """
    has_phd = (
        "ph.d" in text
        or "phd" in text
        or "doctoral" in text
    )

    if not has_phd:
        return False

    # Direct eligibility / qualification wording.
    if _has_any(
        text,
        (
            "eligib",
            "qualif",
            "requirement",
            "requirements",
            "applicant",
            "applicants",
            "apply for",
            "applying for",
            "can i apply",
            "would i qualify",
            "whether my",
            "my marks",
            "my degree",
            "my background",
            "my qualification",
        ),
    ):
        return True

    # Personal qualification information + an explicit Ph.D. admission
    # target is also an eligibility situation.
    has_personal_qualification = bool(
        re.search(
            r"\bmy\b.*\b(?:degree|background|qualification|marks|score|cgpa|cpi)\b",
            text,
        )
        or re.search(
            r"\bi\s+have\b.*\b(?:b\.?\s*tech|b\.?\s*sc|bachelor|master|cgpa|cpi|\d+(?:\.\d+)?\s*%)\b",
            text,
        )
        or PERCENT_RE.search(text)
        or CGPA_RE.search(text)
    )

    has_admission_target = _has_any(
        text,
        (
            "ph.d admission",
            "ph.d. admission",
            "phd admission",
            "phd. admission",
            "ph.d admissions",
            "ph.d. admissions",
            "phd admissions",
            "phd. admissions",
            "regular ph.d",
            "regular ph.d.",
            "regular phd",
            "regular phd.",
            "considering ph.d",
            "considering ph.d.",
            "considering phd",
            "considering phd.",
            "want regular ph.d",
            "want regular ph.d.",
            "want regular phd",
            "want regular phd.",
            "want to pursue ph.d",
            "want to pursue ph.d.",
            "want to pursue phd",
            "want to pursue phd.",
        ),
    )

    return (
        has_personal_qualification
        and has_admission_target
    )


def _mtech_eligibility_signal(
    text: str,
) -> bool:
    """Recognize M.Tech eligibility situations."""
    has_mtech = (
        "m.tech" in text
        or "mtech" in text
    )

    if not has_mtech:
        return False

    if _has_any(
        text,
        (
            "eligib",
            "qualif",
            "requirement",
            "requirements",
            "applicant",
            "applicants",
            "apply for",
            "applying for",
            "can i apply",
            "would i qualify",
            "whether my",
            "my marks",
            "my degree",
            "my background",
            "my qualification",
        ),
    ):
        return True

    has_personal_qualification = bool(
        re.search(
            r"\bmy\b.*\b(?:degree|background|qualification|marks|score|cgpa|cpi)\b",
            text,
        )
        or re.search(
            r"\bi\s+have\b.*\b(?:b\.?\s*tech|b\.?\s*sc|bachelor|master|cgpa|cpi|\d+(?:\.\d+)?\s*%)\b",
            text,
        )
        or PERCENT_RE.search(text)
        or CGPA_RE.search(text)
    )

    has_admission_target = _has_any(
        text,
        (
            "m.tech admission",
            "mtech admission",
            "considering m.tech",
            "considering mtech",
            "want to pursue m.tech",
            "want to pursue mtech",
        ),
    )

    return has_personal_qualification and has_admission_target



def _select_intent(
    text: str,
) -> str:
    """Select a primary intent using explicit precedence."""
    # -----------------------------------------------------
    # Structured urgent/navigation situations first.
    # -----------------------------------------------------

    if _has_any(
        text,
        (
            "injured",
            "injury",
            "hurt",
            "bleeding",
            "accident",
            "medical emergency",
            "ambulance",
            "unconscious",
            "fainted",
        ),
    ):
        return "emergency"

    if _has_any(
        text,
        (
            "where should i take",
            "where can i take",
            "where do i go",
            "how do i reach",
            "need directions",
            "give me directions",
            "directions to",
            "route to",
            "take me to",
        ),
    ):
        return "navigation"

    # -----------------------------------------------------
    # Eligibility must beat generic admission/program words.
    # -----------------------------------------------------

    if _phd_eligibility_signal(text):
        return "phd_eligibility"

    if _mtech_eligibility_signal(text):
        return "mtech_eligibility"

    # -----------------------------------------------------
    # Research-interest language must beat generic "degree/program".
    # -----------------------------------------------------

    if _research_interest_signal(text):
        return "research"

    # -----------------------------------------------------
    # Explicit comparison situations.
    # -----------------------------------------------------

    if _has_any(
        text,
        (
            "deciding between",
            "choosing between",
            "comparing",
            "difference between",
            "versus",
            " vs ",
        ),
    ):
        if _has_any(
            text,
            (
                "hostel",
                "single occupancy",
                "double occupancy",
                "room",
            ),
        ):
            return "hostel"

        if _has_any(
            text,
            (
                "m.tech",
                "mtech",
                "m.s.",
                "m.s ",
                "ms by research",
            ),
        ):
            return "programs"

    # -----------------------------------------------------
    # Financial assistance as a distinct information domain.
    # -----------------------------------------------------

    if _has_any(
        text,
        (
            "financial assistance",
            "financial aid",
            "scholarship",
            "stipend",
            "funding",
            "assistantship",
        ),
    ):
        return "financial_assistance"

    # -----------------------------------------------------
    # Hostel before generic fees.
    # -----------------------------------------------------

    if _has_any(
        text,
        (
            "hostel",
            "accommodation",
            "occupancy",
            "bedding",
            "share a room",
            "sharing a room",
            "shared room",
            "single room",
            "double room",
        ),
    ):
        return "hostel"

    # -----------------------------------------------------
    # Generic research vocabulary.
    # -----------------------------------------------------

    if _has_any(
        text,
        (
            "research",
            "research area",
            "research areas",
            "research topic",
            "research topics",
            "research work",
            "research direction",
            "research directions",
            "supervisor",
        ),
    ):
        return "research"

    # -----------------------------------------------------
    # Admission / fees / programs / organization.
    # -----------------------------------------------------

    if _has_any(
        text,
        (
            "admission",
            "apply",
            "applying",
            "application",
            "before applying",
            "planning to apply",
            "preparing to apply",
            "shortlisting",
            "shortlist",
        ),
    ):
        return "admission"

    if _has_any(
        text,
        (
            "fee",
            "fees",
            "cost",
            "costs",
            "charge",
            "charges",
            "tuition",
            "rent",
            "expense",
            "expenses",
        ),
    ):
        return "fees"

    if _has_any(
        text,
        (
            "program",
            "programs",
            "programme",
            "programmes",
            "degree program",
            "degree programs",
            "course",
            "courses",
        ),
    ):
        return "programs"

    if _has_any(
        text,
        (
            "facility",
            "facilities",
            "amenity",
            "amenities",
            "campus",
            "library",
            "lab",
            "laboratory",
            "sports",
        ),
    ):
        return "facilities"

    if _has_any(
        text,
        (
            "department",
            "departments",
            "school",
            "schools",
            "centre",
            "centres",
            "center",
            "centers",
        ),
    ):
        return "department_or_school"

    return "general_information"


# =========================================================
# Goal detection
# =========================================================

GOAL_PATTERNS = (
    (
        "determine_eligibility",
        (
            "am i eligible",
            "can i apply",
            "can i get admitted",
            "would i qualify",
            "do i qualify",
            "would i be eligible",
            "qualify",
            "qualified",
            "qualification",
            "eligibility",
            "eligible",
            "qualifies me",
            "meet the requirement",
            "meet the requirements",
            "meets the requirement",
            "meets the criteria",
            "whether my marks meet",
            "whether my qualification",
            "whether my background",
            "can my background work",
            "does my background",
            "matches the requirement",
            "matches the criteria",
            "my marks",
        ),
    ),
    (
        "minimize_cost",
        (
            "save money",
            "save cost",
            "save on cost",
            "reduce the cost",
            "reduces the cost",
            "reduce cost",
            "lower the cost",
            "lower cost",
            "cheapest",
            "less expensive",
            "keep the cost low",
            "cost-effective",
            "budget-friendly",
            "to save money",
        ),
    ),
    (
        "compare_options",
        (
            "compare",
            "comparing",
            "difference between",
            "deciding between",
            "choosing between",
            "rather than",
            "versus",
            " vs ",
            "which one fits",
            "which option fits",
            "which option suits",
            "single and double occupancy",
        ),
    ),
    (
        "choose_option",
        (
            "help me choose",
            "help me decide",
            "trying to decide",
            "i'm deciding",
            "i am deciding",
            "which path should i",
            "which option should i",
            "which would suit",
            "which would fit",
        ),
    ),
    (
        "find_relevant_research",
        (
            "research interest",
            "research interests",
            "research direction",
            "research directions",
            "research topic",
            "research topics",
            "research area",
            "research areas",
            "research fit",
            "research focus",
            "area that suits me",
            "research before approaching",
            "potential supervisor",
        ),
    ),
    (
        "estimate_cost",
        (
            "estimate",
            "expected expense",
            "expected cost",
            "roughly how much",
            "how much would i",
            "calculate",
            "budget for",
            "cost me",
        ),
    ),
    (
        "plan_application",
        (
            "planning to apply",
            "preparing to apply",
            "before applying",
            "shortlist",
            "shortlisting",
            "application decision",
        ),
    ),
    (
        "get_contact",
        (
            "contact number",
            "phone number",
            "whom should i contact",
            "who should i contact",
            "contact",
            "email",
        ),
    ),
    (
        "locate_service",
        (
            "where should i take",
            "where can i take",
            "where do i go",
            "where should i go",
            "take him",
            "take her",
        ),
    ),
    (
        "understand_requirements",
        (
            "requirements",
            "requirement",
            "conditions",
            "criteria",
            "rules",
            "what should i check",
            "what do i need",
        ),
    ),
    (
        "find_information",
        (
            "find out",
            "understand",
            "explain",
            "tell me",
            "help me understand",
            "need to know",
            "want to know",
            "i need information",
            "i want to know",
        ),
    ),
)


def _select_goal(
    text: str,
    intent: str,
) -> str:
    """Select the most specific goal."""
    candidates: list[tuple[int, int, str]] = []

    for index, (goal, patterns) in enumerate(
        GOAL_PATTERNS
    ):
        score = 0

        for pattern in patterns:

            if pattern not in text:
                continue

            words = len(
                pattern.split()
            )

            score += (
                4
                if words >= 3
                else 2
            )

        if score:
            candidates.append(
                (
                    score,
                    -index,
                    goal,
                )
            )

    if candidates:
        candidates.sort(
            reverse=True
        )
        return candidates[0][2]

    if intent in {
        "phd_eligibility",
        "mtech_eligibility",
    }:
        return "determine_eligibility"

    if intent == "research":
        return (
            "find_relevant_research"
            if _research_interest_signal(text)
            else "find_information"
        )

    if intent in {
        "navigation",
        "emergency",
    }:
        return "locate_service"

    if intent == "admission":
        return "plan_application"

    return "find_information"


# =========================================================
# Entity extraction
# =========================================================

def _normalize_entity_surface(
    term: str,
) -> str:
    """
    Normalize an entity for comparison without changing the main query text.

    Abbreviation punctuation is treated as optional so equivalent forms such
    as "M.S.", "M.S", and "MS" compare consistently. Spaces and all other
    characters retain their semantic role.

    This helper is intentionally generic: entity vocabulary may be supplied
    by another institution at runtime.
    """
    value = normalize_text(
        term
    )

    # Remove periods that belong to letter-based abbreviations.
    # Examples:
    #   "m.s."            -> "ms"
    #   "m.s by research" -> "ms by research"
    #   "m.tech."         -> "mtech"
    value = re.sub(
        r"(?<=[a-z])\.(?=[a-z])",
        "",
        value,
    )
    value = re.sub(
        r"(?<=[a-z])\.(?=\s|$)",
        "",
        value,
    )

    return value


def _canonical_entity(
    term: str,
) -> str:
    """
    Return the canonical public representation for an equivalent entity.

    Canonicalization is performed on the punctuation-tolerant comparison
    form, so configured aliases such as "M.S. by Research" and
    "MS by Research" can resolve to the same public entity.

    The vocabulary remains configurable; this function contains only generic
    alias normalization for the default academic vocabulary.
    """
    normalized = _normalize_entity_surface(
        term
    )

    aliases = {
        "phd": "phd",
        "mtech": "m.tech",
        "ms": "m.s.",
        "ms by research": "m.s. by research",
        "msc": "m.sc",
        "btech": "b.tech",
        "bsc": "b.sc",
        "ba": "b.a",
    }

    return aliases.get(
        normalized,
        term,
    )


def _extract_entities(
    text: str,
    entity_terms: Sequence[str] | None,
) -> tuple[str, ...]:
    """
    Extract explicitly named reusable/configured entities.

    Matching is:
        - boundary-aware
        - punctuation-tolerant for abbreviation-style entities
        - longest-match-first for overlapping candidates
        - non-destructive for separate entities

    Examples:
        "M.Sc."              -> m.sc
        "M.S."               -> m.s.
        "M.S. by Research"  -> m.s. by research
        "MS by Research"    -> m.s. by research
        "M.Sc. and M.S."    -> m.sc, m.s.

    A shorter entity is discarded only when it overlaps the span of a longer
    configured entity. This prevents "m.s." from being extracted inside
    "m.s. by research" while still preserving a separate "m.s." elsewhere.

    The entity vocabulary can be supplied by ``entity_terms``. The matching
    mechanism is institution-agnostic.
    """
    terms = (
        tuple(entity_terms)
        if entity_terms is not None
        else DEFAULT_ENTITY_TERMS
    )

    match_text = _normalize_entity_surface(
        text
    )

    # Candidate:
    #   (start, end, canonical_entity, span_length)
    candidates: list[tuple[int, int, str, int]] = []

    for term in terms:
        normalized_term = normalize_text(
            term
        )

        if not normalized_term:
            continue

        match_key = _normalize_entity_surface(
            normalized_term
        )

        if not match_key:
            continue

        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(match_key)}"
            rf"(?![a-z0-9])"
        )

        for match in re.finditer(
            pattern,
            match_text,
        ):
            start, end = match.span()

            candidates.append(
                (
                    start,
                    end,
                    _canonical_entity(term),
                    end - start,
                )
            )

    # Evaluate longer candidates first within overlapping regions.
    candidates.sort(
        key=lambda item: (
            item[0],
            -item[3],
        )
    )

    selected: list[tuple[int, int, str]] = []

    for start, end, canonical, _ in candidates:
        overlap = any(
            start < existing_end
            and end > existing_start
            for existing_start, existing_end, _ in selected
        )

        if overlap:
            continue

        selected.append(
            (
                start,
                end,
                canonical,
            )
        )

    selected.sort(
        key=lambda item: item[0]
    )

    return _dedupe_preserve_order(
        item[2]
        for item in selected
    )


# =========================================================
# User facts
# =========================================================

def _extract_degree_fact(
    text: str,
) -> str | None:
    """
    Extract the student's current/background degree.

    Target mentions such as:
        "apply for M.Tech"
    do not overwrite:
        "my bachelor's"
    """
    # Explicit bachelor ownership.
    if (
        _has_any(
            text,
            (
                "my bachelor's",
                "my bachelors",
                "my bachelor degree",
                "my bachelor's degree",
                "i have a bachelor's",
                "i have a bachelors",
                "i have a bachelor",
                "my b.tech",
                "my btech",
                "i have a b.tech",
                "i have b.tech",
                "i have a btech",
                "i have btech",
                "my b.sc",
                "my bsc",
                "i have a b.sc",
                "i have b.sc",
                "my b.a",
                "my ba ",
            ),
        )
        or re.search(
            r"\bi\s+have\s+(?:a\s+)?"
            r"(?:four[- ]year\s+)?"
            r"(?:b\.?\s*tech|b\.?\s*sc|b\.?\s*a)\b",
            text,
        )
    ):
        return "bachelors_degree"

    # Explicit master's ownership.
    if (
        _has_any(
            text,
            (
                "my master's",
                "my masters",
                "my master degree",
                "my master's degree",
                "i have a master's",
                "i have a masters",
                "i have a master",
                "my m.tech",
                "my mtech",
                "i have a m.tech",
                "i have m.tech",
                "i have a mtech",
                "i have mtech",
                "my m.sc",
                "my msc",
                "i have a m.sc",
                "i have m.sc",
            ),
        )
    ):
        return "masters_degree"

    # General "my degree/background" phrasing.
    if (
        "my degree" in text
        or "my background" in text
        or "my qualification" in text
    ):
        if re.search(
            r"\b(?:bachelor|b\.?\s*tech|b\.?\s*sc|b\.?\s*a)\b",
            text,
        ):
            return "bachelors_degree"

        if re.search(
            r"\b(?:master|m\.?\s*tech|m\.?\s*sc)\b",
            text,
        ):
            return "masters_degree"

    return None


def _extract_user_facts(
    text: str,
) -> dict[str, Any]:
    """Extract explicit personal/background facts only."""
    facts: dict[str, Any] = {}

    degree = _extract_degree_fact(
        text
    )

    if degree:
        facts["degree"] = degree

    if (
        (
            "four-year" in text
            or "four year" in text
        )
        and
        _has_any(
            text,
            (
                "degree",
                "bachelor",
                "b.tech",
                "btech",
                "b.sc",
                "bsc",
                "b.a",
                "ba ",
            ),
        )
    ):
        facts[
            "degree_duration_years"
        ] = 4

    percentages = tuple(
        float(match.group("value"))
        for match in PERCENT_RE.finditer(
            text
        )
    )

    if percentages:
        facts[
            "percentages"
        ] = percentages

    cgpas = tuple(
        float(match.group("value"))
        for match in CGPA_RE.finditer(
            text
        )
    )

    if cgpas:
        facts[
            "cgpa_values"
        ] = cgpas

    category_patterns = (
        (
            "SC",
            (
                "sc category",
                "category is sc",
                "i am sc",
                "i'm sc",
            ),
        ),
        (
            "ST",
            (
                "st category",
                "category is st",
                "i am st",
                "i'm st",
            ),
        ),
        (
            "OBC",
            (
                "obc category",
                "category is obc",
                "i am obc",
                "i'm obc",
            ),
        ),
        (
            "EWS",
            (
                "ews category",
                "category is ews",
                "i am ews",
                "i'm ews",
            ),
        ),
        (
            "GEN",
            (
                "general category",
                "gen category",
                "category is general",
            ),
        ),
        (
            "PwD",
            (
                "pwd category",
                "person with disability",
            ),
        ),
    )

    for category, patterns in category_patterns:

        if _has_any(
            text,
            patterns,
        ):
            facts["category"] = category
            break

    if _has_any(
        text,
        (
            "work full-time",
            "working full-time",
            "full-time employee",
        ),
    ):
        facts[
            "employment_status"
        ] = "full_time"

    elif _has_any(
        text,
        (
            "work part-time",
            "working part-time",
            "part-time employee",
        ),
    ):
        facts[
            "employment_status"
        ] = "part_time"

    elif _has_any(
        text,
        (
            "not employed",
            "unemployed",
        ),
    ):
        facts[
            "employment_status"
        ] = "not_employed"

    return facts


# =========================================================
# Constraints
# =========================================================

def _extract_stay_duration(
    text: str,
) -> dict[str, Any] | None:
    """Extract one explicit duration."""
    match = DURATION_RE.search(
        text
    )

    if not match:
        return None

    raw_unit = match.group(
        "unit"
    ).lower()

    if raw_unit.startswith("year"):
        normalized_unit = "years"
    elif raw_unit.startswith("month"):
        normalized_unit = "months"
    elif raw_unit.startswith("week"):
        normalized_unit = "weeks"
    else:
        normalized_unit = "days"

    raw_value = match.group("value").lower()

    value = (
        NUMBER_WORDS.get(raw_value)
        if raw_value in NUMBER_WORDS
        else float(raw_value)
    )

    return {
        "value": float(value),
        "unit": normalized_unit,
    }


def _extract_occupancy(
    text: str,
) -> str | None:
    """Extract explicit or strongly implied occupancy preference."""
    if _has_any(
        text,
        (
            "single occupancy",
            "single room",
            "own room",
            "room for myself",
            "private room",
            "room to myself",
        ),
    ):
        return "single"

    if _has_any(
        text,
        (
            "double occupancy",
            "double room",
            "share a room",
            "sharing a room",
            "shared room",
            "shared hostel room",
            "willing to share",
            "comfortable sharing",
            "comfortable with sharing",
            "sharing",
        ),
    ):
        return "double"

    return None


def _extract_bedding(
    text: str,
) -> str | None:
    """Extract bedding preference; negative wording is checked first."""
    if _has_any(
        text,
        (
            "don't need bedding",
            "do not need bedding",
            "doesn't need bedding",
            "does not need bedding",
            "without bedding",
            "no bedding",
        ),
    ):
        return "without"

    if _has_any(
        text,
        (
            "need bedding",
            "with bedding",
            "bedding included",
            "bedding along with",
            "with 1 bedding set",
            "one bedding set",
        ),
    ):
        return "with"

    return None


def _extract_budget(
    text: str,
) -> float | None:
    """Extract an explicit monetary budget."""
    patterns = (
        r"(?:budget|under|below|within)\s*"
        r"(?:of\s*)?"
        r"(?:₹|rs\.?|inr)?\s*"
        r"(\d[\d,]*(?:\.\d+)?)",
        r"(?:₹|rs\.?|inr)\s*"
        r"(\d[\d,]*(?:\.\d+)?)"
        r"\s*(?:budget|maximum|max)",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            return float(
                match.group(
                    1
                ).replace(
                    ",",
                    "",
                )
            )

    return None


def _extract_constraints(
    text: str,
    intent: str,
    goal: str,
) -> dict[str, Any]:
    """Extract explicit decision constraints."""
    constraints: dict[str, Any] = {}

    if intent == "hostel":

        duration = _extract_stay_duration(
            text
        )

        if duration:
            constraints[
                "stay_duration"
            ] = duration

    occupancy = _extract_occupancy(
        text
    )

    if occupancy:
        constraints[
            "occupancy"
        ] = occupancy

    bedding = _extract_bedding(
        text
    )

    if bedding:
        constraints[
            "bedding"
        ] = bedding

    budget = _extract_budget(
        text
    )

    if budget is not None:
        constraints[
            "budget"
        ] = budget

    interests = _dedupe_preserve_order(
        term
        for term in RESEARCH_INTEREST_TERMS
        if term in text
    )

    if interests:
        constraints[
            "research_interests"
        ] = interests

    if _has_any(
        text,
        (
            "save money",
            "save cost",
            "save on cost",
            "reduce the cost",
            "reduces the cost",
            "reduce cost",
            "lower the cost",
            "lower cost",
            "cheapest",
            "less expensive",
            "keep the cost low",
            "to save money",
        ),
    ):
        constraints[
            "cost_sensitive"
        ] = True

    if _has_any(
        text,
        (
            "research over coursework",
            "prefer research",
            "prefer a research-oriented",
            "research-oriented path",
            "purely coursework",
        ),
    ):
        constraints[
            "research_preference"
        ] = True

    if _has_any(
        text,
        (
            "interdisciplinary",
            "interdisciplinary exposure",
        ),
    ):
        constraints[
            "interdisciplinary_preference"
        ] = True

    if (
        goal == "compare_options"
        and
        _has_any(
            text,
            (
                "between",
                "versus",
                " vs ",
                "difference",
                "single and double",
            ),
        )
    ):
        constraints[
            "comparison_requested"
        ] = True

    return constraints


# =========================================================
# Signals
# =========================================================

def _extract_signals(
    text: str,
    intent: str,
    goal: str,
) -> tuple[str, ...]:
    """Preserve useful conversational framing."""
    signals: list[str] = []

    for phrase in (
        "i am",
        "i'm",
        "i have",
        "i want",
        "i need",
        "i prefer",
        "i'm considering",
        "i am considering",
        "i'm planning",
        "i am planning",
        "i'm deciding",
        "i am deciding",
        "my goal",
        "my interest",
        "my background",
        "suppose",
        "if",
        "because",
        "rather than",
        "before applying",
        "after admission",
        "to save money",
    ):
        if phrase in text:
            signals.append(
                phrase
            )

    signals.extend(
        (
            f"intent:{intent}",
            f"goal:{goal}",
        )
    )

    return _dedupe_preserve_order(
        signals
    )


# =========================================================
# Clarification / confidence
# =========================================================

def _clarification_requirement(
    intent: str,
    goal: str,
    facts: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> tuple[bool, str]:
    """Flag cases where missing user information could matter later."""
    if (
        intent in {
            "phd_eligibility",
            "mtech_eligibility",
        }
        and
        goal == "determine_eligibility"
        and
        not facts
    ):
        return (
            True,
            "Eligibility is being considered without explicit "
            "personal qualification facts.",
        )

    if (
        intent == "hostel"
        and
        goal in {
            "estimate_cost",
            "minimize_cost",
        }
        and
        not any(
            key in constraints
            for key in (
                "stay_duration",
                "occupancy",
                "bedding",
                "budget",
            )
        )
    ):
        return (
            True,
            "The hostel decision lacks a stay, occupancy, bedding, "
            "or budget constraint.",
        )

    if (
        goal == "compare_options"
        and
        not facts
        and
        not constraints
    ):
        return (
            True,
            "A comparison was requested without explicit decision criteria.",
        )

    return False, ""


def _confidence(
    text: str,
    intent: str,
    goal: str,
    entities: Sequence[str],
    facts: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> float:
    """Estimate structural extraction confidence."""
    score = 0.25

    if intent != "general_information":
        score += 0.20

    if goal != "find_information":
        score += 0.15

    if entities:
        score += min(
            0.15,
            0.05 * len(entities),
        )

    if facts:
        score += min(
            0.15,
            0.05 * len(facts),
        )

    if constraints:
        score += min(
            0.15,
            0.05 * len(constraints),
        )

    if len(text.split()) <= 5:
        score -= 0.10

    return round(
        max(
            0.0,
            min(
                1.0,
                score,
            ),
        ),
        3,
    )


# =========================================================
# Public functions
# =========================================================

def understand_student_situation(
    text: str,
    *,
    entity_terms: Sequence[str] | None = None,
) -> StudentSituation:
    """Interpret one natural-language student message."""
    raw_text = str(
        text or ""
    ).strip()

    normalized = normalize_text(
        raw_text
    )

    if not normalized:
        return StudentSituation(
            raw_text="",
            intent="general_information",
            goal="find_information",
            confidence=0.0,
            requires_clarification=True,
            clarification_reason="Empty user message.",
        )

    intent = _select_intent(
        normalized
    )

    goal = _select_goal(
        normalized,
        intent,
    )

    entities = _extract_entities(
        normalized,
        entity_terms,
    )

    facts = _extract_user_facts(
        normalized
    )

    constraints = _extract_constraints(
        normalized,
        intent,
        goal,
    )

    signals = _extract_signals(
        normalized,
        intent,
        goal,
    )

    (
        requires_clarification,
        clarification_reason,
    ) = _clarification_requirement(
        intent,
        goal,
        facts,
        constraints,
    )

    confidence = _confidence(
        normalized,
        intent,
        goal,
        entities,
        facts,
        constraints,
    )

    return StudentSituation(
        raw_text=raw_text,
        intent=intent,
        goal=goal,
        entities=entities,
        constraints=constraints,
        user_facts=facts,
        signals=signals,
        confidence=confidence,
        requires_clarification=requires_clarification,
        clarification_reason=clarification_reason,
    )


def situation_to_dict(
    situation: StudentSituation,
) -> dict[str, Any]:
    """Serialize StudentSituation into a JSON-safe dictionary."""
    result = asdict(
        situation
    )

    result["entities"] = list(
        situation.entities
    )
    result["signals"] = list(
        situation.signals
    )
    result["constraints"] = dict(
        situation.constraints
    )
    result["user_facts"] = dict(
        situation.user_facts
    )

    return result
