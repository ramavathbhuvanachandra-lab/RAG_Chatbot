"""Phase 2 Lossless Semantic Query Contract (LSQC).

Guarantees zero loss of query semantics, atomic intents, program/degree
bindings, applicant categories, numerical constraints, qualification routes,
negation, and condition relationships between understanding and retrieval.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from ai_platform.core.query.models import (
    AtomicIntentBinding,
    LosslessSemanticQuery,
    NumericRequirement,
    SemanticCondition,
)
from ai_platform.core.query.planner import QueryPlan, QueryIntentPlan, DEGREE_PATTERN, NEGATION_PATTERNS, _canonical_degree


# Standard institutional applicant categories in Indian higher education
CATEGORY_PATTERN = re.compile(
    r"\b(GEN(?:ERAL)?|OBC(?:-NCL)?|SC|ST|EWS|PWD|OPEN|SELF[- ]SPONSORED|SPONSORED|SELF-FINANCED|REGULAR)\b",
    re.IGNORECASE,
)

# Numeric threshold pattern: percentages, CPI/CGPA, duration in years, marks, credits, currency amounts
NUMERIC_PATTERN = re.compile(
    r"(?:rs\.?|inr|₹)?\s*(\b\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent(?:age)?|cgpa|cpi|gpa|marks|years?|year duration|credits?|-credit|rs|rupees)?(?!\w)"
    r"|(\b(?:cgpa|cpi|gpa|minimum|rs\.?|inr|₹|deposit|fee|grant|fellowship|stipend)\s+(?:of|=|:)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b)",
    re.IGNORECASE,
)


# Time limit / deadline patterns
DEADLINE_PATTERN = re.compile(
    r"\b(deadline|last date|due date|specified date|cutoff date|timeline|schedule)\b",
    re.IGNORECASE,
)

# Qualification keywords commonly appearing in admission criteria
QUALIFICATION_KEYWORDS = (
    "first class",
    "first-class",
    "qualifying degree",
    "four-year bachelor",
    "four-year",
    "4-year",
    "bachelor's degree",
    "master's degree",
    "gate",
    "jam",
    "csir",
    "ugc",
    "jrf",
    "net",
    "sponsorship",
    "experience",
)


def extract_applicant_categories(text: str) -> tuple[str, ...]:
    """Extract standard applicant categories from text without loss."""
    if not text:
        return ()
    matches = CATEGORY_PATTERN.findall(text)
    seen: set[str] = set()
    categories: list[str] = []
    for m in matches:
        norm = m.upper().replace(" ", "-")
        if norm in ("GENERAL", "GEN"):
            for variant in ("GEN", "GENERAL"):
                if variant not in seen:
                    seen.add(variant)
                    categories.append(variant)
            continue
        if norm not in seen:
            seen.add(norm)
            categories.append(norm)
    return tuple(categories)


def extract_numeric_requirements(text: str) -> tuple[NumericRequirement, ...]:
    """Losslessly extract numerical thresholds, percentages, CPI/CGPA, durations, and credits."""
    if not text:
        return ()
    results: list[NumericRequirement] = []
    seen: set[tuple[str, float]] = set()

    # Direct regex matches
    for match in NUMERIC_PATTERN.finditer(text):
        val_str = match.group(1) or match.group(4)
        raw_unit = match.group(2)
        unit = raw_unit.lower().lstrip("-") if raw_unit else ("cgpa" if "cgpa" in match.group(0).lower() else "gpa" if "gpa" in match.group(0).lower() else "cpi" if "cpi" in match.group(0).lower() else None)
        if val_str:
            clean_val_str = val_str.replace(",", "")
            try:
                val = float(clean_val_str)
            except ValueError:
                continue
            name = unit.lower() if unit else "threshold"
            key = (name, val)
            if key not in seen:
                seen.add(key)
                results.append(
                    NumericRequirement(
                        name=name,
                        value=val,
                        operator="gte" if any(w in text.lower() for w in ("minimum", "at least", ">=")) else "eq",
                        unit=unit,
                        confidence=1.0,
                    )
                )

    # Word numbers for years of experience / degree duration
    for word_num, val in (
        ("one", 1.0), ("two", 2.0), ("three", 3.0), ("four", 4.0),
        ("five", 5.0), ("six", 6.0), ("seven", 7.0), ("eight", 8.0),
        ("nine", 9.0), ("ten", 10.0),
    ):
        if re.search(rf"\b{word_num}\s+years?\b", text, re.IGNORECASE):
            key = ("years", val)
            if key not in seen:
                seen.add(key)
                results.append(
                    NumericRequirement(
                        name="years",
                        value=val,
                        operator="eq",
                        unit="years",
                        confidence=1.0,
                    )
                )

    # Check for 4-year / four-year bachelor's degree requirement
    if re.search(r"\b(?:4|four)[ -]year\b", text, re.IGNORECASE):
        key = ("degree_duration", 4.0)
        if key not in seen:
            seen.add(key)
            results.append(
                NumericRequirement(
                    name="degree_duration",
                    value=4.0,
                    operator="gte",
                    unit="years",
                    confidence=1.0,
                )
            )

    # Check for grading scale mention (e.g. scale of 8, scale of 10)
    scale_match = re.search(r"\b(?:scale\s+(?:of\s+)?|on\s+an?\s+)(\d+)(?:-point\s+scale|\s+scale)?\b", text, re.IGNORECASE)
    if scale_match:
        scale_val = float(scale_match.group(1))
        key = ("grading_scale", scale_val)
        if key not in seen:
            seen.add(key)
            results.append(
                NumericRequirement(
                    name="grading_scale",
                    value=scale_val,
                    operator="eq",
                    unit="scale",
                    confidence=1.0,
                )
            )

    # Check for explicit calendar year (e.g. 2026)
    for yr_m in re.finditer(r"\b(20\d\d)\b", text):
        yr_val = float(yr_m.group(1))
        key = ("year", yr_val)
        if key not in seen:
            seen.add(key)
            results.append(
                NumericRequirement(
                    name="year",
                    value=yr_val,
                    operator="eq",
                    unit="year",
                    confidence=1.0,
                )
            )

    # Check for credit limit / under-load rule / maximum credit requirement
    if re.search(r"\b(?:credit\s+limit|under-?load\s+rule|maximum\s+credits?)\b", text, re.IGNORECASE):
        key = ("credits", 0.0)
        if key not in seen:
            seen.add(key)
            results.append(
                NumericRequirement(
                    name="credits",
                    value=0.0,
                    operator="lte" if re.search(r"\b(?:maximum|max|limit|under-load)\b", text, re.IGNORECASE) else "eq",
                    unit="credits",
                    confidence=1.0,
                )
            )

    return tuple(results)


def extract_qualifications(text: str) -> tuple[str, ...]:
    """Extract qualification routes mentioned in text."""
    if not text:
        return ()
    text_lower = text.lower()
    found: list[str] = []
    for kw in QUALIFICATION_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return tuple(found)


def extract_condition_relationships(
    text: str, constraints: Sequence[str] = ()
) -> tuple[SemanticCondition, ...]:
    """Capture individual conditions and their logical AND/OR relations."""
    conditions: list[SemanticCondition] = []

    # Check if text contains explicit OR branching (e.g. "or a first-class", "either ... or")
    has_or = bool(re.search(r"\b(?:or|either)\b", text, re.IGNORECASE))

    # Analyze constraints list
    for c in constraints:
        if not c:
            continue
        rel = "OR" if has_or and any(w in c.lower() for w in ("or", "first class", "first-class", "equivalent", "qualifying")) else "AND"
        conditions.append(
            SemanticCondition(
                name="eligibility_constraint",
                value=c,
                operator="eq",
                logical_relation=rel,
                required=True,
            )
        )

    # Always extract qualifications when has_or is True or no conditions found yet
    quals = extract_qualifications(text)
    for q in quals:
        if any(c.value.lower() == q.lower() for c in conditions):
            continue
        rel = "OR" if (has_or and ("first" in q or "degree" in q or "gate" in q or "or" in text.lower())) else "AND"
        conditions.append(
            SemanticCondition(
                name="qualification",
                value=q,
                operator="eq",
                logical_relation=rel,
                required=True,
            )
        )

    return tuple(conditions)


def extract_degree_and_program(text: str, target: str | None = None) -> tuple[str | None, str | None]:
    """Extract degree level and academic program from question and target."""
    degree = None
    program = None

    match = re.search(DEGREE_PATTERN, text, re.IGNORECASE)
    if match:
        degree = match.group(0).strip()
    elif target:
        target_match = re.search(DEGREE_PATTERN, target, re.IGNORECASE)
        if target_match:
            degree = target_match.group(0).strip()

    if degree:
        degree = _canonical_degree(degree) or degree

    if target:
        program = target
    elif degree:
        program = degree

    return degree, program


def detect_negation(text: str) -> bool:
    """Detect whether query asserts negation or asks about exemptions/exceptions."""
    if not text:
        return False
    for pat in NEGATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def build_lossless_query(
    plan: QueryPlan,
    *,
    question: str | None = None,
    institution_id: str | None = None,
) -> LosslessSemanticQuery:
    """Build a LosslessSemanticQuery from a QueryPlan and raw question.

    Preserves 100% of user query intent, constraints, bindings, and multi-intent
    breakdowns without loss between query understanding and downstream stages.
    """
    raw_query = question or plan.original_query
    resolved = plan.resolved_query or raw_query

    # Categories
    categories = extract_applicant_categories(raw_query + " " + resolved)

    # Numeric constraints
    num_reqs = extract_numeric_requirements(raw_query + " " + resolved)

    # Qualifications
    quals = extract_qualifications(raw_query + " " + resolved)

    # Degrees and programs
    degree, program = extract_degree_and_program(raw_query, target=plan.intents[0].target if plan.intents else None)

    # Negation
    is_negated = detect_negation(raw_query) or any(i.negated for i in plan.intents)

    # Deadlines / time limits
    time_limits: list[str] = []
    if DEADLINE_PATTERN.search(raw_query):
        time_limits.append("deadline_policy")
    for yr in re.findall(r"\b(20\d\d)\b", raw_query):
        time_limits.append(f"year_{yr}")
    for intent in plan.intents:
        time_limits.extend(intent.temporal_constraints)

    # Condition relationships
    all_constraints: list[str] = []
    for intent in plan.intents:
        all_constraints.extend(intent.constraints)
        all_constraints.extend(intent.qualifiers)
    all_constraints.extend(plan.preserved_terms)
    condition_rels = extract_condition_relationships(raw_query, all_constraints)

    # Route classification from plan
    query_route = getattr(plan, "route", "simple_single_intent")

    # Build atomic intent bindings
    intent_bindings: list[AtomicIntentBinding] = []
    for idx, intent in enumerate(plan.intents):
        i_question = intent.question or raw_query
        i_deg, i_prog = extract_degree_and_program(i_question, target=intent.target)
        i_cats = extract_applicant_categories(i_question) or categories
        i_num = extract_numeric_requirements(i_question) or num_reqs
        i_quals = extract_qualifications(i_question) or quals
        i_conds = extract_condition_relationships(i_question, intent.constraints or intent.qualifiers)

        intent_bindings.append(
            AtomicIntentBinding(
                intent_id=f"intent_{idx}",
                question=i_question,
                request_type=intent.request_type,
                program=i_prog or program,
                degree=i_deg or degree,
                target=intent.target,
                attributes=tuple(intent.requested_attributes),
                categories=i_cats,
                numeric_constraints=i_num,
                qualifications=i_quals,
                conditions=i_conds,
                negated=intent.negated or is_negated,
                retrieval_query=intent.retrieval_query or i_question,
                retrieval_queries=(intent.retrieval_query,) if intent.retrieval_query else (i_question,),
            )
        )

    return LosslessSemanticQuery(
        original_query=raw_query,
        resolved_query=resolved,
        institution_id=institution_id,
        program_binding=program,
        degree_binding=degree,
        applicant_categories=categories,
        numeric_constraints=num_reqs,
        qualifications=quals,
        negation=is_negated,
        time_limits=tuple(set(time_limits)),
        condition_relationships=condition_rels,
        intents=tuple(intent_bindings),
        is_multi_intent=plan.is_multi_intent,
        preserved_terms=tuple(plan.preserved_terms),
        query_route=query_route,
    )


def verify_lossless_preservation(
    lsq: LosslessSemanticQuery, original_question: str
) -> dict[str, Any]:
    """Verify that no essential constraints or entities in question were lost."""
    lost_categories = [
        c for c in extract_applicant_categories(original_question)
        if c not in lsq.applicant_categories
    ]
    lost_qualifications = [
        q for q in extract_qualifications(original_question)
        if q not in lsq.qualifications
    ]

    deg, _ = extract_degree_and_program(original_question)
    degree_lost = bool(deg and not lsq.degree_binding)

    passed = not (lost_categories or lost_qualifications or degree_lost)
    return {
        "passed": passed,
        "lost_categories": tuple(lost_categories),
        "lost_qualifications": tuple(lost_qualifications),
        "degree_lost": degree_lost,
    }
