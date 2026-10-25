"""Three-Route Hybrid Query Understanding Architecture for Institutional RAG.

Provides three principled execution paths:
- Route A: Deterministic Python fast path (clear, standalone factual queries, course codes, facility lookups).
- Route B: Lightweight hybrid processing without a generative LLM (contextual follow-ups, separable multi-intent, comparisons).
- Route C: Existing LLM-based query planner (genuinely ambiguous queries, complex interdependent narratives, low-confidence fallbacks).

Guarantees:
- Zero generative LLM calls for Route A and Route B.
- Exactly one LLM call for Route C.
- Strict preservation of the hybrid retrieval contract (QueryPlan, Query, SemanticQueryFrame, LosslessSemanticQuery).
- Detailed per-turn sub-stage timing and route selection diagnostics.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ai_platform.core.query.models import (
    GuardDecision,
    GuardResult,
    QueryRoute,
)
from ai_platform.core.query.planner import (
    QueryPlan,
    QueryIntentPlan,
)
from ai_platform.core.query.guard import evaluate_security_and_scope
from ai_platform.core.query.planner import (
    _infer_target,
    _requested_attributes,
    _constraints,
    _preserved,
    _safe_retrieval_query,
    _objective_segments,
    _comparison_targets,
    _comparison_attribute,
    _compact_query,
    _repair_scope,
    _relevance,
    _deterministic_intents,
    _deterministic_build_query_plan,
    _intent_semantic_bucket,
    _merge_intents_for_slot,
    _dedupe,
    _clean,
    _cf,
    retrieval_queries_from_plan,
    intent_units_from_plan,
    plan_to_query,
    plan_to_frame,
    should_stop_pipeline,
)
from ai_platform.core.query.lsqc import build_lossless_query


@dataclass(frozen=True)
class HybridRouteDecision:
    """Deterministic route decision emitted by the fast route selector."""
    route: str  # "A", "B", or "C"
    reason: str
    confidence: float
    is_multi_intent: bool
    needs_clarification: bool = False
    clarification_message: str | None = None


@dataclass(frozen=True)
class HybridRouteExecution:
    """Full execution payload containing the authoritative QueryPlan and diagnostics."""
    plan: QueryPlan
    selected_route: str          # Final route executed: "A", "B", or "C"
    initial_route: str           # Route selected before any escalation: "A", "B", or "C"
    route_reason: str            # Human-readable justification
    escalation_happened: bool    # True if escalated from A->B or B->C
    selector_time_seconds: float
    route_time_seconds: float
    total_qu_time_seconds: float
    qu_llm_calls: int            # 0 for Route A and B, 1 for Route C
    operation_timings: dict[str, float]
    pipeline_stop: bool
    safe_response: str | None
    stop_reason: str


def _infer_department_or_topic_target(text: str) -> str | None:
    low = text.casefold()
    dept_map = (
        (r"\b(?:drone\s+and\s+anti[- ]drone|dadt)\b", "Drone and Anti-Drone Technologies"),
        (r"\baerospace\b", "Aerospace Engineering"),
        (r"\b(?:materials\s+engineering|materials)\b", "Materials Engineering"),
        (r"\b(?:electronics\s+engineering|electronics|ece)\b", "Electronics Engineering"),
        (r"\b(?:mathematics\s+and\s+computing|mathematics|maths?)\b", "Mathematics and Computing"),
        (r"\b(?:computer\s+science|cse)\b", "Computer Science and Engineering"),
        (r"\b(?:mechanical\s+engineering|mechanical)\b", "Mechanical Engineering"),
        (r"\b(?:electrical\s+engineering)\b", "Electrical Engineering"),
        (r"\b(?:chemical\s+engineering|chemical)\b", "Chemical Engineering"),
        (r"\b(?:civil\s+engineering|civil)\b", "Civil and Infrastructure Engineering"),
        (r"\b(?:biosciences|bioengineering|bsbe)\b", "Biosciences and Bioengineering"),
        (r"\b(?:chip\s+design|vlsi|semiconductors?)\b", "Chip Design & Semiconductors"),
        (r"\b(?:pmkvy|skill\s+hub)\b", "PMKVY Training"),
        (r"\b(?:gem|gfr|procurement)\b", "GeM & GFR Procurement Training"),
        (r"\b(?:hostel|mess|dining|canteen)\b", "Hostel and Dining Facilities"),
    )
    for pattern, name in dept_map:
        if re.search(pattern, low):
            return name
    return None


def _infer_target_or_topic(text: str, profile: Any = None) -> str | None:
    tgt = _infer_target(text, profile)
    if tgt:
        return tgt
    return _infer_department_or_topic_target(text)


def _is_multi_intent_or_compound(text: str) -> bool:
    low = text.casefold()
    # Multiple explicit questions
    if len(re.findall(r"\?", text)) >= 2:
        return True
    # Coordinated questions e.g. "..., and (what|which|how|where|when)...?"
    if re.search(r",?\s+\b(?:and|aur)\s+(?:what|which|how|where|when|who|is|are|can|kya|kaise|kitna|kitne|kahan)\b", text, re.I):
        return True
    # Coordinated 'both ... and ...' questions
    if re.search(r"\bboth\b.+?\band\b", low):
        return True
    # Distinct comparison markers
    if re.search(r"\b(?:compare|comparison|difference|different|versus|\bvs\b|same\s+thing|ya\s+alag|ya\s+woh\s+alag)\b", low):
        return True
    # Disjunctive question ("... or is ...", "... ya woh ...")
    if re.search(r",?\s+\b(?:or|ya)\b\s+(?:is|are|can|does|did|will|would|shuru|alag)\b", text, re.I):
        return True
    # Multiple distinct entities (2 or more topics/areas mentioned together)
    entities_count = sum(
        1 for k in (
            "ai hardware", "quantum electronics", "photonics", "robotics",
            "quantum computing", "control systems", "semiconductors",
            "vlsi", "battery materials", "energy storage", "gem", "gfr",
            "materials engineering", "electronics engineering", "aerospace",
        ) if k in low
    )
    if entities_count >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Fast Route Selector (Cheapest operation, < 1ms)
# ---------------------------------------------------------------------------

def select_query_route(
    *,
    question: str,
    resolved_question: str,
    chat_history: Sequence[Any] = (),
    conversation_result: Mapping[str, Any] | None = None,
    guard_result: GuardResult | None = None,
) -> HybridRouteDecision:
    """Evaluate cheap, deterministic signals to select Route A, Route B, or Route C.

    Never invokes an LLM.
    """
    conv = dict(conversation_result or {})
    q_orig = _clean(question)
    q_res = _clean(resolved_question or question)
    low_res = q_res.casefold()

    # 1. Scope / Guard stop signal
    if guard_result is not None:
        if guard_result.decision in (GuardDecision.OUT_OF_SCOPE, GuardDecision.SENSITIVE_REDIRECT):
            return HybridRouteDecision(
                route="A",
                reason=f"scope_guard_early_stop: {guard_result.reason}",
                confidence=1.0,
                is_multi_intent=False,
            )

    # 2. Unanchored reference / Clarification needed
    if conv.get("unanchored_reference"):
        target_topic = str(conv.get("active_topic") or "").casefold()
        if "fee" in low_res or "fee" in target_topic:
            target_phrase = "fee details"
        elif "eligib" in low_res or "eligib" in target_topic:
            target_phrase = "eligibility criteria"
        elif "curriculum" in low_res or "course" in low_res or "syllabus" in low_res:
            target_phrase = "course details"
        elif "admission" in low_res or "apply" in low_res or "process" in low_res:
            target_phrase = "admission details"
        else:
            target_phrase = "details"
        clarify_msg = (
            "Could you please specify which programme you are inquiring about "
            f"(for example, B.Tech., M.Tech., MBA, M.Sc., or Ph.D.) so I can provide the accurate {target_phrase}?"
        )
        return HybridRouteDecision(
            route="A",
            reason="unanchored_programme_reference_clarification",
            confidence=1.0,
            is_multi_intent=False,
            needs_clarification=True,
            clarification_message=clarify_msg,
        )

    # 3. Contextual and conversational reference signals
    conv_mode = conv.get("mode", "")
    res_source = conv.get("resolution_source", "")
    res_reason = conv.get("resolution_reason", "")

    # Unresolved ambiguous reference -> Route C
    if res_source in ("ambiguous_preserved", "ambiguous_pronoun", "ambiguous_anaphor") or res_reason in ("ambiguous_anaphor", "ambiguous_pronoun"):
        return HybridRouteDecision(
            route="C",
            reason=f"ambiguous_conversational_reference: {res_reason or 'multiple_plausible_antecedents'}",
            confidence=0.50,
            is_multi_intent=False,
        )

    is_contextual = (
        conv_mode in ("follow_up", "explicit_correction", "clarification_resolution")
        or (q_res.casefold() != q_orig.casefold())
    )

    if is_contextual:
        # Successfully resolved antecedent or correction -> Route B
        return HybridRouteDecision(
            route="B",
            reason="contextual_follow_up_resolved",
            confidence=0.95,
            is_multi_intent=False,
        )

    # 4. Multi-Intent / Separable Objectives / Compound Queries
    segments = _objective_segments(q_res)
    has_separable_intents = len(segments) > 1 and not (
        re.search(r"^\b(?:compare|difference between)\b", low_res)
    )
    if has_separable_intents:
        return HybridRouteDecision(
            route="B",
            reason=f"separable_multi_intent ({len(segments)} intents)",
            confidence=0.90,
            is_multi_intent=True,
        )

    if _is_multi_intent_or_compound(q_res):
        return HybridRouteDecision(
            route="B",
            reason="compound_structured_query",
            confidence=0.88,
            is_multi_intent=True,
        )

    # 5. Comparison between identifiable entities
    comp_targets = _comparison_targets(q_res)
    if comp_targets or any(w in low_res for w in ("compare", "difference", " vs ", "versus", "different")):
        return HybridRouteDecision(
            route="B",
            reason=f"comparison_between_entities: {', '.join(comp_targets) if comp_targets else 'lightweight_comparison'}",
            confidence=0.88,
            is_multi_intent=False,
        )

    # 6. High linguistic / narrative complexity
    # A long paragraph (> 45 words) with multiple conditional subordinations
    words = q_res.split()
    if len(words) >= 45 and any(w in low_res for w in ("provided that", "whereas", "on condition that", "supposing that", "in case of")):
        return HybridRouteDecision(
            route="C",
            reason="complex_interdependent_clauses",
            confidence=0.55,
            is_multi_intent=False,
        )

    # 7. Genuinely uncertain scope requiring semantic assessment
    guard_is_in = guard_result is not None and guard_result.decision == GuardDecision.IN_SCOPE
    if not guard_is_in:
        inst_keywords = (
            "iit", "iitj", "jodhpur", "aerospace", "materials", "electronics",
            "mathematics", "computing", "chip design", "vlsi", "semiconductor",
            "drone", "dadt", "pmkvy", "vtu", "gem", "gfr", "procurement",
            "course", "curriculum", "code", "fee", "hostel", "mess", "admission",
            "eligibility", "b.tech", "m.tech", "m.sc", "mba", "phd", "ph.d",
            "department", "faculty", "research", "credits", "credit", "elective",
        )
        relevance = _relevance(q_res)
        if not any(k in low_res for k in inst_keywords) and relevance < 0.05:
            return HybridRouteDecision(
                route="C",
                reason="low_confidence_scope_uncertainty",
                confidence=0.50,
                is_multi_intent=False,
            )

    # 8. Default: Clear Standalone Factual -> Route A
    return HybridRouteDecision(
        route="A",
        reason="clear_standalone_factual",
        confidence=0.98,
        is_multi_intent=False,
    )


# ---------------------------------------------------------------------------
# Route A: Deterministic Python Fast Path
# ---------------------------------------------------------------------------

def execute_route_a(
    question: str,
    *,
    guard_result: GuardResult | None = None,
    profile: Any = None,
) -> tuple[QueryPlan, bool]:
    """Execute Route A deterministic fast path.

    Returns:
        tuple[QueryPlan, bool]: (plan, should_escalate)
    """
    q_clean = _clean(question)

    # 1. Guard check
    if guard_result and guard_result.decision in (GuardDecision.OUT_OF_SCOPE, GuardDecision.SENSITIVE_REDIRECT):
        plan = QueryPlan(
            original_query=q_clean,
            resolved_query=q_clean,
            domain_decision="out_of_scope",
            domain_confidence=1.0,
            domain_reason=guard_result.reason,
            is_multi_intent=False,
            intents=[],
            preserved_terms=[],
            confidence=1.0,
            route="route_a_fast_path",
        )
        return plan, False

    # 2. Extract deterministic target, requested attributes, and constraints
    target = _infer_target_or_topic(q_clean, profile)
    attrs = _requested_attributes(q_clean)
    constraints = _constraints(q_clean)
    preserved = _dedupe([*_preserved(q_clean), *constraints], 28)

    # Check if multiple separable intents were discovered during extraction
    segments = _objective_segments(q_clean)
    if len(segments) > 1:
        # Multiple separable objectives found: escalate to Route B
        return None, True

    # 3. Build safe retrieval query
    retrieval_query = _safe_retrieval_query(q_clean, target, attrs, constraints)
    req_type = attrs[0] if attrs else "general"

    intent = QueryIntentPlan(
        question=q_clean,
        request_type=req_type,
        target=target,
        requested_attributes=list(attrs),
        constraints=list(constraints),
        retrieval_query=retrieval_query,
        confidence=0.95,
    )

    # 4. Scope repair
    guard_is_in = guard_result is not None and guard_result.decision == GuardDecision.IN_SCOPE
    init_scope = "in_scope" if (guard_is_in or _relevance(q_clean) >= 0.10) else "uncertain"
    scope, scope_conf, scope_reason = _repair_scope(q_clean, init_scope, 0.90)

    plan = QueryPlan(
        original_query=q_clean,
        resolved_query=q_clean,
        domain_decision=scope,
        domain_confidence=scope_conf,
        domain_reason="deterministic_fast_path_verified",
        is_multi_intent=False,
        intents=[intent],
        preserved_terms=preserved,
        confidence=0.95,
        route="route_a_fast_path",
    )
    return plan, False


# ---------------------------------------------------------------------------
# Route B: Lightweight Hybrid Processing Without Generative LLM
# ---------------------------------------------------------------------------

def execute_route_b(
    original_question: str,
    resolved_question: str,
    *,
    conversation_result: Mapping[str, Any] | None = None,
    guard_result: GuardResult | None = None,
    profile: Any = None,
) -> tuple[QueryPlan, bool]:
    """Execute Route B lightweight non-generative hybrid query processing.

    Returns:
        tuple[QueryPlan, bool]: (plan, should_escalate)
    """
    q_orig = _clean(original_question)
    q_res = _clean(resolved_question or original_question)
    conv = dict(conversation_result or {})

    # 1. Guard check
    if guard_result and guard_result.decision in (GuardDecision.OUT_OF_SCOPE, GuardDecision.SENSITIVE_REDIRECT):
        plan = QueryPlan(
            original_query=q_orig,
            resolved_query=q_res,
            domain_decision="out_of_scope",
            domain_confidence=1.0,
            domain_reason=guard_result.reason,
            is_multi_intent=False,
            intents=[],
            preserved_terms=[],
            confidence=1.0,
            route="route_b_lightweight",
        )
        return plan, False

    # 2. Decompose intents deterministically
    intents = _deterministic_intents(q_res)
    if not intents:
        # Fallback to single intent
        target = _infer_target_or_topic(q_res, profile)
        attrs = _requested_attributes(q_res)
        constraints = _constraints(q_res)
        intents = [
            QueryIntentPlan(
                question=q_res,
                request_type=attrs[0] if attrs else "general",
                target=target,
                requested_attributes=list(attrs),
                constraints=list(constraints),
                retrieval_query=_safe_retrieval_query(q_res, target, attrs, constraints),
                confidence=0.85,
            )
        ]
    else:
        for it in intents:
            if not it.target:
                fallback_tgt = _infer_department_or_topic_target(it.question or q_res)
                if fallback_tgt:
                    it.target = fallback_tgt

    # Check if confidence is extremely low (< 0.55), indicating need for Route C
    min_intent_conf = min((i.confidence for i in intents), default=0.0)
    if min_intent_conf < 0.55:
        return None, True

    is_multi = len(intents) > 1
    preserved = _dedupe([*_preserved(q_res), *_constraints(q_res)], 28)

    # 3. Scope repair
    guard_is_in = guard_result is not None and guard_result.decision == GuardDecision.IN_SCOPE
    init_scope = "in_scope" if (guard_is_in or _relevance(q_res) >= 0.10) else "uncertain"
    scope, scope_conf, scope_reason = _repair_scope(q_res, init_scope, 0.85)

    raw_mode = conv.get("mode")
    mode = str(raw_mode) if raw_mode else ("follow_up" if q_res.casefold() != q_orig.casefold() else "standalone")

    plan = QueryPlan(
        original_query=q_orig,
        resolved_query=q_res,
        conversation_mode=mode,
        domain_decision=scope,
        domain_confidence=scope_conf,
        domain_reason="lightweight_hybrid_resolution",
        is_multi_intent=is_multi,
        intents=intents,
        preserved_terms=preserved,
        confidence=min(0.92, max((i.confidence for i in intents), default=0.85)),
        route="route_b_lightweight",
    )
    return plan, False


# ---------------------------------------------------------------------------
# Route C: Existing LLM-Based Query Planner
# ---------------------------------------------------------------------------

def execute_route_c(
    original_question: str,
    resolved_question: str,
    *,
    planner_fn: Callable[..., QueryPlan],
    conversation_result: Mapping[str, Any] | None = None,
) -> QueryPlan:
    """Execute Route C using the existing local LLM query planner.

    Guarantees exactly ONE LLM call.
    """
    q_orig = _clean(original_question)
    q_res = _clean(resolved_question or original_question)
    conv = dict(conversation_result or {})

    # Invoke existing planner with the resolved/original question
    plan = planner_fn(question=q_res)

    # Enforce contract fields
    mode = conv.get("mode", plan.conversation_mode or "standalone")
    updated_dict = plan.model_dump()
    updated_dict["original_query"] = q_orig
    updated_dict["resolved_query"] = q_res
    updated_dict["conversation_mode"] = mode
    updated_dict["route"] = "route_c_llm_planner"

    return QueryPlan.model_validate(updated_dict)


# ---------------------------------------------------------------------------
# Canonical Orchestrator: Hybrid Route Pipeline
# ---------------------------------------------------------------------------

def run_hybrid_query_understanding(
    *,
    question: str,
    resolved_question: str,
    chat_history: Sequence[Any] = (),
    conversation_result: Mapping[str, Any] | None = None,
    guard_result: GuardResult | None = None,
    planner_fn: Callable[..., QueryPlan] | None = None,
    profile: Any = None,
) -> HybridRouteExecution:
    """Run the complete Three-Route Hybrid Query Understanding stage.

    Flow:
    1. Pre-Route Selection (< 1ms).
    2. Route Execution (A, B, or C).
    3. Safe Escalation if Route A or B returns should_escalate.
    4. Sub-stage timing and diagnostics collection.
    """
    t0_all = time.perf_counter()
    conv = dict(conversation_result or {})
    q_orig = _clean(question)
    q_res = _clean(resolved_question or question)

    op_timings: dict[str, float] = {}

    # Fast route selection
    t0_sel = time.perf_counter()
    decision = select_query_route(
        question=q_orig,
        resolved_question=q_res,
        chat_history=chat_history,
        conversation_result=conv,
        guard_result=guard_result,
    )
    t_sel = time.perf_counter() - t0_sel
    op_timings["route_selection"] = t_sel

    # Early clarification stop
    if decision.needs_clarification:
        plan = QueryPlan(
            original_query=q_orig,
            resolved_query=q_res,
            domain_decision="in_scope",
            domain_confidence=1.0,
            domain_reason="unanchored_reference_clarification",
            is_multi_intent=False,
            intents=[],
            preserved_terms=[],
            ambiguous=True,
            needs_clarification=True,
            clarification_reason=decision.clarification_message or "Programme clarification required",
            confidence=1.0,
            route="route_a_fast_path",
        )
        t_total = time.perf_counter() - t0_all
        return HybridRouteExecution(
            plan=plan,
            selected_route="A",
            initial_route="A",
            route_reason=decision.reason,
            escalation_happened=False,
            selector_time_seconds=t_sel,
            route_time_seconds=0.0001,
            total_qu_time_seconds=t_total,
            qu_llm_calls=0,
            operation_timings=op_timings,
            pipeline_stop=True,
            safe_response=decision.clarification_message,
            stop_reason="unanchored_programme_clarification",
        )

    selected_route = decision.route
    initial_route = decision.route
    route_reason = decision.reason
    escalation_happened = False
    llm_calls_count = 0

    t0_exec = time.perf_counter()
    plan: QueryPlan | None = None

    # Route A execution with safe escalation
    if selected_route == "A":
        t0_a = time.perf_counter()
        plan, should_escalate = execute_route_a(
            q_res,
            guard_result=guard_result,
            profile=profile,
        )
        op_timings["route_a_fast_path"] = time.perf_counter() - t0_a
        if should_escalate:
            escalation_happened = True
            selected_route = "B"
            route_reason = f"escalated_from_A_to_B: separable_intents_discovered"

    # Route B execution with safe escalation
    if selected_route == "B":
        t0_b = time.perf_counter()
        plan, should_escalate = execute_route_b(
            q_orig,
            q_res,
            conversation_result=conv,
            guard_result=guard_result,
            profile=profile,
        )
        op_timings["route_b_lightweight"] = time.perf_counter() - t0_b
        if should_escalate:
            escalation_happened = True
            selected_route = "C"
            route_reason = f"escalated_from_B_to_C: low_confidence_or_complex_structure"

    # Route C execution (if selected or escalated)
    if selected_route == "C":
        t0_c = time.perf_counter()
        if planner_fn is None:
            # Fallback to deterministic build if planner_fn not provided
            plan = _deterministic_build_query_plan(q_res)
            op_timings["route_c_fallback"] = time.perf_counter() - t0_c
        else:
            plan = execute_route_c(
                q_orig,
                q_res,
                planner_fn=planner_fn,
                conversation_result=conv,
            )
            op_timings["query_planning_llm"] = time.perf_counter() - t0_c
            llm_calls_count = 1

    t_exec = time.perf_counter() - t0_exec
    t_total = time.perf_counter() - t0_all

    # Build safe stop response if applicable
    guard_stop = (
        guard_result is not None
        and guard_result.decision in (GuardDecision.OUT_OF_SCOPE, GuardDecision.SENSITIVE_REDIRECT)
    )
    pipeline_stop = guard_stop or (plan is not None and should_stop_pipeline(plan))
    safe_response = None
    stop_reason = ""
    if guard_stop and guard_result:
        safe_response = guard_result.safe_response
        stop_reason = guard_result.reason
    elif pipeline_stop and plan:
        stop_reason = plan.domain_reason or "high_confidence_out_of_scope"
        safe_response = "This is not an institutional request. I can only assist with academic, administrative, and campus information regarding IIT Jodhpur."

    return HybridRouteExecution(
        plan=plan,
        selected_route=selected_route,
        initial_route=initial_route,
        route_reason=route_reason,
        escalation_happened=escalation_happened,
        selector_time_seconds=t_sel,
        route_time_seconds=t_exec,
        total_qu_time_seconds=t_total,
        qu_llm_calls=llm_calls_count,
        operation_timings=op_timings,
        pipeline_stop=pipeline_stop,
        safe_response=safe_response,
        stop_reason=stop_reason,
    )
