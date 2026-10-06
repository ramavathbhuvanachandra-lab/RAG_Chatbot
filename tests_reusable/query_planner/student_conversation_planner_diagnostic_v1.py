#!/usr/bin/env python3
"""
Student Conversation -> Query Planner Production Diagnostic V3

Purpose
-------
A standalone adversarial diagnostic for the current ai_platform conversation
resolver + canonical query planner.

This file intentionally does NOT import any previous diagnostic runner.
It is safe to install at:
    tests_reusable/query_planner/student_conversation_planner_diagnostic_v1.py

Production code is not modified.

Scope
-----
- Conversation resolution
- Reference/entity resolution
- Ambiguity protection
- Language handling via planner output
- Scope
- Intent count / multi-intent
- Target
- Request type
- Requested attributes
- Constraints / qualifiers
- Negation
- Comparison targets
- Retrieval-query formulation
- Per-stage timing
- Root-cause classification

Not tested
----------
- Retrieval database
- Ranking
- Evidence verifier
- Context builder
- Answer LLM
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.planner import build_query_plan, retrieval_queries_from_plan


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    question: str
    history: tuple[tuple[str, str], ...] = ()
    expect: dict[str, Any] | None = None


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def contains_ci(value: Any, wanted: Any) -> bool:
    return norm(wanted) in norm(value)


def list_contains_ci(values: Sequence[Any], wanted: Any) -> bool:
    nw = norm(wanted)
    return any(nw == norm(v) or nw in norm(v) for v in values)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(v) for v in value if str(v).strip()]
    except TypeError:
        return []


def actual_attrs(intent: Any) -> list[str]:
    for name in ("requested_attributes", "attributes", "qualifiers"):
        value = getattr(intent, name, None)
        items = _string_list(value)
        if items:
            return items
    return []


def actual_constraints(intent: Any) -> list[str]:
    out: list[str] = []
    for name in ("constraints", "qualifiers", "temporal_constraints", "relations"):
        out.extend(_string_list(getattr(intent, name, None)))
    return out


def actual_comparisons(intent: Any) -> list[str]:
    return _string_list(getattr(intent, "comparison_targets", None))


def _safe_history(history: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for item in history:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
        elif hasattr(item, "role") and hasattr(item, "content"):
            out.append((str(item.role), str(item.content)))
        elif isinstance(item, dict):
            out.append((
                str(item.get("role") or item.get("type") or "user"),
                str(item.get("content") or item.get("text") or ""),
            ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Adversarial suite
# ---------------------------------------------------------------------------

CASES: tuple[Case, ...] = (
    # 1-10: difficult standalone English
    Case("Q01", "standalone_high_info", "I have a B.Tech degree with 7.2 CGPA and a valid GATE score. Does that make me eligible for the M.Tech admission route, and what additional academic condition could still disqualify me?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility"}),
    Case("Q02", "standalone_high_info", "I am in my final semester and my degree will be completed only after the admission process. Can I still be considered for M.Tech, or is completed graduation mandatory?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "constraint_contains": ["final semester"]}),
    Case("Q03", "semantic_trap", "I am not asking whether I am eligible for M.Tech. I only need the application fee and whether there is any separate registration charge.",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "fee", "forbidden_attrs": ["eligibility"]}),
    Case("Q04", "semantic_trap", "When you say 'apply', I mean the actual admission process steps. What are the steps rather than the eligibility rules?",
         expect={"scope": "in_scope", "type": "process"}),
    Case("Q05", "multi_intent", "For M.Tech, I need three things together: eligibility, the list of documents, and the application deadline.",
         expect={"scope": "in_scope", "target": "M.Tech", "multi": True, "intent_count_exact": 3}),
    Case("Q06", "comparison", "Compare M.Tech and MBA admissions specifically on entrance-exam requirements and selection process. Do not collapse them into one generic admission answer.",
         expect={"scope": "in_scope", "multi": True, "comparisons_all": ["M.Tech", "MBA"]}),
    Case("Q07", "negation", "I do not want hostel information. I only want to know whether M.Tech students need work experience.",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "negated": False}),
    Case("Q08", "temporal", "What is the deadline for M.Tech applications for the current admission cycle? I am asking about the application deadline, not the semester start date.",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "deadline"}),
    Case("Q09", "out_of_scope", "Can you tell me which private bank currently gives the highest interest rate on an education loan?",
         expect={"scope": "out_of_scope", "stop": True}),
    Case("Q10", "attribute_preservation", "For M.Sc admission, tell me the exact eligibility threshold, qualifying examination, and whether a minimum percentage is mandatory.",
         expect={"scope": "in_scope", "target": "M.Sc", "type": "eligibility", "attrs": ["percentage"]}),

    # 11-20: short / compressed student language
    Case("Q11", "short", "M.Tech eligibility?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility"}),
    Case("Q12", "short", "M.Tech fee?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "fee"}),
    Case("Q13", "short", "Documents for MBA?",
         expect={"scope": "in_scope", "target": "MBA", "type": "documents"}),
    Case("Q14", "short", "M.Sc deadline?",
         expect={"scope": "in_scope", "target": "M.Sc", "type": "deadline"}),
    Case("Q15", "short", "Work experience compulsory for M.Tech?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["experience"]}),
    Case("Q16", "short_hinglish", "M.Tech ka form kab bharna hai?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "deadline"}),
    Case("Q17", "short_hinglish", "M.Tech mein GATE zaroori hai kya?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["GATE"]}),
    Case("Q18", "short_hinglish", "MBA ke liye experience chahiye?",
         expect={"scope": "in_scope", "target": "MBA", "type": "eligibility", "attrs": ["experience"]}),
    Case("Q19", "short_hinglish", "hostel fees kitni hain B.Tech ke liye?",
         expect={"scope": "in_scope", "target": "B.Tech", "type": "fee"}),
    Case("Q20", "short_hinglish", "admission ka process batao, documents nahi.",
         expect={"scope": "in_scope", "type": "process", "forbidden_attrs": ["documents"]}),

    # 21-30: Hindi / Devanagari / mixed
    Case("Q21", "hindi", "M.Tech में दाखिले की पात्रता क्या है?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility"}),
    Case("Q22", "hindi", "M.Sc के लिए आवेदन शुल्क कितना है?",
         expect={"scope": "in_scope", "target": "M.Sc", "type": "fee"}),
    Case("Q23", "hindi", "MBA में प्रवेश के लिए कौन-कौन से दस्तावेज़ चाहिए?",
         expect={"scope": "in_scope", "target": "MBA", "type": "documents"}),
    Case("Q24", "hindi", "M.Tech के आवेदन की अंतिम तिथि क्या है?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "deadline"}),
    Case("Q25", "hindi", "क्या M.Tech के लिए कार्य अनुभव अनिवार्य है?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["experience"]}),
    Case("Q26", "hinglish", "M.Tech admission ke liye minimum CGPA kitna chahiye?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["CGPA"]}),
    Case("Q27", "hinglish", "Agar GATE score nahi hai to M.Tech ke liye apply kar sakte hain?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["GATE"]}),
    Case("Q28", "hinglish", "M.Sc ka form bharne ke baad documents kab submit karne hote hain?",
         expect={"scope": "in_scope", "target": "M.Sc", "type": "process", "attrs": ["documents"]}),
    Case("Q29", "hinglish", "MBA aur M.Tech dono ka fee structure compare karo, admission fee nahi to application fee specifically.",
         expect={"scope": "in_scope", "multi": True, "comparisons_all": ["MBA", "M.Tech"]}),
    Case("Q30", "hindi_complex", "मैंने B.Tech पूरा कर लिया है और GATE में अच्छा स्कोर है, लेकिन मेरे पास work experience नहीं है। क्या M.Tech के लिए यह समस्या होगी?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["experience", "GATE"]}),

    # 31-40: conversational follow-ups designed to break reference resolution
    Case("Q31", "follow_up", "What is the M.Tech eligibility?",
         history=(("user", "I am considering M.Tech admission."),),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "conv_mode": "follow_up"}),
    Case("Q32", "follow_up_short", "Fee?",
         history=(("user", "I want to know about M.Tech admission."),),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "fee", "conv_mode": "follow_up", "entity": "M.Tech"}),
    Case("Q33", "follow_up_short", "Documents?",
         history=(("user", "Tell me about MBA admission."),),
         expect={"scope": "in_scope", "target": "MBA", "type": "documents", "conv_mode": "follow_up", "entity": "MBA"}),
    Case("Q34", "follow_up_short", "Deadline?",
         history=(("user", "What are the M.Sc admission requirements?"),),
         expect={"scope": "in_scope", "target": "M.Sc", "type": "deadline", "conv_mode": "follow_up"}),
    Case("Q35", "follow_up_anaphor", "What about the fee for that one?",
         history=(("user", "I need the M.Tech admission details."), ("assistant", "Sure, I can help with M.Tech.")),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "fee", "conv_mode": "follow_up"}),
    Case("Q36", "follow_up_anaphor", "And the deadline for the second one?",
         history=(("user", "Compare M.Tech and MBA admissions."),),
         expect={"scope": "in_scope", "multi": True, "type": "deadline"}),
    Case("Q37", "follow_up_anaphor", "Can I apply for it this year?",
         history=(("user", "I am asking about M.Tech."),),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "general", "conv_mode": "follow_up"}),
    Case("Q38", "follow_up_anaphor", "What if I don't have it?",
         history=(("user", "Is work experience compulsory for M.Tech?"),),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "conv_mode": "follow_up"}),
    Case("Q39", "follow_up_anaphor", "How much is it?",
         history=(("user", "What is the M.Tech application fee?"),),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "fee", "conv_mode": "follow_up"}),
    Case("Q40", "follow_up_shift", "No, forget the eligibility. Just tell me the deadline.",
         history=(("user", "What is the M.Tech eligibility?"),),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "deadline"}),

    # 41-50: ambiguity / multi-intent / comparison / contradiction
    Case("Q41", "ambiguity", "What about the fee for the first one?",
         history=(("user", "Tell me about M.Tech and MBA."),),
         expect={"scope": "in_scope", "ambiguous": True}),
    Case("Q42", "ambiguity", "And the eligibility for that program?",
         history=(("user", "Compare M.Tech, MBA, and M.Sc."),),
         expect={"scope": "in_scope", "ambiguous": True}),
    Case("Q43", "ambiguity", "What documents do I need for it?",
         history=(("user", "I am considering M.Tech and PhD."),),
         expect={"scope": "in_scope", "ambiguous": True}),
    Case("Q44", "multi_intent", "I need to know whether M.Tech requires GATE, whether work experience is compulsory, and what documents prove eligibility.",
         expect={"scope": "in_scope", "target": "M.Tech", "multi": True, "intent_count_exact": 3}),
    Case("Q45", "comparison", "Which is more demanding for admission, M.Tech or MBA, in terms of qualifying exam and minimum academic requirements? Keep the two targets separate.",
         expect={"scope": "in_scope", "multi": True, "comparisons_all": ["M.Tech", "MBA"]}),
    Case("Q46", "negation", "Don't tell me the application deadline. I want to know the admission fee, and specifically not the hostel fee.",
         expect={"scope": "in_scope", "type": "fee"}),
    Case("Q47", "contradictory", "I want the M.Tech deadline, but actually I mean the deadline for MBA, not M.Tech.",
         expect={"scope": "in_scope", "target": "MBA", "type": "deadline"}),
    Case("Q48", "dense_constraints", "I am a final-year B.Tech student from another university, have 7.0 CGPA, qualified GATE, and have zero work experience. For M.Tech, which of these facts matter to eligibility?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["CGPA", "GATE", "experience"]}),
    Case("Q49", "multi_intent", "For M.Sc, tell me the application fee and deadline, and also whether the required qualification can be in a related discipline.",
         expect={"scope": "in_scope", "target": "M.Sc", "multi": True}),
    Case("Q50", "oos_trap", "Can you recommend the best laptop under INR 80,000 for running local LLMs as an IIT student?",
         expect={"scope": "out_of_scope", "stop": True}),

    # 51-60: high-breakage cases / real student phrasing
    Case("Q51", "high_breakage", "I read that M.Tech eligibility and GATE qualification are separate things. I have the required degree but my GATE score is below the stated cut-off. Am I eligible or not?",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["GATE"]}),
    Case("Q52", "high_breakage", "Suppose the notification says one thing but I am asking about the programme's general eligibility. Which rule should I use?",
         expect={"scope": "in_scope", "type": "eligibility"}),
    Case("Q53", "high_breakage", "I already know the documents. I am asking whether any of them have to be uploaded at the application stage versus produced later during admission.",
         expect={"scope": "in_scope", "type": "process", "attrs": ["documents"]}),
    Case("Q54", "high_breakage", "M.Tech ka eligibility batao but please don't mix it with PhD eligibility. Main sirf M.Tech ki baat kar raha hoon.",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "forbidden_targets": ["PhD"]}),
    Case("Q55", "high_breakage", "M.Tech ke liye form fill karna hai. Application fee, last date aur required documents ek saath bata sakte ho?",
         expect={"scope": "in_scope", "target": "M.Tech", "multi": True}),
    Case("Q56", "high_breakage_followup", "What about the second requirement?",
         history=(("user", "For M.Tech, what are the eligibility requirements?"), ("assistant", "The requirements include academic qualifications and GATE-related conditions.")),
         expect={"scope": "in_scope", "target": "M.Tech", "conv_mode": "follow_up"}),
    Case("Q57", "high_breakage_followup", "Actually, compare that with MBA.",
         history=(("user", "What are the M.Tech admission requirements?"),),
         expect={"scope": "in_scope", "target": "M.Tech", "multi": True, "comparisons_all": ["MBA"]}),
    Case("Q58", "high_breakage_followup", "Is experience compulsory if I am applying directly after graduation?",
         history=(("user", "We were discussing M.Tech admissions."),),
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["experience"]}),
    Case("Q59", "language_switch", "M.Tech ki eligibility kya hai? Also tell me whether there is any relaxation in CGPA for reserved categories.",
         expect={"scope": "in_scope", "target": "M.Tech", "type": "eligibility", "attrs": ["CGPA"]}),
    Case("Q60", "language_switch", "I know the deadline already. मुझे सिर्फ यह बताइए कि application submit करने के बाद documents कैसे verify होते हैं?",
         expect={"scope": "in_scope", "type": "process", "attrs": ["documents"]}),
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

STAGES = (
    "conversation",
    "scope",
    "intent_count",
    "target",
    "type",
    "attributes",
    "constraints",
    "negation",
    "comparison",
    "retrieval_query",
)


def _primary_failure(stages: dict[str, bool]) -> str:
    for stage in STAGES:
        if not stages.get(stage, False):
            return stage
    return "PASS"


def evaluate_case(case: Case) -> dict[str, Any]:
    expect = case.expect or {}
    started = time.perf_counter()

    # ---------------- Conversation ----------------
    conv_started = time.perf_counter()
    try:
        conv = resolve_conversation(
            question=case.question,
            chat_history=_safe_history(case.history),
        )
        conv_elapsed = time.perf_counter() - conv_started
    except Exception as exc:
        conv_elapsed = time.perf_counter() - conv_started
        return {
            "case": case,
            "elapsed": time.perf_counter() - started,
            "conversation_elapsed": conv_elapsed,
            "planner_elapsed": 0.0,
            "conversation_source": "",
            "conversation_reason": "",
            "stages": {s: False for s in STAGES},
            "primary_failure": "conversation",
            "conversation_error": f"{type(exc).__name__}: {exc}",
            "plan_error": "",
            "conv": None,
            "plan": None,
            "details": ["conversation resolver crashed"],
        }

    resolved = str(conv.get("resolved_question") or case.question).strip()
    mode = str(conv.get("mode") or "").strip()
    entity = str(conv.get("active_entity") or "").strip()
    source = str(conv.get("resolution_source") or "").strip()
    reason = str(conv.get("resolution_reason") or "").strip()

    conversation_ok = True
    details: list[str] = []

    expected_mode = expect.get("conv_mode")
    if expected_mode is not None and mode != expected_mode:
        conversation_ok = False
        details.append(f"conversation mode expected={expected_mode!r} observed={mode!r}")

    expected_entity = expect.get("entity")
    if expected_entity is not None and norm(entity) != norm(expected_entity):
        conversation_ok = False
        details.append(f"conversation entity expected={expected_entity!r} observed={entity!r}")

    expected_contains = expect.get("resolved_contains")
    if expected_contains and not contains_ci(resolved, expected_contains):
        conversation_ok = False
        details.append(f"resolved question missing expected text={expected_contains!r}")

    if expect.get("ambiguous"):
        ambiguous_safe = (
            source in {"ambiguous_preserved", "fallback"}
            or not entity
        )
        if not ambiguous_safe:
            conversation_ok = False
            details.append("ambiguous case was not conservatively preserved")

    if expect.get("preserve_original") and norm(resolved) != norm(case.question):
        conversation_ok = False
        details.append("original wording was not preserved where required")

    # ---------------- Planner ----------------
    plan_started = time.perf_counter()
    try:
        plan = build_query_plan(question=resolved)
        plan_elapsed = time.perf_counter() - plan_started
    except Exception as exc:
        plan_elapsed = time.perf_counter() - plan_started
        stages = {s: False for s in STAGES}
        stages["conversation"] = conversation_ok
        return {
            "case": case,
            "elapsed": time.perf_counter() - started,
            "conversation_elapsed": conv_elapsed,
            "planner_elapsed": plan_elapsed,
            "conversation_source": source,
            "conversation_reason": reason,
            "stages": stages,
            "primary_failure": "planner",
            "conversation_error": "",
            "plan_error": f"{type(exc).__name__}: {exc}",
            "conv": conv,
            "plan": None,
            "details": details + ["query planner crashed"],
        }

    intents = list(getattr(plan, "intents", None) or [])
    primary = intents[0] if intents else None

    stages: dict[str, bool] = {"conversation": conversation_ok}

    # Scope
    expected_scope = expect.get("scope")
    stages["scope"] = True if expected_scope is None else (
        str(getattr(plan, "domain_decision", "")) == str(expected_scope)
    )
    if "stop" in expect:
        observed_stop = (
            getattr(plan, "domain_decision", None) == "out_of_scope"
            and float(getattr(plan, "domain_confidence", 0.0) or 0.0) >= 0.90
        )
        stages["scope"] &= observed_stop == bool(expect["stop"])

    # Intent count
    if "multi" in expect:
        stages["intent_count"] = bool(getattr(plan, "is_multi_intent", False)) == bool(expect["multi"])
    else:
        stages["intent_count"] = True
    if "intent_count_exact" in expect:
        stages["intent_count"] &= len(intents) == int(expect["intent_count_exact"])

    # Target
    if expect.get("ambiguous"):
        stages["target"] = primary is None or not getattr(primary, "target", None)
    elif expect.get("target") is not None:
        stages["target"] = (
            primary is not None
            and norm(getattr(primary, "target", "")) == norm(expect["target"])
        )
    else:
        stages["target"] = True

    forbidden_targets = expect.get("forbidden_targets", [])
    if forbidden_targets:
        observed_targets = [
            str(getattr(i, "target", "") or "") for i in intents
        ]
        stages["target"] &= all(
            not list_contains_ci(observed_targets, t) for t in forbidden_targets
        )

    # Request type
    expected_type = expect.get("type")
    if expected_type is not None:
        stages["type"] = (
            primary is not None
            and str(getattr(primary, "request_type", "") or "") == str(expected_type)
        )
    else:
        stages["type"] = True

    # Attributes
    expected_attrs = list(expect.get("attrs", ()))
    forbidden_attrs = list(expect.get("forbidden_attrs", ()))
    all_attrs: list[str] = []
    for intent in intents:
        all_attrs.extend(actual_attrs(intent))

    if expected_attrs:
        observed_attrs = actual_attrs(primary) if primary is not None else []
        stages["attributes"] = all(
            list_contains_ci(observed_attrs, wanted)
            or any(contains_ci(x, wanted) for x in observed_attrs)
            for wanted in expected_attrs
        )
    else:
        stages["attributes"] = True

    if forbidden_attrs:
        stages["attributes"] &= all(
            not any(contains_ci(x, forbidden) for x in all_attrs)
            for forbidden in forbidden_attrs
        )

    # Constraints
    expected_constraints = list(expect.get("constraint_contains", ()))
    all_constraints: list[str] = []
    for intent in intents:
        all_constraints.extend(actual_constraints(intent))
    stages["constraints"] = all(
        list_contains_ci(all_constraints, wanted)
        or any(contains_ci(c, wanted) for c in all_constraints)
        for wanted in expected_constraints
    )

    # Negation
    if "negated" in expect:
        stages["negation"] = (
            any(bool(getattr(intent, "negated", False)) for intent in intents)
            == bool(expect["negated"])
        )
    else:
        stages["negation"] = True

    # Comparison
    expected_comparisons = list(expect.get("comparisons_all", ()))
    observed_comparisons: list[str] = []
    for intent in intents:
        observed_comparisons.extend(actual_comparisons(intent))

    stages["comparison"] = all(
        list_contains_ci(observed_comparisons, wanted)
        or any(contains_ci(x, wanted) for x in observed_comparisons)
        for wanted in expected_comparisons
    )

    # Retrieval formulation
    try:
        retrievals = tuple(retrieval_queries_from_plan(plan, limit=3))
    except Exception as exc:
        retrievals = ()
        details.append(f"retrieval formulation crashed: {type(exc).__name__}: {exc}")

    retrieval_blob = " | ".join(retrievals)
    retrieval_ok = bool(retrievals) and bool(retrieval_blob.strip())

    if primary is not None:
        target = str(getattr(primary, "target", "") or "")
        attrs = actual_attrs(primary)

        if target:
            retrieval_ok &= norm(target) in norm(retrieval_blob)

        for attr in attrs[:4]:
            retrieval_ok &= norm(attr) in norm(retrieval_blob)

    for comparison_target in expected_comparisons:
        retrieval_ok &= norm(comparison_target) in norm(retrieval_blob)

    stages["retrieval_query"] = retrieval_ok

    elapsed = time.perf_counter() - started
    failure = _primary_failure(stages)

    if failure != "PASS":
        details.append(f"earliest contract break={failure}")

    return {
        "case": case,
        "elapsed": elapsed,
        "conversation_elapsed": conv_elapsed,
        "planner_elapsed": plan_elapsed,
        "conversation_source": source,
        "conversation_reason": reason,
        "stages": stages,
        "primary_failure": failure,
        "conversation_error": "",
        "plan_error": "",
        "conv": conv,
        "plan": plan,
        "retrievals": retrievals,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rate(ok: int, total: int) -> float:
    return 100.0 * ok / total if total else 0.0


def print_case_result(index: int, total: int, result: dict[str, Any]) -> None:
    case = result["case"]
    elapsed = result["elapsed"]
    conv_t = result.get("conversation_elapsed", 0.0)
    plan_t = result.get("planner_elapsed", 0.0)
    status = result["primary_failure"]

    speed = ""
    if max(conv_t, plan_t) >= 15:
        speed = " | VERY SLOW"
    elif max(conv_t, plan_t) >= 5:
        speed = " | SLOW"

    print(
        f"[{index:02d}/{total:02d}] {case.case_id} "
        f"{status:<18} total={elapsed:7.2f}s "
        f"conv={conv_t:7.2f}s planner={plan_t:7.2f}s{speed}",
        flush=True,
    )

    if status != "PASS":
        print(f"       CATEGORY: {case.category}", flush=True)
        print(f"       Q: {case.question}", flush=True)
        if result.get("details"):
            for d in result["details"][:4]:
                print(f"       WHY: {d}", flush=True)

        conv = result.get("conv")
        plan = result.get("plan")
        if conv:
            print(
                f"       CONV: mode={conv.get('mode')!r} "
                f"entity={conv.get('active_entity')!r} "
                f"source={conv.get('resolution_source')!r} "
                f"reason={conv.get('resolution_reason')!r}",
                flush=True,
            )
            print(
                f"       RESOLVED: {conv.get('resolved_question')!r}",
                flush=True,
            )
        if plan:
            intents = list(getattr(plan, "intents", None) or [])
            print(
                f"       PLAN: scope={getattr(plan, 'domain_decision', None)!r} "
                f"multi={getattr(plan, 'is_multi_intent', None)!r} "
                f"intents={len(intents)}",
                flush=True,
            )
            for i, intent in enumerate(intents[:4], start=1):
                print(
                    f"       INTENT{i}: "
                    f"target={getattr(intent, 'target', None)!r} "
                    f"type={getattr(intent, 'request_type', None)!r} "
                    f"attrs={actual_attrs(intent)!r} "
                    f"constraints={actual_constraints(intent)!r}",
                    flush=True,
                )


def print_report(results: list[dict[str, Any]]) -> None:
    total = len(results)
    passed = sum(r["primary_failure"] == "PASS" for r in results)
    failed = total - passed

    stage_stats: dict[str, tuple[int, int]] = {}
    for stage in STAGES:
        ok = sum(bool(r["stages"].get(stage)) for r in results)
        stage_stats[stage] = (ok, total - ok)

    roots = Counter(r["primary_failure"] for r in results if r["primary_failure"] != "PASS")

    deterministic = sum(
        r.get("conversation_source") in {"deterministic", "explicit_subject"}
        for r in results
    )
    conv_llm = sum(
        r.get("conversation_source") in {"llm", "llm_rejected"}
        for r in results
    )
    conv_ambiguous = sum(
        r.get("conversation_source") == "ambiguous_preserved"
        for r in results
    )

    times = sorted(float(r["elapsed"]) for r in results)
    conv_times = sorted(float(r.get("conversation_elapsed", 0.0)) for r in results)
    planner_times = sorted(float(r.get("planner_elapsed", 0.0)) for r in results)

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
        return values[idx]

    print()
    print("=" * 100)
    print("STUDENT CONVERSATION / QUERY PLANNER DIAGNOSTIC V3")
    print("=" * 100)
    print(f"CASES: {total}")
    print(f"OVERALL: {passed}/{total} PASS ({_rate(passed, total):.1f}%) | {failed}/{total} FAIL")
    print()

    print("STAGE SCORECARD")
    print("-" * 100)
    print(f"{'Stage':<24}{'PASS':>8}{'FAIL':>8}{'RATE':>10}")
    print("-" * 100)
    for stage in STAGES:
        ok, bad = stage_stats[stage]
        print(f"{stage:<24}{ok:>8}{bad:>8}{_rate(ok, total):>9.1f}%")
    print()

    print("LLM / RESOLUTION PROFILE")
    print("-" * 100)
    print(f"Conversation deterministic / explicit : {deterministic}")
    print(f"Conversation LLM / rejected path      : {conv_llm}")
    print(f"Conversation ambiguity-preserved      : {conv_ambiguous}")
    print(f"Planner LLM opportunities             : {total}")
    print()

    print("ROOT-CAUSE OVERVIEW")
    print("-" * 100)
    if roots:
        for root, count in roots.most_common():
            print(f"{root:<24}{count:>4} case(s)")
    else:
        print("No contract failures detected.")
    print()

    print("PERFORMANCE")
    print("-" * 100)
    print(f"Median total case time                : {percentile(times, 0.50):.2f}s")
    print(f"P95 total case time                   : {percentile(times, 0.95):.2f}s")
    print(f"Median conversation time              : {percentile(conv_times, 0.50):.2f}s")
    print(f"P95 conversation time                 : {percentile(conv_times, 0.95):.2f}s")
    print(f"Median planner time                   : {percentile(planner_times, 0.50):.2f}s")
    print(f"P95 planner time                      : {percentile(planner_times, 0.95):.2f}s")
    print()

    print("SLOWEST 5 CASES")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["elapsed"], reverse=True)[:5]:
        print(
            f"{r['case'].case_id:<6} "
            f"{r['elapsed']:>8.2f}s | "
            f"conv={r.get('conversation_elapsed', 0.0):>7.2f}s | "
            f"planner={r.get('planner_elapsed', 0.0):>7.2f}s | "
            f"{r['primary_failure']}"
        )
    print()

    if roots:
        primary_root = roots.most_common(1)[0][0]
        print("ENGINEERING INTERPRETATION")
        print("-" * 100)
        print(f"Primary observed breakpoint : {primary_root}")
        if primary_root == "conversation":
            print("Focus next on conversation resolution / API behavior.")
        elif primary_root in {"scope", "intent_count", "target", "type", "attributes", "constraints", "negation", "comparison"}:
            print("Focus next on planner normalization/reconciliation and contract preservation.")
        elif primary_root == "retrieval_query":
            print("Conversation + planner contracts mostly hold; retrieval-query formulation is the next target.")
        elif primary_root == "planner":
            print("Focus next on planner execution/parsing/fallback behavior.")
    else:
        print("ENGINEERING INTERPRETATION")
        print("-" * 100)
        print("No contract break observed in this suite.")
        print("Do not freeze blindly; expand the adversarial corpus before production freeze.")

    print()
    print("VERDICT")
    print("-" * 100)
    if passed == total:
        print("PASS — all tested contracts passed.")
    else:
        print(
            f"NOT READY TO FREEZE — {failed} case(s) expose at least one contract failure. "
            "Fix the earliest/root breakpoint before moving downstream."
        )
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone adversarial diagnostic for conversation + query planner."
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Run first N cases only; 0 runs all 60.",
    )
    parser.add_argument(
        "--show-passes",
        action="store_true",
        help="Also print compact PASS diagnostics.",
    )
    args = parser.parse_args()

    if args.max_cases < 0:
        raise SystemExit("--max-cases must be >= 0")

    cases = CASES[:args.max_cases] if args.max_cases else CASES
    results: list[dict[str, Any]] = []

    print("=" * 100, flush=True)
    print("STUDENT CONVERSATION / QUERY PLANNER DIAGNOSTIC V3", flush=True)
    print("=" * 100, flush=True)
    print(f"Running {len(cases)} adversarial cases...", flush=True)
    print("Production code changed: NONE", flush=True)
    print()

    for idx, case in enumerate(cases, start=1):
        result = evaluate_case(case)
        results.append(result)
        if args.show_passes or result["primary_failure"] != "PASS":
            print_case_result(idx, len(cases), result)
        else:
            print_case_result(idx, len(cases), result)

    print_report(results)
    return 0 if all(r["primary_failure"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
