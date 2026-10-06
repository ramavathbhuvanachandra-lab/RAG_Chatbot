#!/usr/bin/env python3
"""
Conversation -> Query Planner ADVERSARIAL 50-question stress gate.

TEST ONLY. Does not modify production code.

This is intentionally NOT a "make the tests easy to pass" suite.
It includes:
- realistic college-student wording
- short follow-ups
- Hindi / Hinglish / Devanagari
- code-switching
- typos / colloquial phrasing
- long constraint-heavy questions
- reference resolution
- topic switching
- comparisons
- negation
- ambiguity
- multi-intent decomposition
- entity/attribute preservation
- prompt-injection-shaped user text
- out-of-scope requests

Run from repo root:
    PYTHONPATH=. python -u \
      tests_reusable/query_planner/run_conversation_planner_adversarial_50q_stress_gate_v3.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Mapping

from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.planner import build_query_plan


@dataclass
class IntentRule:
    type: Optional[str] = None
    target_contains: List[str] = field(default_factory=list)
    attributes_any: List[str] = field(default_factory=list)
    constraints_any: List[str] = field(default_factory=list)


@dataclass
class Case:
    name: str
    question: str
    history: List[Dict[str, str]] = field(default_factory=list)

    expected_mode: Optional[str] = None
    expected_entity: Optional[str] = None
    expected_resolved_contains: List[str] = field(default_factory=list)
    forbidden_resolved_entities: List[str] = field(default_factory=list)

    expected_scope: Optional[str] = None
    expected_multi: Optional[bool] = None
    min_intents: Optional[int] = None
    max_intents: Optional[int] = None

    # For each rule, at least one produced intent must satisfy the rule.
    intent_rules: List[IntentRule] = field(default_factory=list)

    # Useful for ambiguity / prompt-injection tests.
    forbidden_plan_targets: List[str] = field(default_factory=list)


def H(*pairs: tuple[str, str]) -> List[Dict[str, str]]:
    return [{"role": role, "content": content} for role, content in pairs]


def rule(
    type: Optional[str] = None,
    target: Optional[str] = None,
    attrs: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
) -> IntentRule:
    return IntentRule(
        type=type,
        target_contains=[] if target is None else [target],
        attributes_any=attrs or [],
        constraints_any=constraints or [],
    )


CASES: List[Case] = [
    # ================================================================
    # A. Realistic standalone student questions
    # ================================================================
    Case(
        "01_mtech_fee_realistic",
        "What exactly is the application fee for M.Tech, and is it paid during the online application itself?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "02_mba_docs_realistic",
        "For the MBA application, which documents do I have to upload and which ones are only needed later at verification?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "MBA", ["documents"])],
    ),
    Case(
        "03_msc_deadline_realistic",
        "What's the last date by which an M.Sc applicant has to submit the application?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "M.Sc", ["deadline"])],
    ),
    Case(
        "04_phd_eligibility_realistic",
        "What are the actual eligibility requirements for PhD admission, not just the general admission process?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "PhD", ["eligibility"])],
    ),
    Case(
        "05_mtech_workexp_realistic",
        "For M.Tech admissions, does prior industry work experience have any mandatory role?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "M.Tech", ["work experience"])],
    ),
    Case(
        "06_mba_apply_realistic",
        "I know the MBA eligibility already. I only need the steps for submitting the application.",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("process", "MBA", ["application process"])],
    ),
    Case(
        "07_mtech_registration_docs",
        "Which documents are required specifically for M.Tech registration, rather than for the initial application?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "M.Tech", ["documents"])],
    ),
    Case(
        "08_phd_application_fee",
        "How much is the PhD application processing fee?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "PhD", ["application fee"])],
    ),
    Case(
        "09_mtech_admission_process",
        "Walk me through the M.Tech admission process from application to selection.",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("process", "M.Tech", ["admission process"])],
    ),
    Case(
        "10_mba_eligibility",
        "I have a non-engineering bachelor's degree. Before anything else, am I eligible for the MBA program?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "MBA", ["eligibility"])],
    ),

    # ================================================================
    # B. Very short / elliptical follow-ups
    # ================================================================
    Case(
        "11_followup_fee_oneword",
        "Fee?",
        H(("user", "What are the admission requirements for M.Tech?")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "12_followup_documents_oneword",
        "Documents?",
        H(("user", "How do I apply for MBA?")),
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "MBA", ["documents"])],
    ),
    Case(
        "13_followup_deadline_oneword",
        "Deadline?",
        H(("user", "Tell me about M.Sc eligibility.")),
        expected_mode="follow_up",
        expected_entity="M.Sc",
        expected_resolved_contains=["M.Sc"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "M.Sc", ["deadline"])],
    ),
    Case(
        "14_followup_short_workexp",
        "Is experience compulsory?",
        H(("user", "I'm planning to apply for M.Tech.")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "M.Tech", ["work experience"])],
    ),
    Case(
        "15_followup_short_apply",
        "How do I apply?",
        H(("user", "What is the MBA eligibility?")),
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("process", "MBA", ["application process"])],
    ),
    Case(
        "16_followup_short_lastdate",
        "When's the last date?",
        H(("user", "How does M.Tech admission work?")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "M.Tech", ["deadline"])],
    ),
    Case(
        "17_followup_that_program",
        "What about the application fee for that program?",
        H(("user", "What are the admission requirements for M.Tech?")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "18_followup_and_docs",
        "And the documents?",
        H(("user", "What is the M.Tech application fee?")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "M.Tech", ["documents"])],
    ),

    # ================================================================
    # C. Context switching and reference resolution traps
    # ================================================================
    Case(
        "19_explicit_switch_mtech_to_phd",
        "Forget M.Tech for a moment. What documents are required for PhD?",
        H(("user", "Tell me about M.Tech admission.")),
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "PhD", ["documents"])],
    ),
    Case(
        "20_explicit_switch_mba_to_msc",
        "I was asking about MBA earlier, but now tell me the M.Sc deadline.",
        H(("user", "How do I apply for MBA?")),
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "M.Sc", ["deadline"])],
    ),
    Case(
        "21_first_program_reference",
        "What about the fee for the first one?",
        H(
            ("user", "Compare M.Tech and MBA admission processes."),
        ),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "22_second_program_reference",
        "And the deadline for the second one?",
        H(
            ("user", "Compare M.Tech and MBA admission processes."),
        ),
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "MBA", ["deadline"])],
    ),
    Case(
        "23_ambiguous_that_program_no_guess",
        "What about that program's fee?",
        H(
            ("user", "Tell me about M.Tech."),
            ("assistant", "M.Tech has its own admission information."),
            ("user", "MBA also has a separate process."),
        ),
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        forbidden_resolved_entities=["M.Tech", "MBA"],
        forbidden_plan_targets=["M.Tech", "MBA"],
    ),
    Case(
        "24_pronoun_after_single_entity",
        "Can I apply for it this year?",
        H(("user", "How do I apply for MBA?")),
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("process", "MBA")],
    ),
    Case(
        "25_topic_change_without_entity",
        "Okay, now what about the deadline?",
        H(
            ("user", "What are M.Tech eligibility requirements?"),
            ("assistant", "Here are the eligibility requirements."),
        ),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "M.Tech", ["deadline"])],
    ),

    # ================================================================
    # D. Hindi / Hinglish standalone
    # ================================================================
    Case(
        "26_hindi_mtech_fee",
        "M.Tech ki application fee kitni hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "27_hindi_mba_documents",
        "MBA ke liye kaun-kaun se documents chahiye?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "MBA", ["documents"])],
    ),
    Case(
        "28_hindi_msc_deadline",
        "M.Sc ke form ki last date kya hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "M.Sc", ["deadline"])],
    ),
    Case(
        "29_hindi_phd_eligibility",
        "PhD ke liye eligibility kya hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "PhD", ["eligibility"])],
    ),
    Case(
        "30_hinglish_workexp",
        "M.Tech ke liye work experience zaroori hai kya?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "M.Tech", ["work experience"])],
    ),
    Case(
        "31_devanagari_fee",
        "M.Tech की आवेदन फीस कितनी है?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "32_devanagari_docs",
        "MBA के लिए कौन से डॉक्यूमेंट चाहिए?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "MBA", ["documents"])],
    ),
    Case(
        "33_devanagari_eligibility",
        "PhD में दाखिले की पात्रता क्या है?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "PhD", ["eligibility"])],
    ),
    Case(
        "34_hindi_application_process",
        "M.Sc ke liye apply kaise karna hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("process", "M.Sc", ["application process"])],
    ),
    Case(
        "35_code_mixed_question",
        "M.Tech mein 60 percent se kam marks hain, lekin GATE qualified hoon. Eligible hoon kya?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[
            rule(
                "eligibility",
                "M.Tech",
                ["eligibility"],
                ["GATE", "60 percent", "less than"],
            )
        ],
    ),

    # ================================================================
    # E. Hindi / Hinglish follow-ups
    # ================================================================
    Case(
        "36_hindi_followup_fee",
        "फीस कितनी है?",
        H(("user", "What are the M.Tech admission requirements?")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "37_hindi_followup_docs",
        "documents kaunse chahiye?",
        H(("user", "How do I apply for MBA?")),
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("documents", "MBA", ["documents"])],
    ),
    Case(
        "38_hindi_followup_deadline",
        "deadline kab hai?",
        H(("user", "What is the M.Sc eligibility?")),
        expected_mode="follow_up",
        expected_entity="M.Sc",
        expected_resolved_contains=["M.Sc"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("deadline", "M.Sc", ["deadline"])],
    ),
    Case(
        "39_hindi_followup_eligibility",
        "पात्रता क्या है?",
        H(("user", "Tell me about PhD admission.")),
        expected_mode="follow_up",
        expected_entity="PhD",
        expected_resolved_contains=["PhD"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "PhD", ["eligibility"])],
    ),
    Case(
        "40_hindi_followup_process",
        "आवेदन कैसे करें?",
        H(("user", "What is the MBA eligibility?")),
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("process", "MBA", ["application process"])],
    ),
    Case(
        "41_hinglish_followup_workexp",
        "experience compulsory hai?",
        H(("user", "What are the M.Tech admission requirements?")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "M.Tech", ["work experience"])],
    ),

    # ================================================================
    # F. High-information / constraint-heavy student queries
    # ================================================================
    Case(
        "42_constraint_gate_and_low_percentage",
        "I completed B.Tech with 59%, I have a valid GATE score, and I fall under EWS. Can I still apply for M.Tech, and does any category-specific relaxation apply?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[
            rule(
                "eligibility",
                "M.Tech",
                ["eligibility"],
                ["59%", "GATE", "EWS", "relaxation"],
            )
        ],
    ),
    Case(
        "43_constraint_final_year",
        "I'm in the final semester of B.Tech and my final result is not out yet. Can I submit an M.Tech application now, or do I have to wait for graduation?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[
            rule(
                "eligibility",
                "M.Tech",
                ["eligibility", "application"],
                ["final semester", "result", "graduation"],
            )
        ],
    ),
    Case(
        "44_numeric_followup",
        "Is 7.1 CGPA enough?",
        H(("user", "What are the M.Tech eligibility requirements?")),
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "M.Tech", ["CGPA", "eligibility"])],
    ),
    Case(
        "45_fee_plus_documents_complex",
        "For M.Tech, tell me the application fee and also which supporting documents I need to upload at the time of submission.",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=True,
        min_intents=2,
        max_intents=3,
        intent_rules=[
            rule("fee", "M.Tech", ["application fee"]),
            rule("documents", "M.Tech", ["documents"]),
        ],
    ),
    Case(
        "46_eligibility_plus_workexp_complex",
        "I'm in my final year of B.Tech with 7.1 CGPA and a GATE qualification. Please tell me whether I'm eligible for M.Tech and separately whether work experience is compulsory.",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=True,
        min_intents=2,
        max_intents=3,
        intent_rules=[
            rule("eligibility", "M.Tech", ["eligibility"]),
            rule("eligibility", "M.Tech", ["work experience"]),
        ],
    ),
    Case(
        "47_three_intents_natural_language",
        "Before I apply for MBA, I want to know the eligibility, the documents I should keep ready, and the final application deadline.",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=True,
        min_intents=3,
        max_intents=4,
        intent_rules=[
            rule("eligibility", "MBA", ["eligibility"]),
            rule("documents", "MBA", ["documents"]),
            rule("deadline", "MBA", ["deadline"]),
        ],
    ),

    # ================================================================
    # G. Semantic traps: negation / exclusion / comparison / qualifiers
    # ================================================================
    Case(
        "48_negation_fee_only",
        "I'm not asking about M.Tech eligibility. I only want to know the application fee.",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("fee", "M.Tech", ["application fee"])],
    ),
    Case(
        "49_negation_workexp",
        "Is it true that work experience is not compulsory for M.Tech admission?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("eligibility", "M.Tech", ["work experience"], ["not required", "not compulsory"])],
    ),
    Case(
        "50_comparison_process",
        "M.Tech aur MBA dono ka admission process compare karke batao. Kis mein GATE ka role hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        intent_rules=[rule("comparison", "M.Tech and MBA", ["admission process", "GATE"])],
    ),
]

# NOTE:
# The exact terminology used by the planner may differ slightly from the expected
# natural-language attributes above. Matching below is deliberately substring-based.
# This keeps the gate focused on semantic coverage rather than one exact phrase.


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from dict/Mapping or object/dataclass."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _assert_production_api_compatibility() -> None:
    """Fail clearly on test-harness/API mismatch, not with an opaque AttributeError."""
    import inspect

    resolver_sig = inspect.signature(resolve_conversation)
    planner_sig = inspect.signature(build_query_plan)

    if "question" not in resolver_sig.parameters:
        raise RuntimeError(
            "TEST HARNESS/API MISMATCH: resolve_conversation must expose `question`."
        )
    if "chat_history" not in resolver_sig.parameters:
        raise RuntimeError(
            "TEST HARNESS/API MISMATCH: resolve_conversation must expose `chat_history`."
        )
    if "question" not in planner_sig.parameters:
        raise RuntimeError(
            "TEST HARNESS/API MISMATCH: build_query_plan must expose `question`."
        )



def _contains(value: Any, needle: str) -> bool:
    return _norm(needle) in _norm(value)


def _intent_list(plan: Any) -> List[Any]:
    return list(_get(plan, "intents", None) or [])


def _rule_matches_intent(rule_obj: IntentRule, intent: Any) -> bool:
    if rule_obj.type is not None and _get(intent, "type", None) != rule_obj.type:
        return False

    target = _get(intent, "target", None)
    for wanted in rule_obj.target_contains:
        if not _contains(target, wanted):
            return False

    attrs = list(_get(intent, "attributes", None) or [])
    if rule_obj.attributes_any:
        if not any(
            any(_contains(attr, wanted) for wanted in rule_obj.attributes_any)
            for attr in attrs
        ):
            return False

    constraints = list(_get(intent, "constraints", None) or [])
    if rule_obj.constraints_any:
        if not any(
            any(_contains(c, wanted) for wanted in rule_obj.constraints_any)
            for c in constraints
        ):
            return False

    return True


def _check_case(case: Case):
    # Production conversation API is keyword-only.
    resolved = resolve_conversation(
        question=case.question,
        chat_history=case.history,
    )

    resolved_question = _get(resolved, "resolved_question", case.question)
    plan = build_query_plan(question=resolved_question)

    failures: List[str] = []
    intents = _intent_list(plan)

    mode = _get(resolved, "mode", None)
    entity = _get(resolved, "active_entity", _get(resolved, "entity", None))
    scope = _get(plan, "scope", None)
    multi = bool(_get(plan, "multi", False))

    if case.expected_mode is not None and mode != case.expected_mode:
        failures.append(f"mode expected={case.expected_mode!r} got={mode!r}")

    if case.expected_entity is not None and not _contains(entity, case.expected_entity):
        failures.append(
            f"conversation_entity expected={case.expected_entity!r} got={entity!r}"
        )

    for token in case.expected_resolved_contains:
        if not _contains(resolved_question, token):
            failures.append(
                f"resolved_missing={token!r} got={resolved_question!r}"
            )

    for forbidden in case.forbidden_resolved_entities:
        if _contains(resolved_question, forbidden):
            failures.append(
                f"ambiguous_guess_in_resolution={forbidden!r}"
            )

    if case.expected_scope is not None and scope != case.expected_scope:
        failures.append(
            f"scope expected={case.expected_scope!r} got={scope!r}"
        )

    if case.expected_multi is not None and multi != case.expected_multi:
        failures.append(
            f"multi expected={case.expected_multi!r} got={multi!r}"
        )

    if case.min_intents is not None and len(intents) < case.min_intents:
        failures.append(
            f"intent_count_min={case.min_intents} got={len(intents)}"
        )

    if case.max_intents is not None and len(intents) > case.max_intents:
        failures.append(
            f"intent_count_max={case.max_intents} got={len(intents)}"
        )

    for forbidden in case.forbidden_plan_targets:
        for intent in intents:
            if _contains(_get(intent, "target", None), forbidden):
                failures.append(
                    f"ambiguous_guess_in_plan={forbidden!r}"
                )

    for idx, expected_rule in enumerate(case.intent_rules, 1):
        matched = any(
            _rule_matches_intent(expected_rule, intent)
            for intent in intents
        )
        if not matched:
            compact = []
            for intent in intents:
                compact.append(
                    {
                        "type": _get(intent, "type", None),
                        "target": _get(intent, "target", None),
                        "attributes": _get(intent, "attributes", None),
                        "constraints": _get(intent, "constraints", None),
                    }
                )
            failures.append(
                f"intent_rule_{idx}_not_matched "
                f"expected={expected_rule!r} got={compact!r}"
            )

    return (not failures, failures, resolved, plan)


def main() -> int:
    print("=" * 112)
    print("CONVERSATION -> QUERY PLANNER ADVERSARIAL 50-QUESTION STRESS GATE V3")
    print("=" * 112)
    print(f"CASES: {len(CASES)}")
    print("This suite is designed to BREAK weak conversation/planner behavior, not merely produce PASS results.")
    print("Coverage: realistic student language | short follow-ups | Hindi/Hinglish | constraints |")
    print("reference resolution | context switching | comparison | negation | ambiguity | multi-intent | OOS.")
    print()

    passed = 0
    failed = []

    for number, case in enumerate(CASES, 1):
        ok, failures, resolved, plan = _check_case(case)
        intents = _intent_list(plan)

        print("-" * 112)
        print(f"CASE {number:02d}: {case.name}")
        print(f"QUESTION: {case.question}")
        print(f"RESOLVED: {_get(resolved, 'resolved_question', None)}")
        print(f"MODE: {_get(resolved, 'mode', None)}")
        print(f"CONVERSATION ENTITY: {_get(resolved, 'active_entity', _get(resolved, 'entity', None))}")
        print(
            f"PLAN: scope={_get(plan, 'scope', None)} "
            f"confidence={_get(plan, 'scope_confidence', None)} "
            f"multi={_get(plan, 'multi', None)} intents={len(intents)}"
        )

        for idx, intent in enumerate(intents, 1):
            print(
                f"  INTENT {idx}: "
                f"type={_get(intent, 'type', None)} "
                f"target={_get(intent, 'target', None)} "
                f"attrs={_get(intent, 'attributes', None)} "
                f"constraints={_get(intent, 'constraints', None)} "
                f"retrieval={_get(intent, 'retrieval_query', None)}"
            )

        if ok:
            print("RESULT: PASS")
            passed += 1
        else:
            print("RESULT: FAIL")
            for failure in failures:
                print(f"  - {failure}")
            failed.append(case.name)

    print()
    print("=" * 112)
    if not failed:
        print(f"ADVERSARIAL STRESS GATE: PASS ({passed}/{len(CASES)})")
        print("The current Conversation -> Query Planner path passed all adversarial contracts.")
    else:
        print(f"ADVERSARIAL STRESS GATE: FAIL ({passed}/{len(CASES)})")
        print("FAILURES:")
        for name in failed:
            print(f"  - {name}")
        print()
        print("Do NOT fix by weakening the tests.")
        print("Use the failing cases to identify the exact conversation/planner boundary that breaks.")
    print("=" * 112)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
