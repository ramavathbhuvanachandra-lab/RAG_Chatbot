
#!/usr/bin/env python3
"""
Conversation -> Query Planner 40-question stress gate.

TEST ONLY.
Does not modify production code.

Run from repo root:
    PYTHONPATH=. python -u tests_reusable/query_planner/run_conversation_planner_40q_stress_gate.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.planner import build_query_plan


@dataclass
class Case:
    name: str
    question: str
    history: List[Dict[str, str]] = field(default_factory=list)

    # Resolution checks
    expected_mode: Optional[str] = None
    expected_entity: Optional[str] = None
    expected_resolved_contains: List[str] = field(default_factory=list)
    no_guess_entities: List[str] = field(default_factory=list)

    # Planner checks
    expected_scope: Optional[str] = None
    expected_multi: Optional[bool] = None
    expected_intent_count: Optional[int] = None
    expected_type: Optional[str] = None
    expected_target: Optional[str] = None
    expected_attributes: List[str] = field(default_factory=list)
    expected_constraints: List[str] = field(default_factory=list)


CASES: List[Case] = [
    # ------------------------------------------------------------------
    # 1-10: simple standalone English
    # ------------------------------------------------------------------
    Case(
        "01_simple_mtech_fee",
        "What is the M.Tech application fee?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="M.Tech",
        expected_attributes=["application fee"],
    ),
    Case(
        "02_simple_mba_documents",
        "What documents are needed for MBA?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="documents",
        expected_target="MBA",
        expected_attributes=["documents"],
    ),
    Case(
        "03_simple_msc_deadline",
        "What is the M.Sc deadline?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="deadline",
        expected_target="M.Sc",
        expected_attributes=["deadline"],
    ),
    Case(
        "04_simple_phd_eligibility",
        "What is the PhD eligibility?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="PhD",
        expected_attributes=["eligibility"],
    ),
    Case(
        "05_simple_mtech_work_experience",
        "Is work experience required for M.Tech?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="M.Tech",
        expected_attributes=["work experience"],
    ),
    Case(
        "06_simple_mba_apply",
        "How do I apply for MBA?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="process",
        expected_target="MBA",
        expected_attributes=["application process"],
    ),
    Case(
        "07_simple_mtech_registration_docs",
        "What are the registration documents for M.Tech?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="documents",
        expected_target="M.Tech",
        expected_attributes=["documents"],
    ),
    Case(
        "08_simple_phd_fee",
        "What is the PhD application fee?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="PhD",
        expected_attributes=["application fee"],
    ),
    Case(
        "09_simple_mtech_admission",
        "How does M.Tech admission work?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="process",
        expected_target="M.Tech",
        expected_attributes=["admission process"],
    ),
    Case(
        "10_simple_mba_eligibility",
        "Who is eligible for MBA?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="MBA",
        expected_attributes=["eligibility"],
    ),

    # ------------------------------------------------------------------
    # 11-20: very short follow-ups / conversational ellipsis
    # ------------------------------------------------------------------
    Case(
        "11_followup_fee",
        "Fee?",
        [{"role": "user", "content": "What are the admission requirements for M.Tech?"}],
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="M.Tech",
        expected_attributes=["application fee"],
    ),
    Case(
        "12_followup_documents",
        "Documents?",
        [{"role": "user", "content": "How do I apply for MBA?"}],
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="documents",
        expected_target="MBA",
        expected_attributes=["documents"],
    ),
    Case(
        "13_followup_deadline",
        "Deadline?",
        [{"role": "user", "content": "What is the M.Sc eligibility?"}],
        expected_mode="follow_up",
        expected_entity="M.Sc",
        expected_resolved_contains=["M.Sc"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="deadline",
        expected_target="M.Sc",
        expected_attributes=["deadline"],
    ),
    Case(
        "14_followup_eligibility",
        "Eligibility?",
        [{"role": "user", "content": "What documents are needed for PhD?"}],
        expected_mode="follow_up",
        expected_entity="PhD",
        expected_resolved_contains=["PhD"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="PhD",
        expected_attributes=["eligibility"],
    ),
    Case(
        "15_followup_work_experience",
        "Is work experience required?",
        [{"role": "user", "content": "What are the M.Tech admission requirements?"}],
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="M.Tech",
        expected_attributes=["work experience"],
    ),
    Case(
        "16_followup_apply",
        "How do I apply?",
        [{"role": "user", "content": "What is the MBA eligibility?"}],
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="process",
        expected_target="MBA",
        expected_attributes=["application process"],
    ),
    Case(
        "17_followup_requirements",
        "What are the requirements?",
        [{"role": "user", "content": "Tell me about PhD admission."}],
        expected_mode="follow_up",
        expected_entity="PhD",
        expected_resolved_contains=["PhD"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="PhD",
    ),
    Case(
        "18_followup_last_date",
        "When is the last date?",
        [{"role": "user", "content": "How do I apply for M.Tech?"}],
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="deadline",
        expected_target="M.Tech",
        expected_attributes=["deadline"],
    ),
    Case(
        "19_followup_more_info",
        "What about the fee?",
        [{"role": "user", "content": "What documents are needed for MBA?"}],
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="MBA",
        expected_attributes=["application fee"],
    ),
    Case(
        "20_followup_and_question",
        "And the documents?",
        [{"role": "user", "content": "What is the M.Tech application fee?"}],
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="documents",
        expected_target="M.Tech",
        expected_attributes=["documents"],
    ),

    # ------------------------------------------------------------------
    # 21-30: Hindi + Hinglish
    # ------------------------------------------------------------------
    Case(
        "21_hindi_mtech_fee",
        "M.Tech की आवेदन फीस कितनी है?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="M.Tech",
        expected_attributes=["application fee"],
    ),
    Case(
        "22_hindi_mba_documents",
        "MBA के लिए कौन से डॉक्यूमेंट चाहिए?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="documents",
        expected_target="MBA",
        expected_attributes=["documents"],
    ),
    Case(
        "23_hindi_msc_deadline",
        "M.Sc की आखिरी तारीख क्या है?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="deadline",
        expected_target="M.Sc",
        expected_attributes=["deadline"],
    ),
    Case(
        "24_hindi_phd_eligibility",
        "PhD की पात्रता क्या है?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="PhD",
        expected_attributes=["eligibility"],
    ),
    Case(
        "25_hindi_mtech_workexp",
        "M.Tech के लिए work experience जरूरी है?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="M.Tech",
        expected_attributes=["work experience"],
    ),
    Case(
        "26_hinglish_mba_apply",
        "MBA ke liye apply kaise karna hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="process",
        expected_target="MBA",
        expected_attributes=["application process"],
    ),
    Case(
        "27_hinglish_mtech_fee",
        "M.Tech ki application fee kitni hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="M.Tech",
        expected_attributes=["application fee"],
    ),
    Case(
        "28_hinglish_mba_docs",
        "MBA ke documents kaunse chahiye?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="documents",
        expected_target="MBA",
        expected_attributes=["documents"],
    ),
    Case(
        "29_hinglish_msc_deadline",
        "M.Sc ki deadline kab hai?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="deadline",
        expected_target="M.Sc",
        expected_attributes=["deadline"],
    ),
    Case(
        "30_hinglish_followup_fee",
        "फीस कितनी है?",
        [{"role": "user", "content": "What is the M.Tech eligibility?"}],
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="M.Tech",
        expected_attributes=["application fee"],
    ),

    # ------------------------------------------------------------------
    # 31-36: Hindi follow-ups / conversational context
    # ------------------------------------------------------------------
    Case(
        "31_hinglish_followup_docs",
        "documents kaunse chahiye?",
        [{"role": "user", "content": "How do I apply for MBA?"}],
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="documents",
        expected_target="MBA",
        expected_attributes=["documents"],
    ),
    Case(
        "32_hinglish_followup_deadline",
        "deadline kab hai?",
        [{"role": "user", "content": "What is the M.Sc eligibility?"}],
        expected_mode="follow_up",
        expected_entity="M.Sc",
        expected_resolved_contains=["M.Sc"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="deadline",
        expected_target="M.Sc",
        expected_attributes=["deadline"],
    ),
    Case(
        "33_hinglish_followup_workexp",
        "work experience zaroori hai?",
        [{"role": "user", "content": "What are the M.Tech admission requirements?"}],
        expected_mode="follow_up",
        expected_entity="M.Tech",
        expected_resolved_contains=["M.Tech"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="M.Tech",
        expected_attributes=["work experience"],
    ),
    Case(
        "34_hindi_followup_eligibility",
        "पात्रता क्या है?",
        [{"role": "user", "content": "Tell me about PhD admission."}],
        expected_mode="follow_up",
        expected_entity="PhD",
        expected_resolved_contains=["PhD"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="PhD",
        expected_attributes=["eligibility"],
    ),
    Case(
        "35_hindi_followup_apply",
        "आवेदन कैसे करें?",
        [{"role": "user", "content": "What is the MBA eligibility?"}],
        expected_mode="follow_up",
        expected_entity="MBA",
        expected_resolved_contains=["MBA"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="process",
        expected_target="MBA",
        expected_attributes=["application process"],
    ),
    Case(
        "36_hindi_followup_fee",
        "आवेदन शुल्क कितना है?",
        [{"role": "user", "content": "What are the PhD admission requirements?"}],
        expected_mode="follow_up",
        expected_entity="PhD",
        expected_resolved_contains=["PhD"],
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="fee",
        expected_target="PhD",
        expected_attributes=["application fee"],
    ),

    # ------------------------------------------------------------------
    # 37-42: multi-intent / comparison / negation / safety boundaries
    # ------------------------------------------------------------------
    Case(
        "37_multi_two_simple",
        "Tell me M.Tech eligibility and documents.",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=True,
        expected_intent_count=2,
        expected_target="M.Tech",
    ),
    Case(
        "38_multi_three_simple",
        "M.Tech fee, documents, and deadline?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=True,
        expected_intent_count=3,
        expected_target="M.Tech",
    ),
    Case(
        "39_comparison_simple",
        "M.Tech vs MBA: which admission process is different?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="comparison",
        expected_target="M.Tech and MBA",
        expected_attributes=["admission process"],
    ),
    Case(
        "40_negation_simple",
        "Is work experience not required for M.Tech?",
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        expected_type="eligibility",
        expected_target="M.Tech",
        expected_attributes=["work experience"],
        expected_constraints=["not required"],
    ),
    Case(
        "41_ambiguous_no_guess",
        "What about the fee for that program?",
        [
            {"role": "user", "content": "Tell me about M.Tech."},
            {"role": "assistant", "content": "M.Tech is one admission option."},
            {"role": "user", "content": "What about MBA?"}
        ],
        expected_mode="standalone",
        expected_scope="in_scope",
        expected_multi=False,
        no_guess_entities=["M.Tech", "MBA"],
    ),
    Case(
        "42_out_of_scope",
        "Who won yesterday's cricket match?",
        [{"role": "user", "content": "What is the M.Tech application fee?"}],
        expected_mode="standalone",
        expected_scope="out_of_scope",
        expected_multi=False,
    ),
]


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _contains(haystack: Any, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _intent_plan(plan: Any) -> List[Any]:
    return list(getattr(plan, "intents", None) or [])


def _check_case(case: Case) -> tuple[bool, List[str], Any, Any]:
    resolved = resolve_conversation(case.question, case.history)
    plan = build_query_plan(resolved.resolved_question, case.history)

    failures: List[str] = []

    if case.expected_mode is not None and getattr(resolved, "mode", None) != case.expected_mode:
        failures.append(f"mode expected={case.expected_mode!r} got={getattr(resolved, 'mode', None)!r}")

    entity = getattr(resolved, "entity", None)
    if case.expected_entity is not None and not _contains(entity, case.expected_entity):
        failures.append(f"entity expected={case.expected_entity!r} got={entity!r}")

    for token in case.expected_resolved_contains:
        if not _contains(getattr(resolved, "resolved_question", None), token):
            failures.append(
                f"resolved_missing={token!r} got={getattr(resolved, 'resolved_question', None)!r}"
            )

    for forbidden in case.no_guess_entities:
        # "no guess" is about not resolving or planning onto the ambiguous entity.
        if _contains(getattr(resolved, "resolved_question", None), forbidden):
            failures.append(f"guessed_entity_in_resolution={forbidden!r}")
        for intent in _intent_plan(plan):
            if _contains(getattr(intent, "target", None), forbidden):
                failures.append(f"guessed_entity_in_plan={forbidden!r}")

    if case.expected_scope is not None and getattr(plan, "scope", None) != case.expected_scope:
        failures.append(
            f"scope expected={case.expected_scope!r} got={getattr(plan, 'scope', None)!r}"
        )

    if case.expected_multi is not None and bool(getattr(plan, "multi", False)) != case.expected_multi:
        failures.append(
            f"multi expected={case.expected_multi!r} got={getattr(plan, 'multi', None)!r}"
        )

    intents = _intent_plan(plan)
    if case.expected_intent_count is not None and len(intents) != case.expected_intent_count:
        failures.append(
            f"intent_count expected={case.expected_intent_count} got={len(intents)}"
        )

    # For single-intent expectations, inspect the sole intent.
    if len(intents) == 1:
        intent = intents[0]

        if case.expected_type is not None and getattr(intent, "type", None) != case.expected_type:
            failures.append(
                f"type expected={case.expected_type!r} got={getattr(intent, 'type', None)!r}"
            )

        if case.expected_target is not None and not _contains(
            getattr(intent, "target", None), case.expected_target
        ):
            failures.append(
                f"target expected={case.expected_target!r} got={getattr(intent, 'target', None)!r}"
            )

        attrs = list(getattr(intent, "attributes", None) or [])
        for expected_attr in case.expected_attributes:
            if not any(_contains(a, expected_attr) for a in attrs):
                failures.append(
                    f"attribute_missing={expected_attr!r} got={attrs!r}"
                )

        constraints = list(getattr(intent, "constraints", None) or [])
        for expected_constraint in case.expected_constraints:
            if not any(_contains(c, expected_constraint) for c in constraints):
                failures.append(
                    f"constraint_missing={expected_constraint!r} got={constraints!r}"
                )
    elif case.expected_type is not None or case.expected_attributes or case.expected_constraints:
        # Multi-intent cases can legitimately distribute fields across intents.
        combined_types = [getattr(i, "type", None) for i in intents]
        combined_targets = [getattr(i, "target", None) for i in intents]
        combined_attrs = [
            a for i in intents for a in (getattr(i, "attributes", None) or [])
        ]
        combined_constraints = [
            c for i in intents for c in (getattr(i, "constraints", None) or [])
        ]

        if case.expected_type is not None and case.expected_type not in combined_types:
            failures.append(
                f"type_missing_from_multi={case.expected_type!r} got={combined_types!r}"
            )

        if case.expected_target is not None and not any(
            _contains(t, case.expected_target) for t in combined_targets
        ):
            failures.append(
                f"target_missing_from_multi={case.expected_target!r} got={combined_targets!r}"
            )

        for expected_attr in case.expected_attributes:
            if not any(_contains(a, expected_attr) for a in combined_attrs):
                failures.append(
                    f"attribute_missing_from_multi={expected_attr!r} got={combined_attrs!r}"
                )

        for expected_constraint in case.expected_constraints:
            if not any(_contains(c, expected_constraint) for c in combined_constraints):
                failures.append(
                    f"constraint_missing_from_multi={expected_constraint!r} got={combined_constraints!r}"
                )

    return (not failures, failures, resolved, plan)


def main() -> int:
    print("=" * 108)
    print("CONVERSATION -> QUERY PLANNER 42-QUESTION STRESS GATE")
    print("=" * 108)
    print(f"CASES: {len(CASES)}")
    print("Includes simple English, short conversational follow-ups, Hindi, Hinglish, multi-intent,")
    print("comparison, negation, ambiguity protection, and out-of-scope handling.")
    print()

    passed = 0
    failed_cases: List[str] = []

    for case in CASES:
        ok, failures, resolved, plan = _check_case(case)
        intents = _intent_plan(plan)

        print("-" * 108)
        print(f"CASE {case.name}")
        print(f"QUESTION: {case.question}")
        print(f"RESOLVED: {getattr(resolved, 'resolved_question', None)}")
        print(f"MODE: {getattr(resolved, 'mode', None)}")
        print(f"CONVERSATION ENTITY: {getattr(resolved, 'entity', None)}")
        print(
            f"PLAN SCOPE: {getattr(plan, 'scope', None)} "
            f"({getattr(plan, 'scope_confidence', None)})"
        )
        print(f"MULTI: {getattr(plan, 'multi', None)}  INTENTS: {len(intents)}")

        for idx, intent in enumerate(intents, 1):
            print(
                f"INTENT {idx}: "
                f"type={getattr(intent, 'type', None)} "
                f"target={getattr(intent, 'target', None)} "
                f"attrs={getattr(intent, 'attributes', None)} "
                f"constraints={getattr(intent, 'constraints', None)} "
                f"retrieval={getattr(intent, 'retrieval_query', None)}"
            )

        if ok:
            print("RESULT: PASS")
            passed += 1
        else:
            print("RESULT: FAIL")
            for failure in failures:
                print(f"  - {failure}")
            failed_cases.append(case.name)

    print()
    print("=" * 108)
    print(f"STRESS GATE: {'PASS' if not failed_cases else 'FAIL'} ({passed}/{len(CASES)})")
    if failed_cases:
        print("FAILURES:")
        for name in failed_cases:
            print(f"  - {name}")
    else:
        print("All 42 conversation -> planner cases passed.")
    print("=" * 108)

    return 0 if not failed_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
