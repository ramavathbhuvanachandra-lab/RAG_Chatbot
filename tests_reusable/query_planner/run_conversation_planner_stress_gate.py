from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.planner import build_query_plan


@dataclass(frozen=True)
class Case:
    name: str
    history: Sequence[dict[str, str]]
    question: str
    expected_resolved: str | None = None
    expected_mode: str | None = None
    expected_target: str | None = None
    expected_attributes: tuple[str, ...] = ()
    expected_type: str | None = None
    expected_multi: bool | None = None


CASES = (
    Case(
        name="follow_up_fee_same_program",
        history=(
            {"role": "user", "content": "What are the admission requirements for M.Tech?"},
            {"role": "assistant", "content": "The eligibility depends on the admission route."},
        ),
        question="What about the application fee for that program?",
        expected_resolved="What about the application fee for M.Tech?",
        expected_mode="follow_up",
        expected_target="M.Tech",
        expected_attributes=("application fee",),
        expected_type="fee",
        expected_multi=False,
    ),
    Case(
        name="follow_up_documents_same_program",
        history=(
            {"role": "user", "content": "How do I apply for MBA?"},
            {"role": "assistant", "content": "You need to follow the MBA admission process."},
        ),
        question="What documents do I need?",
        expected_resolved="What documents do I need for MBA?",
        expected_mode="follow_up",
        expected_target="MBA",
        expected_attributes=("documents",),
        expected_type="documents",
        expected_multi=False,
    ),
    Case(
        name="follow_up_deadline_same_program",
        history=(
            {"role": "user", "content": "Tell me about M.Sc eligibility."},
            {"role": "assistant", "content": "Here are the eligibility requirements."},
        ),
        question="And what about the deadline?",
        expected_resolved="What about the deadline for M.Sc?",
        expected_mode="follow_up",
        expected_target="M.Sc",
        expected_attributes=("deadline",),
        expected_type="deadline",
        expected_multi=False,
    ),
    Case(
        name="follow_up_work_experience",
        history=(
            {"role": "user", "content": "What are the requirements for M.Tech?"},
            {"role": "assistant", "content": "The program has specific eligibility criteria."},
        ),
        question="Is work experience required?",
        expected_resolved="Is work experience required for M.Tech?",
        expected_mode="follow_up",
        expected_target="M.Tech",
        expected_attributes=("work experience",),
        expected_type="eligibility",
        expected_multi=False,
    ),
    Case(
        name="explicit_topic_switch",
        history=(
            {"role": "user", "content": "What is the M.Tech application fee?"},
            {"role": "assistant", "content": "The fee depends on the applicable category."},
        ),
        question="What documents are required for PhD admission?",
        expected_resolved="What documents are required for PhD admission?",
        expected_mode="standalone",
        expected_target="PhD",
        expected_attributes=("documents",),
        expected_type="documents",
        expected_multi=False,
    ),
    Case(
        name="standalone_fee",
        history=(
            {"role": "user", "content": "What is the M.Tech eligibility?"},
        ),
        question="What is the application fee for the PhD program?",
        expected_resolved="What is the application fee for the PhD program?",
        expected_mode="standalone",
        expected_target="PhD",
        expected_attributes=("application fee",),
        expected_type="fee",
        expected_multi=False,
    ),
    Case(
        name="standalone_multi_three",
        history=(),
        question="What is M.Tech eligibility, what documents are required, and what is the application fee?",
        expected_mode="standalone",
        expected_multi=True,
    ),
    Case(
        name="comparison",
        history=(),
        question="What is the difference between M.Tech and MBA admission processes?",
        expected_mode="standalone",
        expected_target="M.Tech and MBA",
        expected_attributes=("admission process",),
        expected_type="comparison",
        expected_multi=False,
    ),
    Case(
        name="negation",
        history=(),
        question="Is work experience not required for M.Tech admission?",
        expected_mode="standalone",
        expected_target="M.Tech",
        expected_attributes=("work experience",),
        expected_type="eligibility",
        expected_multi=False,
    ),
    Case(
        name="eligibility_with_constraints",
        history=(),
        question="I qualified GATE, but my undergraduate percentage is below the usual threshold. Can I still apply for M.Tech and are there any exceptions?",
        expected_mode="standalone",
        expected_target="M.Tech",
        expected_type="eligibility",
        expected_multi=False,
    ),
    Case(
        name="first_program_follow_up",
        history=(
            {"role": "user", "content": "Compare M.Tech and MBA admission processes."},
            {"role": "assistant", "content": "They differ in their admission requirements and process."},
        ),
        question="What about the application fee for the first program?",
        expected_resolved="What about the application fee for M.Tech?",
        expected_mode="follow_up",
        expected_target="M.Tech",
        expected_attributes=("application fee",),
        expected_type="fee",
        expected_multi=False,
    ),
    Case(
        name="second_program_follow_up",
        history=(
            {"role": "user", "content": "Compare M.Tech and MBA admission processes."},
            {"role": "assistant", "content": "They differ in their admission requirements and process."},
        ),
        question="What about the deadline for the second program?",
        expected_resolved="What about the deadline for MBA?",
        expected_mode="follow_up",
        expected_target="MBA",
        expected_attributes=("deadline",),
        expected_type="deadline",
        expected_multi=False,
    ),
    Case(
        name="ambiguous_reference_should_not_guess",
        history=(
            {"role": "user", "content": "Tell me about M.Tech and MBA."},
            {"role": "assistant", "content": "Both are postgraduate programs."},
        ),
        question="What about the application fee for that program?",
        expected_mode="standalone",
    ),
    Case(
        name="out_of_scope",
        history=(
            {"role": "user", "content": "What is M.Tech eligibility?"},
        ),
        question="Who won yesterday's cricket match?",
        expected_mode="standalone",
    ),
    Case(
        name="long_eligibility_plus_work_experience",
        history=(),
        question="I am currently in my final year of B.Tech and I want to apply for M.Tech after graduation. I have a CGPA of 7.1 and I have qualified GATE. Please tell me whether I am eligible and whether any work experience is required.",
        expected_mode="standalone",
        expected_target="M.Tech",
        expected_type="eligibility",
        expected_multi=True,
    ),
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _contains_all(actual: Sequence[str], expected: Sequence[str]) -> bool:
    actual_norm = {_clean(v).casefold() for v in actual}
    return all(_clean(v).casefold() in actual_norm for v in expected)


def main() -> None:
    print("=" * 100)
    print("CONVERSATION -> QUERY PLANNER PRODUCTION STRESS GATE")
    print("=" * 100)
    print(f"CASES: {len(CASES)}")
    print()

    failures: list[str] = []

    for index, case in enumerate(CASES, start=1):
        started = time.perf_counter()

        conv = resolve_conversation(
            question=case.question,
            chat_history=case.history,
        )
        resolved = _clean(conv.get("resolved_question"))
        mode = _clean(conv.get("mode")).casefold()

        plan = build_query_plan(resolved)

        elapsed = time.perf_counter() - started

        intent = plan.intents[0] if len(plan.intents) == 1 else None
        actual_target = _clean(getattr(intent, "target", None)) if intent else ""
        actual_attrs = tuple(getattr(intent, "requested_attributes", ()) or ()) if intent else ()
        actual_type = _clean(getattr(intent, "request_type", None)) if intent else ""
        all_targets = [_clean(getattr(item, "target", None)) for item in plan.intents]
        all_types = [_clean(getattr(item, "request_type", None)) for item in plan.intents]

        checks: list[str] = []

        if case.expected_resolved is not None:
            checks.append(
                "resolved"
                if resolved.casefold() == case.expected_resolved.casefold()
                else "FAIL:resolved"
            )

        if case.expected_mode is not None:
            checks.append(
                "mode"
                if mode == case.expected_mode.casefold()
                else "FAIL:mode"
            )

        if case.expected_target is not None:
            if case.expected_multi:
                checks.append(
                    "target"
                    if all_targets and all(value.casefold() == case.expected_target.casefold() for value in all_targets)
                    else "FAIL:target"
                )
            else:
                checks.append(
                    "target"
                    if actual_target.casefold() == case.expected_target.casefold()
                    else "FAIL:target"
                )

        if case.expected_attributes:
            checks.append(
                "attributes"
                if _contains_all(actual_attrs, case.expected_attributes)
                else "FAIL:attributes"
            )

        if case.expected_type is not None:
            if case.expected_multi:
                checks.append(
                    "type"
                    if all_types and all(value == case.expected_type for value in all_types)
                    else "FAIL:type"
                )
            else:
                checks.append(
                    "type"
                    if actual_type == case.expected_type
                    else "FAIL:type"
                )

        if case.expected_multi is not None:
            checks.append(
                "multi"
                if bool(plan.is_multi_intent) == case.expected_multi
                else "FAIL:multi"
            )

        # For deliberately ambiguous cases, do not require a fabricated target.
        if case.name == "ambiguous_reference_should_not_guess":
            if actual_target.casefold() in {"m.tech", "mba"}:
                checks.append("FAIL:ambiguous_guess")
            else:
                checks.append("no_guess")

        passed = not any(item.startswith("FAIL:") for item in checks)

        print("-" * 100)
        print(f"CASE {index}: {case.name}")
        print(f"QUESTION: {case.question}")
        print(f"RESOLVED: {resolved}")
        print(f"MODE: {mode}")
        print(f"CONVERSATION SOURCE: {_clean(conv.get('source')) or 'n/a'}")
        print(f"PLAN SCOPE: {plan.domain_decision} ({plan.domain_confidence:.2f})")
        print(f"MULTI: {plan.is_multi_intent}  INTENTS: {len(plan.intents)}")

        for n, item in enumerate(plan.intents, start=1):
            print(
                f"INTENT {n}: "
                f"type={_clean(getattr(item, 'request_type', ''))} "
                f"target={_clean(getattr(item, 'target', ''))} "
                f"attrs={list(getattr(item, 'requested_attributes', ()) or ())} "
                f"constraints={list(getattr(item, 'constraints', ()) or ())} "
                f"retrieval={_clean(getattr(item, 'retrieval_query', ''))}"
            )

        print(f"CHECKS: {', '.join(checks) if checks else 'structural-only'}")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")
        print(f"TIME: {elapsed:.2f}s")

        if not passed:
            failures.append(case.name)

    print()
    print("=" * 100)
    if failures:
        print("STRESS GATE: FAIL")
        print("FAILURES:")
        for name in failures:
            print(f"  - {name}")
        print("=" * 100)
        raise SystemExit(1)

    print(f"STRESS GATE: PASS ({len(CASES)}/{len(CASES)})")
    print("Conversation + Query Planner boundary passed all asserted contracts.")
    print("Ready to freeze this layer and move to verification/ranking.")
    print("=" * 100)


if __name__ == "__main__":
    main()
