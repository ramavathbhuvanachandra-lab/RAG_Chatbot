#!/usr/bin/env python3
"""
Fast semantic regression gate for Query Planner V2.4.

The fake model deliberately returns bad/noisy LLM structures. The test verifies
that deterministic reconciliation corrects them from the user's wording.
No live LLM calls are made.
"""

from __future__ import annotations

import json

from ai_platform.core.query.planner import build_query_plan


class FakeModel:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _prompt):
        return json.dumps(self.payload)


def run_case(name: str, question: str, llm_intents: list[dict], expected: list[tuple]):
    plan = build_query_plan(
        question=question,
        model=FakeModel(
            {
                "domain_decision": "in_scope",
                "domain_confidence": 0.95,
                "intents": llm_intents,
            }
        ),
    )

    observed = [
        (intent.request_type, intent.target, tuple(intent.requested_attributes))
        for intent in plan.intents
    ]

    expected_norm = [
        (request_type, target, tuple(attrs))
        for request_type, target, attrs in expected
    ]

    assert observed == expected_norm, (
        f"{name} failed\n"
        f"EXPECTED: {expected_norm}\n"
        f"OBSERVED: {observed}"
    )

    return plan


def main() -> int:
    cases = [
        (
            "final_semester_eligibility",
            (
                "I am in my final semester and my degree will be completed only after "
                "the admission process. Can I still be considered for M.Tech, or is "
                "completed graduation mandatory?"
            ),
            [
                {
                    "question": "same question",
                    "request_type": "process",
                    "target": "M.Tech",
                    "requested_attributes": ["process"],
                }
            ],
            [("eligibility", "M.Tech", ["eligibility"])],
        ),
        (
            "disclaimed_eligibility",
            (
                "I am not asking whether I am eligible for M.Tech. I only need the "
                "application fee and whether there is any separate registration charge."
            ),
            [
                {
                    "question": "fee",
                    "request_type": "fee",
                    "target": "M.Tech",
                    "requested_attributes": ["application fee"],
                },
                {
                    "question": "eligibility",
                    "request_type": "eligibility",
                    "target": "M.Tech",
                    "requested_attributes": ["eligibility"],
                },
            ],
            [("fee", "M.Tech", ["application fee"])],
        ),
        (
            "contrastive_process",
            (
                "When you say 'apply', I mean the actual admission process steps. "
                "What are the steps rather than the eligibility rules?"
            ),
            [
                {"question": "eligibility", "request_type": "eligibility",
                 "requested_attributes": ["eligibility"]},
                {"question": "process", "request_type": "process",
                 "requested_attributes": ["process"]},
                {"question": "policy", "request_type": "policy",
                 "requested_attributes": ["policy"]},
            ],
            [("process", None, ["process"])],
        ),
        (
            "already_known_not_requested",
            (
                "I already know the eligibility requirements, just tell me the "
                "steps for applying."
            ),
            [
                {"question": "eligibility", "request_type": "eligibility",
                 "requested_attributes": ["eligibility"]},
                {"question": "process", "request_type": "process",
                 "requested_attributes": ["process"]},
            ],
            [("process", None, ["process"])],
        ),
        (
            "predicate_negation_is_still_eligibility",
            "Why am I not eligible for M.Tech?",
            [
                {"question": "eligibility", "request_type": "eligibility",
                 "target": "M.Tech", "requested_attributes": ["eligibility"]},
            ],
            [("eligibility", "M.Tech", ["eligibility"])],
        ),
        (
            "topic_correction_target",
            (
                "I want the M.Tech deadline, but actually I mean the deadline for "
                "MBA, not M.Tech."
            ),
            [
                {"question": "deadline", "request_type": "deadline",
                 "target": "M.Tech", "requested_attributes": ["deadline"]},
            ],
            [("deadline", "MBA", ["deadline"])],
        ),
        (
            "fee_plus_deadline",
            "For M.Tech, give me the application fee and the deadline.",
            [
                {"question": "everything", "request_type": "general",
                 "target": "M.Tech", "requested_attributes": ["application fee", "deadline"]},
            ],
            [
                ("fee", "M.Tech", ["application fee"]),
                ("deadline", "M.Tech", ["deadline"]),
            ],
        ),
        (
            "documents_are_not_eligibility",
            "What documents are required for MBA admission?",
            [
                {"question": "documents", "request_type": "eligibility",
                 "target": "MBA", "requested_attributes": ["eligibility"]},
            ],
            [("documents", "MBA", ["documents"])],
        ),
        (
            "work_experience",
            "Is work experience compulsory for M.Tech?",
            [
                {"question": "generic", "request_type": "process",
                 "target": "M.Tech", "requested_attributes": ["process"]},
            ],
            [("eligibility", "M.Tech", ["work experience"])],
        ),
        (
            "genuine_three_way_multi_intent",
            (
                "For M.Tech I need the eligibility requirements, required documents, "
                "and application deadline."
            ),
            [
                {"question": "generic", "request_type": "general",
                 "target": "M.Tech", "requested_attributes": []},
            ],
            [
                ("eligibility", "M.Tech", ["eligibility"]),
                ("documents", "M.Tech", ["documents"]),
                ("deadline", "M.Tech", ["deadline"]),
            ],
        ),
        (
            "not_hostel_fee",
            (
                "Don't tell me the hostel fee. I only want the M.Tech application fee."
            ),
            [
                {"question": "hostel", "request_type": "fee", "target": "M.Tech",
                 "requested_attributes": ["hostel fee"]},
                {"question": "application", "request_type": "fee", "target": "M.Tech",
                 "requested_attributes": ["application fee"]},
            ],
            [("fee", "M.Tech", ["application fee"])],
        ),
        (
            "hindi_eligibility",
            "M.Tech में दाखिले की पात्रता क्या है?",
            [
                {"question": "generic", "request_type": "process",
                 "target": "M.Tech", "requested_attributes": ["process"]},
            ],
            [("eligibility", "M.Tech", ["eligibility"])],
        ),
    ]

    print("=" * 88)
    print("QUERY PLANNER V2.4 SEMANTIC REGRESSION GATE")
    print("=" * 88)

    passed = 0
    for index, (name, question, llm_intents, expected) in enumerate(cases, 1):
        try:
            plan = run_case(name, question, llm_intents, expected)
            print(
                f"[{index:02d}/{len(cases):02d}] PASS "
                f"{name:<36} intents={len(plan.intents)}"
            )
            passed += 1
        except Exception as exc:
            print(f"[{index:02d}/{len(cases):02d}] FAIL {name}")
            print(f"       {type(exc).__name__}: {exc}")
            return 1

    print()
    print(f"REGRESSION GATE: PASS ({passed}/{len(cases)})")
    print("Production behavior tested: semantic reconciliation only.")
    print("Live LLM: NOT USED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
