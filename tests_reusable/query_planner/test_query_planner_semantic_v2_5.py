#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ai_platform.core.query import planner


@dataclass
class FakeModel:
    payload: dict[str, Any]
    calls: int = 0

    def invoke(self, _prompt: Any) -> str:
        self.calls += 1
        return json.dumps(self.payload)


def run_case(name: str, question: str, payload: dict[str, Any], expected: list[tuple[str | None, str | None, tuple[str, ...]]]) -> None:
    fake = FakeModel(payload)
    plan = planner.build_query_plan(question=question, model=fake)
    observed = [
        (i.request_type, i.target, tuple(i.requested_attributes))
        for i in plan.intents
    ]
    if observed != expected:
        raise AssertionError(
            f"{name} failed\nEXPECTED: {expected}\nOBSERVED: {observed}\n"
        )
    assert fake.calls == 1, (name, fake.calls)


CASES = [
    (
        "natural_target_and_eligibility",
        "I have a B.Tech degree with 7.2 CGPA and a valid GATE score. Does that make me eligible for the M.Tech admission route, and what additional academic condition could still disqualify me?",
        {"intents": [{"question": "eligibility", "request_type": "eligibility", "target": None, "requested_attributes": ["eligibility"]}]},
        [("eligibility", "M.Tech", ("eligibility",))],
    ),
    (
        "specific_facet_not_duplicate_generic",
        "Is work experience required for M.Tech?",
        {"intents": [
            {"question": "Is work experience required for M.Tech?", "request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["work experience", "eligibility"]},
            {"question": "What are eligibility requirements?", "request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["eligibility"]},
        ]},
        [("eligibility", "M.Tech", ("work experience",))],
    ),
    (
        "disclaimed_topic_not_promoted",
        "I am not asking whether I am eligible for M.Tech. I only need the application fee and whether there is any separate registration charge.",
        {"intents": [
            {"question": "fee", "request_type": "fee", "target": "M.Tech", "requested_attributes": ["application fee"]},
            {"question": "eligibility", "request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["eligibility"]},
        ]},
        [("fee", "M.Tech", ("application fee",))],
    ),
    (
        "contrastive_process_is_one_objective",
        "When you say apply, I mean the actual admission process steps. What are the steps rather than the eligibility rules?",
        {"intents": [
            {"question": "eligibility", "request_type": "eligibility", "target": None, "requested_attributes": ["eligibility"]},
            {"question": "process", "request_type": "process", "target": None, "requested_attributes": ["process"]},
            {"question": "policy", "request_type": "policy", "target": None, "requested_attributes": ["policy"]},
        ]},
        [("process", None, ("process",))],
    ),
    (
        "final_semester_is_constraint_not_attribute",
        "I am in my final semester and my degree will be completed only after the admission process. Can I still be considered for M.Tech, or is completed graduation mandatory?",
        {"intents": [{"question": "eligibility", "request_type": "process", "target": "M.Tech", "requested_attributes": ["process"]}]},
        [("eligibility", "M.Tech", ("eligibility",))],
    ),
    (
        "correction_prefers_actual_target",
        "Forget M.Tech; I meant PhD documents.",
        {"intents": [{"question": "documents", "request_type": "documents", "target": "M.Tech", "requested_attributes": ["documents"]}]},
        [("documents", "PhD", ("documents",))],
    ),
]

print("=" * 88)
print("QUERY PLANNER V2.5 SEMANTIC REGRESSION GATE")
print("=" * 88)

for idx, case in enumerate(CASES, 1):
    run_case(*case)
    print(f"[{idx:02d}/{len(CASES):02d}] PASS {case[0]}")

# Additional direct contract checks against helper behavior.
assert planner._requested_attributes("Is work experience required for M.Tech?") == ["work experience"]
assert "eligibility" not in planner._requested_attributes(
    "I am not asking whether I am eligible for M.Tech. I only need the application fee."
)
assert planner._infer_target(
    "I have a B.Tech degree with 7.2 CGPA and a valid GATE score. Does that make me eligible for the M.Tech admission route?"
) == "M.Tech"

print(f"\nREGRESSION GATE: PASS ({len(CASES) + 3}/{len(CASES) + 3})")
