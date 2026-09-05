"""
Phase 8A.2 — Retrieval Scope Diagnosis

Purpose
-------
Inspect real retrieval/answer behavior for scope-sensitive institutional
questions before modifying the retrieval framework.

This is a diagnostic runner, not yet a strict pytest gate.

Focus:
    - department scope
    - school scope
    - research-topic scope
    - false negatives
    - cross-organization contamination
"""

from __future__ import annotations

import time

from backend.graph import create_graph


# =========================================================
# Scope-sensitive cases
# =========================================================

CASES = [
    {
        "id": "control_01",
        "question": (
            "What research areas are related to control systems "
            "in Electrical Engineering?"
        ),
        "expected_signals": [
            "control",
            "electrical",
        ],
        "forbidden_signals": [
            "smart city management",
            "intelligent transportation system",
        ],
    },
    {
        "id": "robotics_01",
        "question": (
            "What research areas are related to robotics "
            "at IIT Jodhpur?"
        ),
        "expected_signals": [
            "robot",
            "control",
            "electrical",
            "embedded",
            "cyber-physical",
        ],
        "forbidden_signals": [],
    },
    {
        "id": "ee_research_01",
        "question": (
            "What research areas are available in "
            "Electrical Engineering?"
        ),
        "expected_signals": [
            "electrical",
            "control",
            "vlsi",
            "power",
            "signal",
            "communication",
        ],
        "forbidden_signals": [],
    },
    {
        "id": "aids_programs_01",
        "question": (
            "What programs are available in the School of "
            "Artificial Intelligence and Data Science?"
        ),
        "expected_signals": [
            "artificial intelligence",
            "data",
        ],
        "forbidden_signals": [
            "hostel",
            "electrical engineering",
            "control systems",
        ],
    },
    {
        "id": "aids_research_01",
        "question": (
            "What research is being done in the School of "
            "Artificial Intelligence and Data Science?"
        ),
        "expected_signals": [
            "artificial intelligence",
            "data",
            "research",
        ],
        "forbidden_signals": [
            "electrical engineering",
            "power systems",
            "hostel",
        ],
    },
    {
        "id": "generic_research_01",
        "question": (
            "What research opportunities are available "
            "at IIT Jodhpur?"
        ),
        "expected_signals": [
            "research",
        ],
        "forbidden_signals": [],
    },
]


# =========================================================
# Helpers
# =========================================================

def _text(value) -> str:
    if value is None:
        return ""

    return str(value)


def _contains_any(
    text: str,
    signals: list[str],
) -> list[str]:

    normalized = text.lower()

    return [
        signal
        for signal in signals
        if signal.lower() in normalized
    ]


def _print_separator():
    print("=" * 100)


# =========================================================
# Main
# =========================================================

def main():

    _print_separator()

    print(
        "IITJ V1 — PHASE 8A.2 SCOPE DIAGNOSIS"
    )

    print(
        f"Cases: {len(CASES)}"
    )

    _print_separator()

    print(
        "Creating graph..."
    )

    graph = create_graph()

    print(
        "Graph ready."
    )

    _print_separator()

    for index, case in enumerate(
        CASES,
        start=1,
    ):

        case_id = case["id"]
        question = case["question"]

        print(
            f"[{index}/{len(CASES)}] {case_id}"
        )

        print(
            f"QUESTION: {question}"
        )

        started = time.perf_counter()

        try:

            result = graph.invoke(
                {
                    "question": question,
                    "chat_history": [],
                }
            )

            latency = (
                time.perf_counter()
                - started
            )

            answer = _text(
                result.get(
                    "answer",
                    "",
                )
            )

            context = _text(
                result.get(
                    "context",
                    "",
                )
            )

            resolved = _text(
                result.get(
                    "resolved_question",
                    "",
                )
            )

            mode = _text(
                result.get(
                    "conversation_mode",
                    "",
                )
            )

            combined = (
                answer
                + "\n"
                + context
            ).lower()

            expected_hits = _contains_any(
                combined,
                case["expected_signals"],
            )

            forbidden_hits = _contains_any(
                combined,
                case["forbidden_signals"],
            )

            print(
                f"LATENCY: {latency:.2f}s"
            )

            print(
                f"MODE: {mode}"
            )

            print(
                f"RESOLVED: {resolved}"
            )

            print(
                "\nEXPECTED SIGNALS FOUND:"
            )

            print(
                expected_hits
                if expected_hits
                else "NONE"
            )

            print(
                "\nFORBIDDEN SIGNALS FOUND:"
            )

            print(
                forbidden_hits
                if forbidden_hits
                else "NONE"
            )

            print(
                "\nANSWER:"
            )

            print(
                answer
            )

            print(
                "\nCONTEXT:"
            )

            print(
                context
            )

            print(
                f"\nCONTEXT CHARACTERS: "
                f"{len(context)}"
            )

        except Exception as exc:

            latency = (
                time.perf_counter()
                - started
            )

            print(
                f"ERROR after {latency:.2f}s"
            )

            print(
                repr(exc)
            )

        _print_separator()


if __name__ == "__main__":
    main()