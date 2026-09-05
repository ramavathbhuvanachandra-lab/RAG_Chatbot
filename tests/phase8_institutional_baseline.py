"""
Phase 8A — Institutional Baseline Regression

Purpose
-------
Establish a real baseline for institutional question-answering before
changing retrieval, evidence, or knowledge logic.

This is intentionally a baseline runner, not a pass/fail unit test.

The result of this run will be used to identify:
    - retrieval failures
    - scope/entity failures
    - evidence gaps
    - knowledge/data conflicts
    - answer-quality problems
    - unknown/unsupported handling problems
"""

from __future__ import annotations

import time

from backend.graph import create_graph


# =========================================================
# Baseline questions
# =========================================================
#
# These questions deliberately cover different institutional
# domains and different difficulty levels.
#
# They are NOT intended to represent every possible question.
# They establish the first Phase-8 measurement point.
# =========================================================

QUESTIONS = [
    # -----------------------------------------------------
    # General institute
    # -----------------------------------------------------

    (
        "general_01",
        "What is IIT Jodhpur?",
    ),

    (
        "general_02",
        "What are the major academic areas available at IIT Jodhpur?",
    ),

    (
        "general_03",
        "What schools and departments are mentioned in the institute information?",
    ),

    # -----------------------------------------------------
    # Ph.D. / admissions
    # -----------------------------------------------------

    (
        "phd_01",
        "What are the regular Ph.D. eligibility requirements?",
    ),

    (
        "phd_02",
        "What are the eligibility requirements for the four-year bachelor's route to Ph.D. admission?",
    ),

    (
        "phd_03",
        "What percentage is required for regular Ph.D. admission?",
    ),

    (
        "phd_04",
        "What is the GATE requirement for Ph.D. admission?",
    ),

    (
        "phd_05",
        "What financial assistance is available to Ph.D. students?",
    ),

    # -----------------------------------------------------
    # M.Tech / M.Sc.
    # -----------------------------------------------------

    (
        "pg_01",
        "What M.Tech. programs are available?",
    ),

    (
        "pg_02",
        "What M.Sc. programs are available?",
    ),

    (
        "pg_03",
        "What entrance exam is required for M.Sc. admission?",
    ),

    # -----------------------------------------------------
    # Departments / schools
    # -----------------------------------------------------

    (
        "org_01",
        "What research areas are available in Electrical Engineering?",
    ),

    (
        "org_02",
        "What research areas are related to control systems?",
    ),

    (
        "org_03",
        "What programs are available in the School of Artificial Intelligence and Data Science?",
    ),

    (
        "org_04",
        "What research is being done in the School of Artificial Intelligence and Data Science?",
    ),

    # -----------------------------------------------------
    # Research
    # -----------------------------------------------------

    (
        "research_01",
        "What research opportunities are available at IIT Jodhpur?",
    ),

    (
        "research_02",
        "Which research areas combine hardware and software?",
    ),

    (
        "research_03",
        "What research areas are related to robotics?",
    ),

    # -----------------------------------------------------
    # Hostel
    # -----------------------------------------------------

    (
        "hostel_01",
        "What hostel accommodation is available?",
    ),

    (
        "hostel_02",
        "What are the short-term hostel rates?",
    ),

    (
        "hostel_03",
        "What are the single-occupancy hostel rates?",
    ),

    (
        "hostel_04",
        "What happens for a hostel stay longer than ten days?",
    ),

    (
        "hostel_05",
        "Do hostel rates differ between students and visitors?",
    ),

    # -----------------------------------------------------
    # Fees / finance
    # -----------------------------------------------------

    (
        "finance_01",
        "What are the hostel accommodation charges?",
    ),

    (
        "finance_02",
        "Does the semester fee include hostel fees?",
    ),

    (
        "finance_03",
        "What financial assistance is available for Ph.D. students?",
    ),

    # -----------------------------------------------------
    # Academic rules / registration
    # -----------------------------------------------------

    (
        "academic_01",
        "Can a Ph.D. student be enrolled in another academic program at the same time?",
    ),

    (
        "academic_02",
        "What are the attendance requirements for Ph.D. students?",
    ),

    # -----------------------------------------------------
    # Unsupported / adversarial
    # -----------------------------------------------------

    (
        "unknown_01",
        "What is the exact salary of every IIT Jodhpur professor?",
    ),

    (
        "unknown_02",
        "Who will be the most popular professor at IIT Jodhpur in 2028?",
    ),

    (
        "unknown_03",
        "Can you guarantee that I will get a hostel room after admission?",
    ),
]


# =========================================================
# Formatting helpers
# =========================================================

def _print_separator():
    print("=" * 100)


def _safe_text(value):
    if value is None:
        return ""

    return str(value)


# =========================================================
# Main baseline runner
# =========================================================

def main():
    _print_separator()

    print(
        "IITJ V1 — PHASE 8A INSTITUTIONAL BASELINE"
    )

    print(
        f"Questions: {len(QUESTIONS)}"
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

    for index, (case_id, question) in enumerate(
        QUESTIONS,
        start=1,
    ):

        print(
            f"[{index}/{len(QUESTIONS)}] "
            f"{case_id}"
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

            answer = result.get(
                "answer",
                "",
            )

            context = result.get(
                "context",
                "",
            )

            resolved_question = result.get(
                "resolved_question",
                "",
            )

            conversation_mode = result.get(
                "conversation_mode",
                "",
            )

            print(
                f"LATENCY: {latency:.2f}s"
            )

            print(
                f"MODE: {conversation_mode}"
            )

            print(
                f"RESOLVED: "
                f"{_safe_text(resolved_question)}"
            )

            print(
                "ANSWER:"
            )

            print(
                _safe_text(answer)
            )

            print(
                f"CONTEXT CHARACTERS: "
                f"{len(_safe_text(context))}"
            )

        except Exception as exc:

            latency = (
                time.perf_counter()
                - started
            )

            print(
                f"ERROR after {latency:.2f}s: "
                f"{exc}"
            )

        print()

    _print_separator()

    print(
        "BASELINE COMPLETE"
    )

    print(
        "Do not treat this run as a pass/fail gate."
    )

    print(
        "Use the output to identify the first major Phase-8 failure patterns."
    )

    _print_separator()


if __name__ == "__main__":
    main()