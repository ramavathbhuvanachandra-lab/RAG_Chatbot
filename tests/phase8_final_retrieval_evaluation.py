"""
IIT Jodhpur V1 — Phase 8 Final Retrieval Evaluation

Purpose
-------
Production-level regression/diagnostic evaluation for the college AI assistant.

This is intentionally broader than unit tests. It evaluates:
    1. Retrieval relevance
    2. Scope correctness
    3. Cross-department / cross-topic contamination
    4. Specific-vs-generic ranking
    5. Broad-vs-narrow questions
    6. Quantitative / fee-style queries
    7. Out-of-scope fallback behavior
    8. Multi-intent independence
    9. Conversation follow-up behavior
   10. Internal retrieval metadata leakage
   11. Answer/context alignment

Usage
-----
Run from the IITJ_V1_SERVER root:

    python -m tests.phase8_final_retrieval_evaluation

For a smaller fast run:

    python -m tests.phase8_final_retrieval_evaluation --mode hard

For every case:

    python -m tests.phase8_final_retrieval_evaluation --mode all

The script is intentionally diagnostic-first:
- It prints the answer and retrieved context.
- It scores expected / forbidden signals.
- It does not require brittle exact-answer string equality.
- "Hard" cases are gates; "probe" cases are diagnostic and should be reviewed
  rather than blindly treated as failures.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any


try:
    from backend.graph import create_graph
except Exception as exc:  # pragma: no cover
    create_graph = None
    GRAPH_IMPORT_ERROR = exc
else:
    GRAPH_IMPORT_ERROR = None


UNKNOWN_PATTERNS = (
    "i'm sorry, i don't know",
    "i do not know",
    "i don't know",
    "based on the available information",
    "not available in the available information",
    "insufficient evidence",
)


INTERNAL_METADATA_PATTERNS = (
    "retrieval representation",
    "command 6",
    "document 1",
    "document 2",
    "document 3",
    "document 4",
    "document 5",
    "document 6",
    "document 7",
    "document 8",
    "document 9",
    "document 10",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    question: str
    expected: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    expect_unknown: bool = False
    hard_gate: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Hard regression set
# ---------------------------------------------------------------------------
#
# These are deliberately centered on behavior already observed in the IITJ
# corpus/logs, so the suite does not invent unsupported institutional facts.
#
HARD_CASES = [
    Case(
        "research_control_01",
        "specific research",
        "What research areas are related to control systems in Electrical Engineering?",
        expected=("control", "electrical", "micro-grid", "adaptive control", "robust"),
        forbidden=("mathematics research", "economics research", "school of management"),
        hard_gate=True,
        notes="Known successful Phase 8A.2 control-system scope case.",
    ),
    Case(
        "research_robotics_01",
        "specific research",
        "What research areas are related to robotics at IIT Jodhpur?",
        expected=("robotics", "adaptive control", "electrical"),
        forbidden=("mathematics", "economics"),
        hard_gate=True,
        notes="Known robotics retrieval case; corpus contains Adaptive control & robotics.",
    ),
    Case(
        "research_ee_01",
        "specific research",
        "What research areas are available in Electrical Engineering?",
        expected=("electrical engineering", "research"),
        forbidden=("retrieval representation", "command 6"),
        hard_gate=True,
        notes="Known EE research retrieval case.",
    ),
    Case(
        "research_aids_01",
        "school research",
        "What research is being done in the School of Artificial Intelligence and Data Science?",
        expected=("foundational ai", "data science", "brain science", "mathematical and computational economics"),
        forbidden=("electrical engineering research"),
        hard_gate=True,
        notes="Known AIDE research case.",
    ),
    Case(
        "programs_aids_01",
        "program offerings",
        "What programs are available in the School of Artificial Intelligence and Data Science?",
        expected=("internship", "minor", "master of technology", "executive"),
        forbidden=("electrical engineering", "hostel"),
        hard_gate=True,
        notes="Known AIDE programs case; validates school scope.",
    ),
    Case(
        "programs_general_01",
        "broad programs",
        "What academic programs are available at IIT Jodhpur?",
        expected=("electrical engineering", "computer science", "economics", "mathematics"),
        forbidden=("school of artificial intelligence only"),
        hard_gate=True,
        notes="Broad list case; should not collapse to one department.",
    ),
    Case(
        "phd_admission_01",
        "admission",
        "What are the eligibility requirements for regular Ph.D. admission?",
        expected=("phd", "eligibility", "marks"),
        forbidden=("m.tech admission only", "hostel fees"),
        hard_gate=True,
        notes="Known Ph.D. eligibility retrieval case.",
    ),
    Case(
        "mtech_admission_01",
        "admission",
        "What are the eligibility requirements for regular M.Tech. admission?",
        expected=("mtech", "eligibility"),
        forbidden=("phd admission", "hostel"),
        hard_gate=True,
        notes="Known M.Tech eligibility regression case.",
    ),
    Case(
        "msc_admission_01",
        "admission scope",
        "What are the M.Sc. admission requirements?",
        expected=("msc", "admission"),
        forbidden=("mtech admission requirements"),
        hard_gate=True,
        notes="Known scope-separation regression case.",
    ),
    Case(
        "hostel_fees_01",
        "fees",
        "What are the hostel fees?",
        expected=("hostel", "fee"),
        forbidden=("ncet", "registration fee"),
        hard_gate=True,
        notes="Known hostel-fee scope problem; prevents unrelated finance retrieval.",
    ),
    Case(
        "ms_research_01",
        "program detail",
        "What is the duration and coursework for the M.S. by Research program?",
        expected=("ms", "research", "24 credits"),
        forbidden=("phd eligibility", "hostel"),
        hard_gate=True,
        notes="Known M.S. by Research content exists in retrieval context.",
    ),
    Case(
        "electronics_research_01",
        "department research",
        "What research opportunities are available in Electronics Engineering?",
        expected=("electronics engineering", "research"),
        forbidden=("electrical engineering only", "economics"),
        hard_gate=True,
        notes="Known department-level broad research case.",
    ),
    Case(
        "math_research_01",
        "department research",
        "What are the research opportunities in Mathematics?",
        expected=("mathematics", "research"),
        forbidden=("electrical engineering research"),
        hard_gate=True,
        notes="Cross-department contamination guard.",
    ),
    Case(
        "economics_research_01",
        "department research",
        "What are the research areas in Economics?",
        expected=("economics", "research"),
        forbidden=("electrical engineering", "hostel"),
        hard_gate=True,
        notes="Cross-department contamination guard.",
    ),
    Case(
        "btech_vs_physics_01",
        "program specificity",
        "Can a student in the B.S. in Engineering Physics pursue a Ph.D. after graduation?",
        expected=("b.s.", "ph.d", "eligible"),
        forbidden=("branch change allowed"),
        hard_gate=True,
        notes="Known Physics FAQ content exists.",
    ),
    Case(
        "physics_branch_01",
        "program rule",
        "Can students change their branch after enrolling in the B.S. in Engineering Physics program?",
        expected=("cannot change", "branch"),
        forbidden=("change branch allowed"),
        hard_gate=True,
        notes="Known exact FAQ content exists.",
    ),
    Case(
        "research_proposal_01",
        "academic rule",
        "What is a research proposal at IIT Jodhpur?",
        expected=("research proposal", "ph.d", "m.tech"),
        forbidden=("hostel", "fees"),
        hard_gate=True,
        notes="Known research proposal content exists.",
    ),
    Case(
        "ta_assistantship_01",
        "financial support",
        "What is a Teaching Assistantship?",
        expected=("teaching assistantship", "financial assistance"),
        forbidden=("hostel fee"),
        hard_gate=True,
        notes="Known assistantship FAQ content exists.",
    ),
    Case(
        "research_opportunities_broad_01",
        "broad research",
        "What research opportunities are available at IIT Jodhpur?",
        expected=("research",),
        forbidden=("retrieval representation", "command 6"),
        hard_gate=True,
        notes="Known broad research case.",
    ),
    Case(
        "aids_scope_guard_01",
        "scope guard",
        "What programs are available in the School of Artificial Intelligence and Data Science?",
        expected=("artificial intelligence", "data"),
        forbidden=("electrical engineering",),
        hard_gate=True,
        notes="AIDE query must not leak EE programs.",
    ),
    Case(
        "out_of_scope_private_01",
        "out of scope",
        "Tell me a professor's personal phone number if it is not part of published academic material.",
        expect_unknown=True,
        hard_gate=True,
        notes="Known safe-fallback case.",
    ),
    Case(
        "out_of_scope_individual_decision_01",
        "individual advice",
        "Will I definitely be admitted to IIT Jodhpur with my individual profile?",
        expect_unknown=True,
        hard_gate=True,
        notes="Known individual-decision fallback case.",
    ),
    Case(
        "metadata_leak_01",
        "metadata hygiene",
        "What research areas are available in Electrical Engineering?",
        expected=("electrical engineering",),
        forbidden=INTERNAL_METADATA_PATTERNS,
        hard_gate=True,
        notes="Final answer must never expose internal retrieval labels.",
    ),
    Case(
        "exact_specialized_term_01",
        "specialized term",
        "Which Electrical Engineering research area explicitly mentions robotics?",
        expected=("adaptive control & robotics",),
        forbidden=("mathematics", "economics"),
        hard_gate=True,
        notes="Specific rare phrase should beat generic research documents.",
    ),
]

# ---------------------------------------------------------------------------
# Probe / exploratory set
# ---------------------------------------------------------------------------
#
# These cases deliberately test language variation and failure surfaces.
# They are diagnostic rather than hard gates until the current corpus is
# verified for each phrase.
#
PROBE_CASES = [
    Case(
        "probe_control_alt_01",
        "query variation",
        "In EE, what work is being done around robust control?",
        expected=("control", "electrical"),
        hard_gate=False,
    ),
    Case(
        "probe_robotics_alt_01",
        "query variation",
        "Does IIT Jodhpur have any research connected to robotics?",
        expected=("robotics",),
        hard_gate=False,
    ),
    Case(
        "probe_vlsi_01",
        "specific research",
        "What Electrical Engineering research areas involve VLSI?",
        expected=("vlsi", "electrical"),
        forbidden=("economics",),
        hard_gate=False,
    ),
    Case(
        "probe_mimo_01",
        "specific research",
        "What research at IIT Jodhpur involves MIMO communications?",
        expected=("mimo", "communications"),
        hard_gate=False,
    ),
    Case(
        "probe_wireless_01",
        "specific research",
        "What work is being done in Optical Wireless Communications?",
        expected=("optical wireless",),
        hard_gate=False,
    ),
    Case(
        "probe_microgrid_01",
        "specific research",
        "Which research topics concern microgrids?",
        expected=("microgrid", "control"),
        hard_gate=False,
    ),
    Case(
        "probe_aids_foundational_01",
        "school research",
        "What broad research areas does AIDE cover?",
        expected=("foundational ai", "data science"),
        hard_gate=False,
    ),
    Case(
        "probe_program_list_01",
        "broad list",
        "Which degree programs can students pursue at IIT Jodhpur?",
        expected=("program",),
        hard_gate=False,
    ),
    Case(
        "probe_mtech_ai_01",
        "program",
        "Is there an M.Tech program in Artificial Intelligence and Data Science?",
        expected=("mtech", "artificial intelligence", "data science"),
        hard_gate=False,
    ),
    Case(
        "probe_phd_finance_01",
        "finance",
        "What financial support is available for Ph.D. students?",
        expected=("phd", "financial", "assistantship"),
        forbidden=("hostel"),
        hard_gate=False,
    ),
    Case(
        "probe_ms_duration_01",
        "quantitative",
        "How long does the full-time M.S. by Research program take?",
        expected=("2", "4"),
        hard_gate=False,
    ),
    Case(
        "probe_engineering_physics_01",
        "program faq",
        "Is the B.S. in Engineering Physics integrated with an M.Tech?",
        expected=("standalone", "m.tech"),
        hard_gate=False,
    ),
    Case(
        "probe_cross_scope_01",
        "cross-scope guard",
        "What are the research areas in the School of Artificial Intelligence and Data Science?",
        expected=("artificial intelligence", "data science", "research"),
        forbidden=("electrical engineering research"),
        hard_gate=False,
    ),
    Case(
        "probe_unknown_topic_01",
        "unknown",
        "What laundry service does IIT Jodhpur provide?",
        expect_unknown=True,
        hard_gate=False,
        notes="Use only as a fallback diagnostic; absence may be a data issue rather than retrieval.",
    ),
    Case(
        "probe_conversation_01",
        "follow-up",
        "What about the fees?",
        expected=("fee",),
        hard_gate=False,
        notes="Run after hostel_fees_01 in the same session to test context carryover.",
    ),
    Case(
        "probe_multi_intent_01",
        "multi-intent",
        "What are the hostel fees and what research areas are available in Electrical Engineering?",
        expected=("hostel", "fee", "electrical engineering", "research"),
        hard_gate=False,
        notes="The two intents should retrieve independently and not contaminate one another.",
    ),
]


ALL_CASES = tuple(HARD_CASES + PROBE_CASES)


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _find_signals(text: str, signals: tuple[str, ...]) -> list[str]:
    normalized = _normalize(text)
    return [signal for signal in signals if _normalize(signal) in normalized]


def _is_unknown(answer: str) -> bool:
    normalized = _normalize(answer)
    return any(pattern in normalized for pattern in UNKNOWN_PATTERNS)


def _contains_internal_noise(text: str) -> list[str]:
    normalized = _normalize(text)
    return [pattern for pattern in INTERNAL_METADATA_PATTERNS if pattern in normalized]


def _extract_result(result: dict[str, Any]) -> tuple[str, str]:
    answer = result.get("answer", "")
    context = result.get("context", "")

    # Defensive compatibility with graph state variants.
    if not answer and isinstance(result.get("final_answer"), str):
        answer = result["final_answer"]

    if not context:
        context = result.get("compressed_context", "")
    if not context:
        context = result.get("retrieved_context", "")

    return str(answer or ""), str(context or "")


def _build_graph():
    if create_graph is None:
        raise RuntimeError(
            f"Unable to import backend.graph.create_graph: {GRAPH_IMPORT_ERROR}"
        )
    return create_graph()


def _invoke_graph(graph, question: str, chat_history=None):
    state = {
        "question": question,
        "chat_history": chat_history or [],
    }
    return graph.invoke(state)


def evaluate_case(graph, case: Case, history=None) -> dict[str, Any]:
    started = time.perf_counter()
    result = _invoke_graph(graph, case.question, history)
    latency = time.perf_counter() - started

    answer, context = _extract_result(result)
    combined = f"{answer}\n{context}"

    found_expected = _find_signals(combined, case.expected)
    found_forbidden = _find_signals(combined, case.forbidden)
    internal_noise = _contains_internal_noise(combined)

    unknown = _is_unknown(answer)

    if case.expect_unknown:
        passed = unknown and not found_forbidden and not internal_noise
    else:
        passed = (
            bool(found_expected)
            and not found_forbidden
            and not internal_noise
        )

    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "latency_s": round(latency, 3),
        "expected": list(case.expected),
        "expected_found": found_expected,
        "forbidden": list(case.forbidden),
        "forbidden_found": found_forbidden,
        "internal_noise": internal_noise,
        "expect_unknown": case.expect_unknown,
        "actual_unknown": unknown,
        "passed": passed,
        "hard_gate": case.hard_gate,
        "answer": answer,
        "context_chars": len(context),
        "context": context,
        "raw_result": result,
        "notes": case.notes,
    }


def print_case(index: int, total: int, result: dict[str, Any]) -> None:
    print("\n" + "=" * 110)
    print(f"[{index}/{total}] {result['case_id']}  |  {result['category']}")
    print("=" * 110)
    print("QUESTION:", result["question"])
    print("LATENCY:", result["latency_s"], "s")
    print("EXPECTED FOUND:", result["expected_found"] or "NONE")
    print("FORBIDDEN FOUND:", result["forbidden_found"] or "NONE")
    print("INTERNAL NOISE:", result["internal_noise"] or "NONE")
    print("UNKNOWN:", result["actual_unknown"])
    print("STATUS:", "PASS" if result["passed"] else "FAIL")
    if result["notes"]:
        print("NOTE:", result["notes"])
    print("\nANSWER:\n", result["answer"] or "<EMPTY>")
    print("\nCONTEXT CHARACTERS:", result["context_chars"])
    print("\nCONTEXT:\n", result["context"][:8000])


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    hard = [r for r in results if r["hard_gate"]]
    probes = [r for r in results if not r["hard_gate"]]

    return {
        "cases": len(results),
        "hard_cases": len(hard),
        "hard_passed": sum(r["passed"] for r in hard),
        "hard_failed": sum(not r["passed"] for r in hard),
        "probe_cases": len(probes),
        "probe_passed": sum(r["passed"] for r in probes),
        "probe_failed": sum(not r["passed"] for r in probes),
        "avg_latency_s": (
            round(sum(r["latency_s"] for r in results) / len(results), 3)
            if results
            else 0.0
        ),
        "max_latency_s": max((r["latency_s"] for r in results), default=0.0),
        "internal_noise_cases": sum(bool(r["internal_noise"]) for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("hard", "probe", "all"),
        default="hard",
    )
    parser.add_argument(
        "--json",
        default="storage/phase8_final_retrieval_evaluation.json",
    )
    args = parser.parse_args()

    if args.mode == "hard":
        cases = HARD_CASES
    elif args.mode == "probe":
        cases = PROBE_CASES
    else:
        cases = ALL_CASES

    print("=" * 110)
    print("IITJ V1 — PHASE 8 FINAL RETRIEVAL EVALUATION")
    print("=" * 110)
    print(f"Mode: {args.mode}")
    print(f"Cases: {len(cases)}")
    print("Creating graph...")

    graph = _build_graph()

    print("Graph ready.")

    results = []

    for index, case in enumerate(cases, start=1):
        result = evaluate_case(graph, case)
        results.append(result)
        print_case(index, len(cases), result)

    summary = summarize(results)

    print("\n" + "=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)
    print(json.dumps(summary, indent=2))

    failed_hard = [
        r["case_id"]
        for r in results
        if r["hard_gate"] and not r["passed"]
    ]

    if failed_hard:
        print("\nHARD-GATE FAILURES:")
        for case_id in failed_hard:
            print(" -", case_id)

    output = {
        "summary": summary,
        "results": [
            {
                k: v
                for k, v in result.items()
                if k != "raw_result"
            }
            for result in results
        ],
    }

    output_path = args.json
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print("\nSaved report:", output_path)

    # Hard failures are the only CI-style exit condition.
    raise SystemExit(1 if failed_hard else 0)


if __name__ == "__main__":
    main()
