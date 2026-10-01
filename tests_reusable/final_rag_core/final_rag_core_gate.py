"""Final end-to-end RAG quality gate.

This gate is deliberately simpler than the component-level regression suites.
It answers one product question:

    If the knowledge base contains the answer, did the real RAG pipeline
    retrieve usable evidence, send it to the answer model, and return a
    grounded final answer?

The existing Node 2 IITJ gate remains responsible for pipeline integrity and
safety checks. This gate sits above it and measures end-to-end usefulness.

Run from the repository root:

    PYTHONPATH=. python tests_reusable/final_rag_core/final_rag_core_gate.py

Optional:

    PYTHONPATH=. python tests_reusable/final_rag_core/final_rag_core_gate.py --limit 10
    PYTHONPATH=. python tests_reusable/final_rag_core/final_rag_core_gate.py --threshold 0.90

The initial 20-question set is a seed set. The intended production gate is a
100+ question golden set, where each case also records its expected supporting
source(s). Until source labels are added, retrieval success is measured as
"usable verified evidence reached the evidence layer" rather than exact source
recall.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NODE2_GATE = ROOT / "tests_reusable" / "phase3_real_rag" / "iitj_node2_real_e2e_gate.py"
DEFAULT_REPORT = (
    ROOT
    / "tests_reusable"
    / "phase3_real_rag"
    / "reports"
    / "node2_real_e2e_report.json"
)


# These are the current Node 2 seed questions. They are intentionally treated
# as answerable for this product gate: they are questions selected from the
# institution corpus to validate the core retrieval -> evidence -> answer path.
SEED_QUESTIONS = (
    "What are the M.Tech eligibility requirements?",
    "What are the admission routes for M.Sc.?",
    "What is the tuition fee?",
    "What are the hostel fees for students?",
    "How do I apply for admission?",
    "What are the eligibility criteria for admission?",
    "What academic programs are available?",
    "What departments are there?",
    "What research opportunities are available?",
    "What research areas are available?",
    "What facilities are available?",
    "What hostel facilities are available?",
    "What are the admission requirements for B.Tech?",
    "What are the admission requirements for Ph.D.?",
    "What is the procedure for applying?",
    "What are the fees for admission?",
    "Where is the institute located?",
    "Can I apply if I do not meet the stated eligibility requirements?",
    "Tell me about the institute's academic programs and research opportunities.",
    "What are the application deadlines?",
)


@dataclass(frozen=True)
class CaseResult:
    question: str
    retrieval_ok: bool
    evidence_ok: bool
    generation_ok: bool
    grounding_ok: bool
    guard_ok: bool
    end_to_end_ok: bool
    evidence_status: str
    coverage_status: str
    verified_count: int
    final_answer: str
    answer_mode: str
    lexical_fallback_used: bool
    failure_reasons: tuple[str, ...]


def _compact(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Node 2 report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Node 2 report root must be a JSON object.")
    return payload


def _cases_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("cases", "results", "items"):
        value = report.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError(
        "Could not find case records in Node 2 report. Expected one of: cases, results, items."
    )


def _question_from_case(case: dict[str, Any]) -> str:
    return _compact(
        case.get("question")
        or case.get("query", {}).get("original_query")
        or case.get("query", {}).get("question")
    )


def _lookup_cases(report_cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_question_from_case(case): case for case in report_cases if _question_from_case(case)}


def _answer_is_fallback(answer: str) -> bool:
    normalized = _compact(answer).casefold()
    return not normalized or normalized.startswith("i'm sorry, i don't know") or normalized.startswith("i am sorry, i don't know")


def _answer_has_internal_leak(answer: str) -> bool:
    patterns = (
        r"\b(?:evidence|document|chunk)\s*\d+\b",
        r"\b(?:the user is asking|the user asks|let me|first[, ]+i|i need to|looking at the evidence|looking through the evidence|the key is|the problem is|hmm|step by step|my reasoning|analysis:)\b",
        r"\b(?:rrf|bm25|retrieval score|retrieval rank|embedding|vector database|source path|chunk id)\b",
    )
    return any(re.search(pattern, answer, flags=re.I) for pattern in patterns)


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def evaluate_case(case: dict[str, Any]) -> CaseResult:
    question = _question_from_case(case)

    retrieval = case.get("retrieval", {}) or {}
    evidence = case.get("evidence", {}) or {}
    generation = case.get("answer_generation", {}) or case.get("generation", {}) or {}
    grounding = case.get("grounding", {}) or {}
    guard = case.get("guard", {}) or {}

    verified_count = _int_value(retrieval.get("verified_count"))

    evidence_status = _compact(evidence.get("status")).casefold()
    coverage_status = _compact(evidence.get("coverage_status") or case.get("coverage", {}).get("status")).casefold()
    package_ready = bool(evidence.get("package_ready"))

    generated = bool(generation.get("generated"))
    final_answer = _compact(
        case.get("final_answer")
        or generation.get("raw_generated_answer")
        or generation.get("answer")
    )

    grounding_status = _compact(grounding.get("status")).casefold()
    guard_status = _compact(guard.get("status")).casefold()
    answer_mode = _compact(generation.get("answer_mode")) or "unknown"
    retrieval_meta = case.get("retrieval", {}) or {}
    lexical_fallback_used = bool(retrieval_meta.get("lexical_fallback_used", False))

    # A sanitized answer is still a successful guard outcome when the guard
    # changed only presentation/internal-leak material and did not fall back
    # or inject replacement facts. Node 2 explicitly reports this as safe.
    guard_ok = guard_status in {"clean", "sanitized_multiple"} and not bool(guard.get("fallback_used"))

    retrieval_ok = verified_count > 0
    evidence_ok = (
        evidence_status in {"supported", "partial"}
        and coverage_status in {"supported", "partial"}
        and package_ready
    )
    generation_ok = generated and not _answer_is_fallback(final_answer) and not _answer_has_internal_leak(final_answer)
    grounding_ok = grounding_status == "grounded"
    end_to_end_ok = retrieval_ok and evidence_ok and generation_ok and grounding_ok and guard_ok

    failures: list[str] = []
    if not retrieval_ok:
        failures.append("no_verified_evidence")
    if evidence_status not in {"supported", "partial"}:
        failures.append(f"evidence_{evidence_status or 'missing'}")
    if coverage_status not in {"supported", "partial"}:
        failures.append(f"coverage_{coverage_status or 'missing'}")
    if not package_ready:
        failures.append("package_not_ready")
    if not generation_ok:
        failures.append("no_clean_user_answer_generated")
    if not grounding_ok:
        failures.append(f"not_grounded:{grounding_status or 'missing'}")
    if not guard_ok:
        failures.append(f"guard_{guard_status or 'missing'}")

    return CaseResult(
        question=question,
        retrieval_ok=retrieval_ok,
        evidence_ok=evidence_ok,
        generation_ok=generation_ok,
        grounding_ok=grounding_ok,
        guard_ok=guard_ok,
        end_to_end_ok=end_to_end_ok,
        evidence_status=evidence_status,
        coverage_status=coverage_status,
        verified_count=verified_count,
        final_answer=final_answer,
        answer_mode=answer_mode,
        lexical_fallback_used=lexical_fallback_used,
        failure_reasons=tuple(failures),
    )


def _run_node2(limit: int) -> Path:
    if not NODE2_GATE.exists():
        raise FileNotFoundError(f"Node 2 gate not found: {NODE2_GATE}")

    command = [
        sys.executable,
        str(NODE2_GATE),
        "--limit",
        str(limit),
    ]
    # Stream Node 2 output live. The previous implementation used
    # capture_output=True, which made a 20-case real-LLM run look frozen until
    # the entire subprocess completed.
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    returncode = process.wait()
    combined_output = "".join(output_lines)

    match = re.search(r"^JSON_REPORT=(.+)$", combined_output, flags=re.MULTILINE)
    report_path = Path(match.group(1).strip()) if match else DEFAULT_REPORT
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    if returncode != 0:
        raise RuntimeError(
            f"Node 2 gate exited with code {returncode}; cannot evaluate the final core gate."
        )

    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Final end-to-end RAG quality gate")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Evaluate an existing Node 2 JSON report instead of running Node 2.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")

    report_path = args.report
    if report_path is None:
        report_path = _run_node2(min(args.limit, len(SEED_QUESTIONS)))

    if not report_path.is_absolute():
        report_path = ROOT / report_path

    report = _load_report(report_path)
    cases = _lookup_cases(_cases_from_report(report))

    selected_questions = SEED_QUESTIONS[: min(args.limit, len(SEED_QUESTIONS))]
    missing = [question for question in selected_questions if question not in cases]
    if missing:
        raise RuntimeError(
            "Node 2 report is missing expected seed questions: "
            + "; ".join(missing)
        )

    results = [evaluate_case(cases[question]) for question in selected_questions]

    retrieval_success = sum(result.retrieval_ok for result in results)
    evidence_success = sum(result.evidence_ok for result in results)
    generation_success = sum(result.generation_ok for result in results)
    grounding_success = sum(result.grounding_ok for result in results)
    guard_success = sum(result.guard_ok for result in results)
    end_to_end_success = sum(result.end_to_end_ok for result in results)
    total = len(results)

    rate = end_to_end_success / total
    required = int((args.threshold * total) + 0.999999)

    print("\nFINAL RAG CORE QUALITY GATE")
    print("=" * 80)
    print(f"Questions evaluated:          {total}")
    print(f"Verified evidence:            {retrieval_success}/{total} ({retrieval_success / total:.1%})")
    print(f"Evidence/package success:     {evidence_success}/{total} ({evidence_success / total:.1%})")
    print(f"Answer generation success:    {generation_success}/{total} ({generation_success / total:.1%})")
    print(f"Grounded answer success:      {grounding_success}/{total} ({grounding_success / total:.1%})")
    print(f"Guard-clean success:          {guard_success}/{total} ({guard_success / total:.1%})")
    print(f"END-TO-END CORE SUCCESS:      {end_to_end_success}/{total} ({rate:.1%})")
    print(f"TARGET:                       >= {args.threshold:.1%} ({required}/{total})")
    print("=" * 80)

    for index, result in enumerate(results, start=1):
        status = "PASS" if result.end_to_end_ok else "FAIL"
        print(
            f"{index:02d}. {status} | verified={result.verified_count} | "
            f"evidence={result.evidence_status or 'missing'} | "
            f"coverage={result.coverage_status or 'missing'} | "
            f"grounding={'grounded' if result.grounding_ok else 'not_grounded'} | "
            f"mode={result.answer_mode} | lexical_fallback={result.lexical_fallback_used} | "
            f"{result.question}"
        )
        if result.failure_reasons:
            print(f"    reasons: {', '.join(result.failure_reasons)}")

    print("=" * 80)
    if end_to_end_success >= required:
        print("FINAL RAG CORE QUALITY GATE: PASS")
        return 0

    print("FINAL RAG CORE QUALITY GATE: FAIL")
    print(
        "The core is not ready for downstream optimization/generalization yet. "
        "Fix the funnel bottleneck before adding more architecture."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())