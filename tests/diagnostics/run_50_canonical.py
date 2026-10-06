from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ai_platform.runtime.engine import ask_with_diagnostics


ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_FILE = ROOT / "tests" / "regression_questions.json"
REPORT_JSON = ROOT / "tests" / "diagnostics_50_canonical.json"
REPORT_TXT = ROOT / "tests" / "diagnostics_50_canonical.txt"


INTERNAL_LEAK_MARKERS = (
    "document 1",
    "document 2",
    "document 3",
    "document 4",
    "document 5",
    "rrf #",
    "rrf rank",
    "final compressed context",
    "final context sent to llm",
    "command 5 knowledge source",
    "command 6",
    "content token count",
    "matched terms:",
    "reranker score",
    "query programs:",
    "document programs:",
    "query topics:",
    "document topics:",
)


UNKNOWN_MARKERS = (
    "i don't know",
    "i do not know",
    "sorry, i don't know",
    "sorry, i do not know",
    "insufficient evidence",
    "not enough information",
    "i don't have enough information",
    "i do not have enough information",
)


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def serializable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if is_dataclass(value):
        try:
            return serializable(asdict(value))
        except Exception:
            return repr(value)

    if hasattr(value, "model_dump"):
        try:
            return serializable(value.model_dump())
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(k): serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            serializable(v)
            for v in value
        ]

    if hasattr(value, "to_dict"):
        try:
            return serializable(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): serializable(v)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass

    return repr(value)


def result_count(result: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = result.get(key)

        if value is None:
            continue

        try:
            return len(value)
        except Exception:
            continue

    return 0


def sources_from_result(result: dict[str, Any]) -> list[str]:
    values: list[str] = []

    for key in (
        "evidence_documents",
        "evidence_candidates",
        "ranked_candidates",
        "verified_candidates",
        "fused_candidates",
        "retrieval_results",
    ):
        items = result.get(key) or ()

        if not isinstance(items, (list, tuple)):
            continue

        for item in items:
            metadata = getattr(item, "metadata", None)

            if metadata is None and isinstance(item, dict):
                metadata = item.get("metadata")

            if not isinstance(metadata, dict):
                continue

            source = (
                metadata.get("source")
                or metadata.get("file_path")
                or metadata.get("path")
                or metadata.get("url")
            )

            source = clean(source)

            if source:
                values.append(source)

    return list(dict.fromkeys(values))


def detect_internal_leaks(answer: str) -> list[str]:
    normalized = clean(answer).casefold()

    return [
        marker
        for marker in INTERNAL_LEAK_MARKERS
        if marker in normalized
    ]


def looks_unknown(answer: str) -> bool:
    normalized = clean(answer).casefold()

    return any(
        marker in normalized
        for marker in UNKNOWN_MARKERS
    )


def extract_query_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    query = result.get("query")
    frame = result.get("query_frame")

    return {
        "resolved_question": clean(
            result.get("resolved_question")
        ),
        "conversation_mode": clean(
            result.get("conversation_mode")
        ),
        "active_topic": clean(
            result.get("active_topic")
        ),
        "active_entity": clean(
            result.get("active_entity")
        ),
        "retrieval_queries": [
            clean(x)
            for x in (
                result.get("retrieval_queries")
                or ()
            )
            if clean(x)
        ],
        "query": serializable(query),
        "query_frame": serializable(frame),
    }


def extract_retrieval_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    retrieval_results = result.get("retrieval_results") or ()

    per_channel: list[int] = []

    if isinstance(retrieval_results, (list, tuple)):
        for channel in retrieval_results:
            try:
                per_channel.append(len(channel))
            except Exception:
                per_channel.append(0)

    return {
        "channel_counts": per_channel,
        "retrieval_result_count": len(retrieval_results),
        "retrieval_weights": serializable(
            result.get("retrieval_weights")
        ),
        "fused_count": result_count(
            result,
            "fused_candidates",
            "fused_docs",
        ),
        "verified_count": result_count(
            result,
            "verified_candidates",
        ),
        "uncertain_count": result_count(
            result,
            "uncertain_candidates",
        ),
        "rejected_count": result_count(
            result,
            "rejected_candidates",
        ),
        "ranked_count": result_count(
            result,
            "ranked_candidates",
            "reranked_docs",
        ),
    }


def extract_evidence_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    assessment = result.get("evidence_assessment")
    coverage = result.get("coverage_assessment") or result.get("evidence_coverage")
    package = result.get("evidence_package")

    return {
        "assessment": serializable(assessment),
        "coverage": serializable(coverage),
        "package": serializable(package),
        "evidence_groups_count": result_count(
            result,
            "evidence_groups",
        ),
        "evidence_documents_count": result_count(
            result,
            "evidence_documents",
        ),
        "evidence_candidates_count": result_count(
            result,
            "evidence_candidates",
        ),
        "evidence_status": clean(
            result.get("evidence_status")
        ),
        "coverage_status": clean(
            result.get("evidence_coverage_status")
        ),
        "question_type": clean(
            result.get("evidence_question_type")
        ),
        "context_characters": len(
            str(
                getattr(package, "context", "")
                if package is not None
                else result.get("context")
                or ""
            ).strip()
        ),
        "sources": sources_from_result(result),
    }


def extract_answer_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    grounding = result.get("answer_grounding") or result.get("grounding_assessment")
    guard = result.get("answer_guard") or result.get("guard_result")

    answer = clean(result.get("answer"))

    return {
        "answer": answer,
        "answer_characters": len(answer),
        "unknown_answer": looks_unknown(answer),
        "grounding": serializable(grounding),
        "guard": serializable(guard),
        "guard_status": clean(
            result.get("answer_guard_status")
        ),
        "guard_reason": clean(
            result.get("answer_guard_reason")
        ),
        "internal_leaks": detect_internal_leaks(answer),
    }


def classify_bottleneck(
    *,
    result: dict[str, Any],
    answer_diag: dict[str, Any],
    retrieval_diag: dict[str, Any],
    evidence_diag: dict[str, Any],
) -> tuple[str, str]:

    if answer_diag["internal_leaks"]:
        return (
            "ANSWER_GUARD",
            "Final answer contains internal retrieval/debug metadata.",
        )

    if not answer_diag["answer"]:
        return (
            "ANSWER",
            "Pipeline completed without a final answer.",
        )

    if answer_diag["unknown_answer"]:
        evidence_status = evidence_diag["evidence_status"]
        coverage_status = evidence_diag["coverage_status"]

        if (
            evidence_status == "insufficient"
            or coverage_status == "insufficient"
        ):
            return (
                "EVIDENCE",
                "System abstained because available evidence was insufficient.",
            )

        return (
            "ANSWER_OR_GROUNDING",
            "System abstained despite non-insufficient evidence.",
        )

    retrieval_count = retrieval_diag["retrieval_result_count"]
    fused_count = retrieval_diag["fused_count"]
    verified_count = retrieval_diag["verified_count"]
    ranked_count = retrieval_diag["ranked_count"]

    if retrieval_count == 0:
        return (
            "RETRIEVAL",
            "No retrieval result channels were produced.",
        )

    if fused_count == 0:
        return (
            "RRF_OR_RETRIEVAL",
            "Retrieval ran but no fused candidates survived.",
        )

    if verified_count == 0 and ranked_count == 0:
        return (
            "VERIFICATION",
            "Candidates existed before verification but none survived.",
        )

    if ranked_count == 0:
        return (
            "RANKING",
            "Verified candidates existed but no ranked candidates survived.",
        )

    evidence_status = evidence_diag["evidence_status"]
    coverage_status = evidence_diag["coverage_status"]

    if evidence_status == "insufficient":
        return (
            "EVIDENCE",
            "Ranked evidence was deemed insufficient.",
        )

    if coverage_status == "insufficient":
        return (
            "COVERAGE",
            "Evidence existed but failed required coverage.",
        )

    if (
        evidence_status
        and evidence_status not in {
            "supported",
            "partially_supported",
        }
    ):
        return (
            "EVIDENCE",
            f"Evidence stage returned status={evidence_status!r}.",
        )

    return (
        "PASS_STRUCTURAL",
        "Retrieval, verification, evidence, and answer stages completed without structural failure.",
    )


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    question = clean(case.get("question"))

    started = time.perf_counter()

    try:
        result = ask_with_diagnostics(
            question,
            chat_history=[],
        )

        elapsed = time.perf_counter() - started

    except Exception as exc:
        elapsed = time.perf_counter() - started

        return {
            "id": case.get("id"),
            "category": case.get("category"),
            "question": question,
            "expected": case.get("expected"),
            "elapsed_seconds": round(elapsed, 3),
            "status": "ENGINE_ERROR",
            "bottleneck": "ENGINE_IMPORT_OR_RUNTIME",
            "diagnosis": f"{type(exc).__name__}: {exc}",
            "query": {},
            "retrieval": {},
            "evidence": {},
            "answer_generation": {},
        }

    query_diag = extract_query_diagnostics(result)
    retrieval_diag = extract_retrieval_diagnostics(result)
    evidence_diag = extract_evidence_diagnostics(result)
    answer_diag = extract_answer_diagnostics(result)

    bottleneck, diagnosis = classify_bottleneck(
        result=result,
        answer_diag=answer_diag,
        retrieval_diag=retrieval_diag,
        evidence_diag=evidence_diag,
    )

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": question,
        "expected": case.get("expected"),
        "elapsed_seconds": round(elapsed, 3),
        "status": "CHECK" if bottleneck != "PASS_STRUCTURAL" else "PASS",
        "bottleneck": bottleneck,
        "diagnosis": diagnosis,
        "query": query_diag,
        "retrieval": retrieval_diag,
        "evidence": evidence_diag,
        "answer_generation": answer_diag,
        "raw_state_keys": sorted(result.keys()),
    }


def render_summary(results: list[dict[str, Any]]) -> str:
    status_counts = Counter(
        item["status"]
        for item in results
    )

    bottleneck_counts = Counter(
        item["bottleneck"]
        for item in results
    )

    total = len(results)

    lines = [
        "=" * 110,
        "IIT JODHPUR — CANONICAL 50-QUESTION RAG DIAGNOSTIC",
        "=" * 110,
        f"TOTAL CASES: {total}",
        f"PASS (STRUCTURAL): {status_counts.get('PASS', 0)}",
        f"CHECK:              {status_counts.get('CHECK', 0)}",
        f"ENGINE ERROR:       {status_counts.get('ENGINE_ERROR', 0)}",
        "",
        "BOTTLENECK DISTRIBUTION",
        "-" * 110,
    ]

    for name, count in bottleneck_counts.most_common():
        lines.append(
            f"{name:<28} {count}"
        )

    lines.extend(
        [
            "",
            "=" * 110,
            "CASE-BY-CASE",
            "=" * 110,
        ]
    )

    for index, item in enumerate(results, start=1):
        retrieval = item.get("retrieval", {})

        lines.extend(
            [
                "",
                f"[{index:02d}/50] {item['status']:<8} "
                f"| {item['bottleneck']:<24} "
                f"| {item['elapsed_seconds']:.2f}s",
                f"Q: {item['question']}",
                f"Diagnosis: {item['diagnosis']}",
                (
                    "Retrieval: "
                    f"channels={retrieval.get('channel_counts', [])} "
                    f"fused={retrieval.get('fused_count', 0)} "
                    f"verified={retrieval.get('verified_count', 0)} "
                    f"ranked={retrieval.get('ranked_count', 0)}"
                ),
                (
                    "Evidence: "
                    f"status={item.get('evidence', {}).get('evidence_status', '')} "
                    f"coverage={item.get('evidence', {}).get('coverage_status', '')} "
                    f"context_chars={item.get('evidence', {}).get('context_characters', 0)}"
                ),
                (
                    "Answer: "
                    + clean(
                        item.get("answer_generation", {}).get("answer")
                    )[:500]
                ),
            ]
        )

        leaks = item.get(
            "answer_generation",
            {},
        ).get(
            "internal_leaks",
            [],
        )

        if leaks:
            lines.append(
                "LEAKS: " + ", ".join(leaks)
            )

    lines.extend(
        [
            "",
            "=" * 110,
            "IMPORTANT",
            "=" * 110,
            "This is a structural diagnostic, not a semantic correctness score.",
            "PASS means the pipeline structurally produced usable evidence and an answer.",
            "CHECK means we have identified a pipeline stage requiring investigation.",
            "The full per-case state is stored in diagnostics_50_canonical.json.",
            "=" * 110,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    if not QUESTIONS_FILE.exists():
        raise SystemExit(
            f"STOP: missing question file: {QUESTIONS_FILE}"
        )

    cases = json.loads(
        QUESTIONS_FILE.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(cases, list):
        raise SystemExit(
            "STOP: regression_questions.json must contain a list."
        )

    if len(cases) < 50:
        raise SystemExit(
            f"STOP: expected at least 50 questions, found {len(cases)}."
        )

    cases = cases[:50]

    print()
    print("=" * 110)
    print("RUNNING CANONICAL 50-QUESTION DIAGNOSTIC")
    print("=" * 110)
    print()
    print(
        f"Question file: {QUESTIONS_FILE}"
    )
    print(
        "Execution path: ai_platform.runtime.engine -> canonical ai_platform graph"
    )
    print()

    results: list[dict[str, Any]] = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        question = clean(case.get("question"))

        print(
            f"[{index:02d}/50] START: {question}"
        )

        result = evaluate_case(case)
        results.append(result)

        print(
            f"[{index:02d}/50] "
            f"{result['status']:<8} "
            f"{result['bottleneck']:<24} "
            f"{result['elapsed_seconds']:.2f}s"
        )

    REPORT_JSON.write_text(
        json.dumps(
            {
                "total": len(results),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = render_summary(results)

    REPORT_TXT.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print()
    print(
        f"JSON REPORT: {REPORT_JSON}"
    )
    print(
        f"TEXT REPORT: {REPORT_TXT}"
    )


if __name__ == "__main__":
    main()
