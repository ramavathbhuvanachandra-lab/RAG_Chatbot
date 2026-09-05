"""
IIT Jodhpur Production Chatbot — Focused Regression

Runs the benchmark in tests/regression_questions.json.

Supported expected labels:

    answerable_in_scope
    unsupported_in_scope

Reports are written incrementally so that results are never lost.

Usage:

    LANGSMITH_TRACING=false python tests/run_regression.py

Optional:

    LANGSMITH_TRACING=false python tests/run_regression.py --batch 1
    LANGSMITH_TRACING=false python tests/run_regression.py --start 1 --end 10
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path


# =========================================================
# Project Path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# =========================================================
# Imports
# =========================================================

from backend.graph import create_graph


# =========================================================
# Paths
# =========================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "tests"
    / "regression_questions.json"
)

REPORT_TXT = (
    PROJECT_ROOT
    / "tests"
    / "regression_report.txt"
)

REPORT_JSON = (
    PROJECT_ROOT
    / "tests"
    / "regression_report.json"
)


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 10

UNKNOWN_PHRASES = [
    "i don't know",
    "i dont know",
    "couldn't find",
    "could not find",
    "not available",
    "not provided",
    "does not contain",
    "insufficient information",
    "not enough information",
    "unable to find",
]

CONTACT_PHRASES = [
    "contact your",
    "contact the",
    "contact student guide",
    "contact swc",
    "contact hwc",
    "student guide",
]

INTERNAL_REFERENCE_PHRASES = [
    "document 1",
    "document 2",
    "document 3",
    "document 4",
    "document 5",
    "rrf score",
    "retrieval rank",
    "chunk id",
]


# =========================================================
# Load Questions
# =========================================================

def load_questions():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing regression file:\n{QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "regression_questions.json must contain a JSON list."
        )

    required = {
        "id",
        "category",
        "question",
        "expected",
    }

    for index, item in enumerate(
        questions,
        start=1,
    ):
        missing = required - set(item.keys())

        if missing:
            raise ValueError(
                f"Question #{index} missing fields: {missing}"
            )

    return questions


# =========================================================
# Create Graph
# =========================================================

print("Loading production graph...", flush=True)

GRAPH = create_graph()

print("Production graph loaded.", flush=True)


# =========================================================
# Helpers
# =========================================================

def clean_text(text):
    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def get_source(document):
    if not document:
        return "UNKNOWN"

    source = document.metadata.get(
        "source",
        "UNKNOWN",
    )

    return str(source).replace(
        "\\",
        "/",
    )


def contains_unknown_response(answer):
    normalized = answer.lower()

    return any(
        phrase in normalized
        for phrase in UNKNOWN_PHRASES
    )


def contains_contact_fallback(answer):
    normalized = answer.lower()

    return any(
        phrase in normalized
        for phrase in CONTACT_PHRASES
    )


def contains_internal_reference(answer):
    normalized = answer.lower()

    return any(
        phrase in normalized
        for phrase in INTERNAL_REFERENCE_PHRASES
    )


# =========================================================
# Classification
# =========================================================

def classify_result(
    expected,
    answer,
    final_docs,
):
    has_evidence = bool(
        final_docs
    )

    has_answer = bool(
        answer.strip()
    )

    unknown = contains_unknown_response(
        answer
    )

    contact_fallback = contains_contact_fallback(
        answer
    )

    internal_leak = contains_internal_reference(
        answer
    )

    # -----------------------------------------------------
    # Unsupported question
    # -----------------------------------------------------

    if expected == "unsupported_in_scope":

        if not unknown:
            return "FAIL_ANSWERED_UNSUPPORTED"

        if contact_fallback:
            return "FAIL_CONTACT_FALLBACK"

        if internal_leak:
            return "FAIL_INTERNAL_LEAK"

        return "PASS"

    # -----------------------------------------------------
    # Answerable question
    # -----------------------------------------------------

    if expected == "answerable_in_scope":

        if not has_evidence:
            return "FAIL_NO_EVIDENCE"

        if not has_answer:
            return "FAIL_EMPTY_ANSWER"

        if contact_fallback:
            return "FAIL_CONTACT_FALLBACK"

        if internal_leak:
            return "FAIL_INTERNAL_LEAK"

        if unknown:
            return "REVIEW_UNEXPECTED_UNKNOWN"

        # Retrieval + answer exist.
        # Semantic correctness still requires review.
        return "REVIEW"

    return "ERROR_EXPECTATION"


# =========================================================
# Run One Question
# =========================================================

def run_question(item):
    question = item["question"]
    expected = item["expected"]

    started = time.perf_counter()

    try:
        result = GRAPH.invoke(
            {
                "question": question,
                "chat_history": [],
            }
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        answer = clean_text(
            result.get(
                "answer",
                "",
            )
        )

        retrieval_results = result.get(
            "retrieval_results",
            [],
        )

        fused_docs = result.get(
            "fused_docs",
            [],
        )

        final_docs = result.get(
            "compressed_docs",
            [],
        )

        dense_count = (
            len(retrieval_results[0])
            if len(retrieval_results) >= 1
            else 0
        )

        bm25_count = (
            len(retrieval_results[1])
            if len(retrieval_results) >= 2
            else 0
        )

        sources = [
            get_source(document)
            for document in final_docs
        ]

        status = classify_result(
            expected=expected,
            answer=answer,
            final_docs=final_docs,
        )

        return {
            "id": item["id"],
            "category": item["category"],
            "question": question,
            "expected": expected,
            "status": status,
            "elapsed_seconds": round(
                elapsed,
                3,
            ),
            "dense_count": dense_count,
            "bm25_count": bm25_count,
            "fused_count": len(fused_docs),
            "final_count": len(final_docs),
            "sources": sources,
            "answer": answer,
            "error": None,
        }

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - started
        )

        return {
            "id": item["id"],
            "category": item["category"],
            "question": question,
            "expected": expected,
            "status": "ERROR",
            "elapsed_seconds": round(
                elapsed,
                3,
            ),
            "dense_count": 0,
            "bm25_count": 0,
            "fused_count": 0,
            "final_count": 0,
            "sources": [],
            "answer": "",
            "error": repr(exc),
        }


# =========================================================
# Append One Result to TXT
# =========================================================

def append_result_to_txt(
    result,
    number,
    total,
):
    """
    Write immediately after each question.

    This prevents losing the entire report if execution stops.
    """

    with open(
        REPORT_TXT,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n"
            + "=" * 110
            + "\n"
        )

        file.write(
            f"[{number}/{total}] "
            f"{result['id']} | "
            f"{result['status']} | "
            f"{result['category']}\n\n"
        )

        file.write(
            f"Question:\n"
            f"{result['question']}\n\n"
        )

        file.write(
            f"Expected:\n"
            f"{result['expected']}\n\n"
        )

        file.write(
            f"Latency:\n"
            f"{result['elapsed_seconds']}s\n\n"
        )

        file.write(
            "Retrieval:\n"
        )

        file.write(
            f"  Dense: {result['dense_count']}\n"
        )

        file.write(
            f"  BM25: {result['bm25_count']}\n"
        )

        file.write(
            f"  Fused: {result['fused_count']}\n"
        )

        file.write(
            f"  Final: {result['final_count']}\n\n"
        )

        file.write(
            "Sources:\n"
        )

        for source in result["sources"]:
            file.write(
                f"  - {source}\n"
            )

        file.write(
            "\nAnswer:\n"
        )

        file.write(
            result["answer"]
            + "\n"
        )

        if result["error"]:
            file.write(
                "\nError:\n"
            )

            file.write(
                result["error"]
                + "\n"
            )

        file.flush()


# =========================================================
# Write Full JSON Snapshot
# =========================================================

def write_json_snapshot(
    results,
):
    """
    Rewrite the JSON report after every completed question.
    """

    payload = {
        "generated_at":
            datetime.now().isoformat(),

        "question_file":
            str(QUESTIONS_FILE),

        "total_completed":
            len(results),

        "results":
            results,
    }

    temp_file = REPORT_JSON.with_suffix(
        ".json.tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.flush()

    temp_file.replace(
        REPORT_JSON
    )


# =========================================================
# Reset Reports Before Run
# =========================================================

def initialize_reports():
    """
    Create non-empty files immediately.
    """

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "IIT JODHPUR PRODUCTION CHATBOT REGRESSION\n"
        )

        file.write(
            "=" * 110
            + "\n"
        )

        file.write(
            f"Started: "
            f"{datetime.now().isoformat()}\n"
        )

        file.write(
            f"Questions file: "
            f"{QUESTIONS_FILE}\n"
        )

        file.write(
            "\nResults will be appended after every completed question.\n"
        )

        file.flush()

    write_json_snapshot(
        []
    )


# =========================================================
# Terminal Output
# =========================================================

def print_result(
    number,
    total,
    result,
):
    answer = result["answer"]

    if len(answer) > 220:
        answer = (
            answer[:220]
            + "..."
        )

    print()
    print(
        f"[{number:02d}/{total:02d}] "
        f"{result['status']:<28} "
        f"{result['elapsed_seconds']:>6.2f}s "
        f"| {result['category']}",
        flush=True,
    )

    print(
        f"Q: {result['question']}",
        flush=True,
    )

    print(
        "R: "
        f"dense={result['dense_count']} "
        f"bm25={result['bm25_count']} "
        f"fused={result['fused_count']} "
        f"final={result['final_count']}",
        flush=True,
    )

    print(
        f"A: {answer}",
        flush=True,
    )

    if result["error"]:
        print(
            f"ERROR: {result['error']}",
            flush=True,
        )


# =========================================================
# Summary
# =========================================================

def build_summary(
    results,
):
    total = len(results)

    answerable = [
        item
        for item in results
        if item["expected"]
        == "answerable_in_scope"
    ]

    unsupported = [
        item
        for item in results
        if item["expected"]
        == "unsupported_in_scope"
    ]

    status_counts = {}

    for item in results:
        status = item["status"]

        status_counts[status] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    total_seconds = sum(
        item["elapsed_seconds"]
        for item in results
    )

    return {
        "total_completed":
            total,

        "answerable_in_scope":
            len(answerable),

        "unsupported_in_scope":
            len(unsupported),

        "unsupported_pass":
            sum(
                item["status"] == "PASS"
                for item in unsupported
            ),

        "unsupported_answered":
            sum(
                item["status"]
                == "FAIL_ANSWERED_UNSUPPORTED"
                for item in unsupported
            ),

        "contact_fallback_failures":
            sum(
                item["status"]
                == "FAIL_CONTACT_FALLBACK"
                for item in results
            ),

        "internal_leak_failures":
            sum(
                item["status"]
                == "FAIL_INTERNAL_LEAK"
                for item in results
            ),

        "errors":
            sum(
                item["status"] == "ERROR"
                for item in results
            ),

        "average_seconds":
            round(
                total_seconds / total,
                3,
            )
            if total
            else 0.0,

        "total_seconds":
            round(
                total_seconds,
                3,
            ),

        "status_counts":
            status_counts,
    }


# =========================================================
# Finalize TXT Report
# =========================================================

def finalize_txt_report(
    summary,
):
    with open(
        REPORT_TXT,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n"
            + "=" * 110
            + "\n"
        )

        file.write(
            "REGRESSION SUMMARY\n"
        )

        file.write(
            "=" * 110
            + "\n"
        )

        for key, value in summary.items():

            file.write(
                f"{key}: {value}\n"
            )

        file.flush()


# =========================================================
# Selection
# =========================================================

def select_questions(
    questions,
    batch=None,
    start=None,
    end=None,
):
    if batch is not None:

        start_index = (
            batch - 1
        ) * BATCH_SIZE

        end_index = (
            start_index
            + BATCH_SIZE
        )

        return questions[
            start_index:end_index
        ]

    if start is not None:

        start_index = (
            start - 1
        )

        end_index = (
            end
            if end is not None
            else len(questions)
        )

        return questions[
            start_index:end_index
        ]

    return questions


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        type=int,
        help="Run one batch of 10 questions.",
    )

    parser.add_argument(
        "--start",
        type=int,
        help="Starting question number.",
    )

    parser.add_argument(
        "--end",
        type=int,
        help="Ending question number.",
    )

    args = parser.parse_args()

    questions = load_questions()

    selected_questions = select_questions(
        questions=questions,
        batch=args.batch,
        start=args.start,
        end=args.end,
    )

    if not selected_questions:

        raise ValueError(
            "No questions selected."
        )

    initialize_reports()

    print()
    print(
        "=" * 110,
        flush=True,
    )

    print(
        "IIT JODHPUR — FOCUSED PRODUCTION REGRESSION",
        flush=True,
    )

    print(
        "=" * 110,
        flush=True,
    )

    print(
        f"Questions selected: "
        f"{len(selected_questions)}",
        flush=True,
    )

    print(
        f"Questions available: "
        f"{len(questions)}",
        flush=True,
    )

    print(
        f"TXT report: "
        f"{REPORT_TXT}",
        flush=True,
    )

    print(
        f"JSON report: "
        f"{REPORT_JSON}",
        flush=True,
    )

    print(
        "Reports are written after every question.",
        flush=True,
    )

    results = []

    total = len(
        selected_questions
    )

    for number, question in enumerate(
        selected_questions,
        start=1,
    ):

        result = run_question(
            question
        )

        results.append(
            result
        )

        # -------------------------------------------------
        # Persist immediately
        # -------------------------------------------------

        append_result_to_txt(
            result,
            number,
            total,
        )

        write_json_snapshot(
            results
        )

        print_result(
            number,
            total,
            result,
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    summary = build_summary(
        results
    )

    finalize_txt_report(
        summary
    )

    write_json_snapshot(
        results
    )

    print()
    print(
        "=" * 110
    )

    print(
        "REGRESSION SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"Total completed: "
        f"{summary['total_completed']}"
    )

    print(
        f"Answerable in scope: "
        f"{summary['answerable_in_scope']}"
    )

    print(
        f"Unsupported in scope: "
        f"{summary['unsupported_in_scope']}"
    )

    print(
        f"Unsupported correctly refused: "
        f"{summary['unsupported_pass']}"
    )

    print(
        f"Unsupported answered: "
        f"{summary['unsupported_answered']}"
    )

    print(
        f"Contact fallback failures: "
        f"{summary['contact_fallback_failures']}"
    )

    print(
        f"Internal leakage failures: "
        f"{summary['internal_leak_failures']}"
    )

    print(
        f"Errors: "
        f"{summary['errors']}"
    )

    print(
        f"Average latency: "
        f"{summary['average_seconds']:.3f}s"
    )

    print(
        f"Total latency: "
        f"{summary['total_seconds']:.3f}s"
    )

    print()
    print(
        f"Full TXT report:\n{REPORT_TXT}"
    )

    print(
        f"\nFull JSON report:\n{REPORT_JSON}"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()