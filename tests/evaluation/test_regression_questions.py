"""
Production Regression Runner — Standard Questions

Purpose
-------
Run the real production graph against the standard regression question set.

Persistence
-----------
Results are persisted incrementally after EVERY question.

Crash-safe behavior:
    - A run file is created before question 1.
    - Every successful/failed question is written immediately.
    - If the process crashes, completed successful questions remain saved.
    - Re-running the same regression automatically resumes from the first
      unfinished/failed question.

A completed run starts a new run on the next invocation.

Files
-----
tests/evaluation/results/standard/
    <run_id>.json
    <run_id>.txt
    latest.json
    latest.txt
"""

from __future__ import annotations

import faulthandler
import json
import os
import statistics
import time

from datetime import datetime
from pathlib import Path
from typing import Any

from backend.graph import create_graph

from tests.evaluation.regression_questions import (
    REGRESSION_QUESTIONS,
)


# =========================================================
# Configuration
# =========================================================

REGRESSION_WARNING_SECONDS = 20.0

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "standard"
)

LATEST_JSON = (
    RESULTS_DIR
    / "latest.json"
)

LATEST_TXT = (
    RESULTS_DIR
    / "latest.txt"
)

# Set REGRESSION_FRESH=1 when you explicitly want a brand-new run
# even if latest.json contains an unfinished run.
FORCE_FRESH = (
    os.environ.get(
        "REGRESSION_FRESH",
        "",
    ).strip()
    == "1"
)


# =========================================================
# Time / filesystem helpers
# =========================================================

def _timestamp() -> str:
    """
    Return a filesystem-safe local timestamp.
    """
    return datetime.now().astimezone().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


def _iso_timestamp() -> str:
    """
    Return an ISO-8601 timestamp with timezone.
    """
    return datetime.now().astimezone().isoformat()


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """
    Atomically replace a text file.

    This protects the checkpoint from being left half-written if the
    process is interrupted during persistence.
    """
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_path.write_text(
        content,
        encoding="utf-8",
    )

    temp_path.replace(
        path
    )


# =========================================================
# JSON helpers
# =========================================================

def _safe_json_value(
    value: Any,
) -> Any:
    """
    Convert common project values into JSON-safe values.
    """
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _safe_json_value(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        return [
            _safe_json_value(
                item
            )
            for item in value
        ]

    return str(
        value
    )


def _load_latest_run() -> dict[str, Any] | None:
    """
    Load latest.json when it exists and contains a valid run.
    """
    if not LATEST_JSON.exists():
        return None

    try:
        return json.loads(
            LATEST_JSON.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None


# =========================================================
# Metrics
# =========================================================

def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    """
    Calculate a percentile using linear interpolation.
    """
    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile
    )

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def _context_character_count(
    result: dict[str, Any],
) -> int:
    """
    Count characters in the final answer context.
    """
    context = result.get(
        "context",
        "",
    )

    if isinstance(
        context,
        str,
    ):
        return len(
            context
        )

    return 0


def _evidence_group_count(
    result: dict[str, Any],
) -> int:
    """
    Count final evidence groups.
    """
    groups = result.get(
        "evidence_groups",
        [],
    )

    if isinstance(
        groups,
        (list, tuple),
    ):
        return len(
            groups
        )

    return 0


def _intent_count(
    result: dict[str, Any],
) -> int:
    """
    Safely read intent count.
    """
    try:
        return int(
            result.get(
                "intent_count",
                1,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return 1


def _build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build aggregate diagnostic statistics.
    """
    completed = [
        item
        for item in results
        if item.get(
            "status"
        ) == "completed"
    ]

    failed = [
        item
        for item in results
        if item.get(
            "status"
        ) == "error"
    ]

    latencies = [
        float(
            item["latency"]
        )
        for item in completed
    ]

    groups = [
        int(
            item.get(
                "groups",
                0,
            )
        )
        for item in completed
    ]

    contexts = [
        int(
            item.get(
                "context_characters",
                0,
            )
        )
        for item in completed
    ]

    intents = [
        int(
            item.get(
                "intent_count",
                1,
            )
        )
        for item in completed
    ]

    summary = {
        "total_questions": len(
            REGRESSION_QUESTIONS
        ),
        "recorded_questions": len(
            results
        ),
        "completed": len(
            completed
        ),
        "errors": len(
            failed
        ),
        "remaining": (
            len(REGRESSION_QUESTIONS)
            - len(completed)
        ),
        "average_latency_seconds": (
            statistics.mean(
                latencies
            )
            if latencies
            else 0.0
        ),
        "p50_latency_seconds": _percentile(
            latencies,
            0.50,
        ),
        "p95_latency_seconds": _percentile(
            latencies,
            0.95,
        ),
        "minimum_latency_seconds": (
            min(latencies)
            if latencies
            else 0.0
        ),
        "maximum_latency_seconds": (
            max(latencies)
            if latencies
            else 0.0
        ),
        "total_evidence_groups": sum(
            groups
        ),
        "average_evidence_groups": (
            statistics.mean(
                groups
            )
            if groups
            else 0.0
        ),
        "total_context_characters": sum(
            contexts
        ),
        "average_context_characters": (
            statistics.mean(
                contexts
            )
            if contexts
            else 0.0
        ),
        "multi_intent_questions": sum(
            count > 1
            for count in intents
        ),
    }

    if completed:

        slowest = max(
            completed,
            key=lambda item: item[
                "latency"
            ],
        )

        fastest = min(
            completed,
            key=lambda item: item[
                "latency"
            ],
        )

        summary[
            "slowest_latency_seconds"
        ] = slowest[
            "latency"
        ]

        summary[
            "slowest_question"
        ] = slowest[
            "question"
        ]

        summary[
            "fastest_latency_seconds"
        ] = fastest[
            "latency"
        ]

        summary[
            "fastest_question"
        ] = fastest[
            "question"
        ]

    else:

        summary[
            "slowest_latency_seconds"
        ] = 0.0

        summary[
            "slowest_question"
        ] = ""

        summary[
            "fastest_latency_seconds"
        ] = 0.0

        summary[
            "fastest_question"
        ] = ""

    return summary


# =========================================================
# Report rendering
# =========================================================

def _build_text_report(
    *,
    payload: dict[str, Any],
) -> str:
    """
    Build the human-readable report from the same persisted payload.
    """
    summary = payload[
        "summary"
    ]

    results = payload[
        "results"
    ]

    lines: list[str] = []

    lines.append(
        "=" * 120
    )

    lines.append(
        "IITJ V1 — STANDARD REGRESSION REPORT"
    )

    lines.append(
        "=" * 120
    )

    lines.append(
        ""
    )

    lines.append(
        f"Run ID: {payload['run_id']}"
    )

    lines.append(
        f"Status: {payload['run_status']}"
    )

    lines.append(
        f"Started: {payload['started_at']}"
    )

    lines.append(
        f"Last updated: {payload['updated_at']}"
    )

    lines.append(
        ""
    )

    lines.append(
        "SUMMARY"
    )

    lines.append(
        "-" * 120
    )

    for key, value in summary.items():

        if isinstance(
            value,
            float,
        ):
            value = f"{value:.2f}"

        lines.append(
            f"{key}: {value}"
        )

    lines.append(
        ""
    )

    lines.append(
        "=" * 120
    )

    for item in results:

        index = item[
            "index"
        ]

        lines.append(
            ""
        )

        lines.append(
            "-" * 120
        )

        lines.append(
            f"[{index}/{len(REGRESSION_QUESTIONS)}] "
            f"{item['question']}"
        )

        lines.append(
            f"Status: {item['status']}"
        )

        lines.append(
            f"Latency: {item['latency']:.2f}s"
        )

        lines.append(
            f"Evidence: {item.get('evidence_status', '')} | "
            f"Coverage: {item.get('coverage_status', '')} | "
            f"Groups: {item.get('groups', 0)} | "
            f"Intents: {item.get('intent_count', 1)}"
        )

        lines.append(
            f"Context characters: "
            f"{item.get('context_characters', 0)}"
        )

        if item.get(
            "answer_guard_status"
        ):
            lines.append(
                f"Answer guard: "
                f"{item['answer_guard_status']} | "
                f"{item.get('answer_guard_reason', '')}"
            )

        if item.get(
            "grounding_status"
        ):
            lines.append(
                f"Grounding: "
                f"{item['grounding_status']}"
            )

        if item.get(
            "error"
        ):
            lines.append(
                "ERROR:"
            )

            lines.append(
                str(
                    item["error"]
                )
            )

        lines.append(
            "ANSWER:"
        )

        lines.append(
            item.get(
                "answer",
                "",
            )
        )

    lines.append(
        ""
    )

    lines.append(
        "=" * 120
    )

    return "\n".join(
        lines
    )


# =========================================================
# Persistence
# =========================================================

def _persist_run(
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """
    Persist the complete run atomically.

    This function is called after EVERY question, so the current state
    is always available on disk.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = payload[
        "run_id"
    ]

    json_path = (
        RESULTS_DIR
        / f"{run_id}.json"
    )

    txt_path = (
        RESULTS_DIR
        / f"{run_id}.txt"
    )

    json_content = json.dumps(
        _safe_json_value(
            payload
        ),
        indent=2,
        ensure_ascii=False,
    )

    txt_content = _build_text_report(
        payload=payload
    )

    _atomic_write_text(
        json_path,
        json_content,
    )

    _atomic_write_text(
        txt_path,
        txt_content,
    )

    # latest.* always points to the current active/most recent run.
    _atomic_write_text(
        LATEST_JSON,
        json_content,
    )

    _atomic_write_text(
        LATEST_TXT,
        txt_content,
    )

    return (
        json_path,
        txt_path,
    )


# =========================================================
# Run initialization / resume
# =========================================================

def _create_new_run() -> dict[str, Any]:
    """
    Create a new empty regression run.
    """
    now = _iso_timestamp()

    payload = {
        "run_id": _timestamp(),
        "type": "standard",
        "run_status": "running",
        "started_at": now,
        "updated_at": now,
        "completed_at": "",
        "summary": _build_summary(
            []
        ),
        "results": [],
    }

    _persist_run(
        payload
    )

    return payload


def _load_or_create_run() -> tuple[dict[str, Any], bool]:
    """
    Resume an unfinished latest run.

    Returns:
        payload, resumed
    """
    if FORCE_FRESH:
        return (
            _create_new_run(),
            False,
        )

    latest = _load_latest_run()

    if (
        latest
        and latest.get(
            "type"
        ) == "standard"
        and latest.get(
            "run_status"
        ) == "running"
    ):
        return (
            latest,
            True,
        )

    return (
        _create_new_run(),
        False,
    )


def _successful_indices(
    results: list[dict[str, Any]],
) -> set[int]:
    """
    Return question indices that completed successfully.

    Errors are deliberately NOT treated as completed so they are retried
    on a resumed run.
    """
    return {
        int(
            item["index"]
        )
        for item in results
        if item.get(
            "status"
        ) == "completed"
    }


# =========================================================
# Regression
# =========================================================

def test_regression_question_set():

    faulthandler.enable()

    payload, resumed = (
        _load_or_create_run()
    )

    results = payload[
        "results"
    ]

    successful_indices = (
        _successful_indices(
            results
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    if resumed:

        print(
            "RESUMING EXISTING STANDARD REGRESSION"
        )

        print(
            f"Run ID: {payload['run_id']}"
        )

        print(
            f"Already completed: "
            f"{len(successful_indices)}"
        )

    else:

        print(
            "STARTING NEW STANDARD REGRESSION"
        )

        print(
            f"Run ID: {payload['run_id']}"
        )

    print(
        f"Total questions: "
        f"{len(REGRESSION_QUESTIONS)}"
    )

    print(
        "=" * 120
    )

    graph = create_graph()

    for index, question in enumerate(
        REGRESSION_QUESTIONS,
        start=1,
    ):

        # -----------------------------------------------------
        # Successful questions are skipped on resume.
        # -----------------------------------------------------

        if index in successful_indices:

            print(
                f"[{index}/{len(REGRESSION_QUESTIONS)}] "
                f"SKIP — already completed",
                flush=True,
            )

            continue

        print(
            "\n"
            + "-" * 120,
            flush=True,
        )

        print(
            f"[{index}/{len(REGRESSION_QUESTIONS)}] "
            f"START: {question}",
            flush=True,
        )

        started = time.perf_counter()

        # -----------------------------------------------------
        # Remove any old error record for this question before retry.
        # -----------------------------------------------------

        results = [
            item
            for item in results
            if int(
                item.get(
                    "index",
                    -1,
                )
            )
            != index
        ]

        try:

            result = graph.invoke(
                {
                    "question": question,
                    "chat_history": [],
                }
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            answer = str(
                result.get(
                    "answer",
                    "",
                )
                or ""
            )

            item = {
                "index": index,
                "question": question,
                "status": "completed",
                "answer": answer,
                "latency": elapsed,
                "evidence_status": result.get(
                    "evidence_status",
                    "",
                ),
                "coverage_status": result.get(
                    "evidence_coverage_status",
                    "",
                ),
                "groups": _evidence_group_count(
                    result
                ),
                "intent_count": _intent_count(
                    result
                ),
                "context_characters": (
                    _context_character_count(
                        result
                    )
                ),
                "answer_guard_status": result.get(
                    "answer_guard_status",
                    "",
                ),
                "answer_guard_reason": result.get(
                    "answer_guard_reason",
                    "",
                ),
                "grounding_status": result.get(
                    "answer_grounding_status",
                    "",
                ),
                "error": "",
            }

            print(
                f"[{index}/{len(REGRESSION_QUESTIONS)}] "
                f"DONE: {elapsed:.2f}s",
                flush=True,
            )

            print(
                f"Evidence: "
                f"{item['evidence_status']} | "
                f"Coverage: "
                f"{item['coverage_status']} | "
                f"Groups: "
                f"{item['groups']} | "
                f"Intents: "
                f"{item['intent_count']}",
                flush=True,
            )

            print(
                "ANSWER:",
                flush=True,
            )

            print(
                answer,
                flush=True,
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            item = {
                "index": index,
                "question": question,
                "status": "error",
                "answer": "",
                "latency": elapsed,
                "evidence_status": "error",
                "coverage_status": "error",
                "groups": 0,
                "intent_count": 1,
                "context_characters": 0,
                "answer_guard_status": "",
                "answer_guard_reason": "",
                "grounding_status": "",
                "error": repr(exc),
            }

            print(
                f"[{index}/{len(REGRESSION_QUESTIONS)}] "
                f"FAILED after {elapsed:.2f}s",
                flush=True,
            )

            print(
                "EXCEPTION:",
                repr(exc),
                flush=True,
            )

        # -----------------------------------------------------
        # Persist THIS question immediately.
        # -----------------------------------------------------

        results.append(
            item
        )

        payload[
            "results"
        ] = sorted(
            results,
            key=lambda value: int(
                value["index"]
            ),
        )

        payload[
            "updated_at"
        ] = _iso_timestamp()

        payload[
            "summary"
        ] = _build_summary(
            payload["results"]
        )

        json_path, txt_path = (
            _persist_run(
                payload
            )
        )

        print(
            f"SAVED CHECKPOINT: "
            f"{json_path}",
            flush=True,
        )

    # ---------------------------------------------------------
    # Final status.
    # ---------------------------------------------------------

    successful_count = len(
        _successful_indices(
            payload["results"]
        )
    )

    if successful_count == len(
        REGRESSION_QUESTIONS
    ):
        payload[
            "run_status"
        ] = "completed"

        payload[
            "completed_at"
        ] = _iso_timestamp()

    else:
        payload[
            "run_status"
        ] = "running"

    payload[
        "updated_at"
    ] = _iso_timestamp()

    payload[
        "summary"
    ] = _build_summary(
        payload["results"]
    )

    json_path, txt_path = (
        _persist_run(
            payload
        )
    )

    # ---------------------------------------------------------
    # Final console summary.
    # ---------------------------------------------------------

    summary = payload[
        "summary"
    ]

    print(
        "\n"
        + "=" * 120,
        flush=True,
    )

    print(
        "STANDARD REGRESSION SUMMARY",
        flush=True,
    )

    print(
        f"Status: "
        f"{payload['run_status']}",
        flush=True,
    )

    print(
        f"Total: "
        f"{summary['total_questions']}",
        flush=True,
    )

    print(
        f"Completed: "
        f"{summary['completed']}",
        flush=True,
    )

    print(
        f"Errors: "
        f"{summary['errors']}",
        flush=True,
    )

    print(
        f"Remaining: "
        f"{summary['remaining']}",
        flush=True,
    )

    print(
        f"Average latency: "
        f"{summary['average_latency_seconds']:.2f}s",
        flush=True,
    )

    print(
        f"P50 latency: "
        f"{summary['p50_latency_seconds']:.2f}s",
        flush=True,
    )

    print(
        f"P95 latency: "
        f"{summary['p95_latency_seconds']:.2f}s",
        flush=True,
    )

    print(
        f"Min latency: "
        f"{summary['minimum_latency_seconds']:.2f}s",
        flush=True,
    )

    print(
        f"Max latency: "
        f"{summary['maximum_latency_seconds']:.2f}s",
        flush=True,
    )

    print(
        f"Total evidence groups: "
        f"{summary['total_evidence_groups']}",
        flush=True,
    )

    print(
        f"Average evidence groups: "
        f"{summary['average_evidence_groups']:.2f}",
        flush=True,
    )

    print(
        f"Total context characters: "
        f"{summary['total_context_characters']}",
        flush=True,
    )

    print(
        f"Average context characters: "
        f"{summary['average_context_characters']:.0f}",
        flush=True,
    )

    print(
        f"Multi-intent questions: "
        f"{summary['multi_intent_questions']}",
        flush=True,
    )

    print(
        f"Slowest: "
        f"{summary['slowest_latency_seconds']:.2f}s",
        flush=True,
    )

    print(
        f"Slowest query: "
        f"{summary['slowest_question']}",
        flush=True,
    )

    print(
        f"JSON: {json_path}",
        flush=True,
    )

    print(
        f"TXT: {txt_path}",
        flush=True,
    )

    print(
        "=" * 120,
        flush=True,
    )

    # ---------------------------------------------------------
    # Diagnostic runner assertion.
    #
    # Errors are recorded and retried on subsequent execution.
    # We only require at least one result to have been recorded.
    # ---------------------------------------------------------

    assert payload[
        "results"
    ]