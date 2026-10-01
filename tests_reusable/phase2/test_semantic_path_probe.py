from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from backend.core.query_understanding import understand_query
from backend.core.semantic_query_gate import decide_semantic_query
from backend.situation_retrieval_integration import (
    prepare_situation_retrieval,
)


OUTPUT_PATH = Path(
    "test_outputs/phase2_semantic_path_probe.txt"
)


TEST_CASES = [
    {
        "id": "Q1",
        "type": "meaning paraphrase",
        "query": (
            "Main reserved category ka student hoon, "
            "kya mujhe tuition fee bharni padegi?"
        ),
    },
    {
        "id": "Q2",
        "type": "meaning + multiple qualifiers",
        "query": (
            "Mehmaan ke liye ek room chahiye, "
            "ek din ke liye, aur bed ka samaan bhi chahiye. "
            "Kitna lagega?"
        ),
    },
    {
        "id": "Q3",
        "type": "natural student language",
        "query": (
            "M.Tech ka jo subject clear nahi hua usko "
            "summer mein dobara lene kaise hota hai?"
        ),
    },
    {
        "id": "Q4",
        "type": "semantic institutional question",
        "query": (
            "Institute mein admission lene ke "
            "alag-alag modes kaun se hain?"
        ),
    },
    {
        "id": "Q5",
        "type": "unknown / adversarial",
        "query": (
            "Bhai woh free wala sachar kya scene hai "
            "aur usme kitna paisa milta hai?"
        ),
    },
]


def serialize(value):
    """
    Convert dataclass-like objects into readable diagnostics.
    """
    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, dict):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            serialize(item)
            for item in value
        ]

    return value


def compact(value):
    """
    JSON formatting for stable human-readable output.
    """
    return json.dumps(
        serialize(value),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def normalize_set(values):
    return {
        str(value).strip().casefold()
        for value in (
            values or []
        )
        if str(value).strip()
    }


def query_novelty(
    original_query: str,
    semantic_query: str,
):
    """
    Diagnostic only.

    Shows terms introduced by semantic understanding
    that were not literally present in the original wording.

    This is NOT a correctness metric.
    """
    original_tokens = set(
        original_query.casefold().split()
    )

    semantic_tokens = set(
        semantic_query.casefold().split()
    )

    return sorted(
        semantic_tokens
        - original_tokens
    )


def run_case(case):
    query = case["query"]

    print()
    print("=" * 100)
    print(
        case["id"],
        "-",
        case["type"],
    )
    print(
        "USER QUERY:",
        query,
    )
    print("=" * 100)

    # =========================================================
    # Existing deterministic semantic path
    # =========================================================

    situation_started = time.perf_counter()

    existing = prepare_situation_retrieval(
        query
    )

    situation_latency = (
        time.perf_counter()
        - situation_started
    )

    situation = existing.get(
        "student_situation"
    )

    decision_context = existing.get(
        "decision_context"
    )

    retrieval_plan = existing.get(
        "retrieval_plan"
    )

    retrieval_control = existing.get(
        "retrieval_control"
    )

    print("\nEXISTING SEMANTIC PATH")

    print(
        "StudentSituation:"
    )
    print(
        compact(situation)
    )

    print(
        "\nDecisionContext:"
    )
    print(
        compact(decision_context)
    )

    print(
        "\nRetrievalPlan:"
    )
    print(
        compact(retrieval_plan)
    )

    print(
        "\nRetrievalControl:"
    )
    print(
        compact(retrieval_control)
    )

    print(
        "\nExisting-path latency_sec:",
        round(
            situation_latency,
            3,
        ),
    )

    # =========================================================
    # New semantic query understanding
    # =========================================================

    understanding_started = time.perf_counter()

    understanding = understand_query(
        query
    )

    understanding_latency = (
        time.perf_counter()
        - understanding_started
    )

    decision = decide_semantic_query(
        understanding
    )

    print(
        "\nLLM SEMANTIC UNDERSTANDING:"
    )

    print(
        "original_query:",
        understanding.original_query,
    )

    print(
        "search_query:",
        understanding.search_query,
    )

    print(
        "intent:",
        understanding.intent,
    )

    print(
        "is_list_question:",
        understanding.is_list_question,
    )

    print(
        "confidence:",
        understanding.confidence,
    )

    print(
        "latency_sec:",
        round(
            understanding_latency,
            3,
        ),
    )

    print(
        "\nSEMANTIC GATE:"
    )

    print(
        "use_semantic_query:",
        decision.use_semantic_query,
    )

    print(
        "reason:",
        decision.reason,
    )

    print(
        "semantic_query:",
        decision.semantic_query,
    )

    print(
        "diagnostic_new_terms:",
        query_novelty(
            query,
            decision.semantic_query,
        ),
    )

    # =========================================================
    # Generic diagnostics
    # =========================================================

    existing_queries = []

    if retrieval_control is not None:
        existing_queries = list(
            getattr(
                retrieval_control,
                "queries",
                [],
            )
            or []
        )

    if not existing_queries:
        existing_queries = list(
            existing.get(
                "generated_queries",
                [],
            )
            or []
        )

    print(
        "\nRETRIEVAL INPUT COMPARISON:"
    )

    print(
        "existing_queries:"
    )

    for index, item in enumerate(
        existing_queries,
        start=1,
    ):
        print(
            f"  {index}. {item}"
        )

    print(
        "\nllm_semantic_query:"
    )

    print(
        " ",
        decision.semantic_query
        or "<REJECTED>",
    )

    # =========================================================
    # Conservative safety invariant
    # =========================================================

    if (
        understanding.confidence <= 0.0
        or understanding.intent.casefold()
        == "unknown"
    ):
        expected_gate = False

        print(
            "\nSAFETY CHECK:"
        )
        print(
            "unknown/zero-confidence semantic result "
            "should not control retrieval."
        )
        print(
            "expected_use_semantic:",
            expected_gate,
        )
        print(
            "actual_use_semantic:",
            decision.use_semantic_query,
        )

        if (
            decision.use_semantic_query
            != expected_gate
        ):
            print(
                "STATUS: FAIL"
            )
        else:
            print(
                "STATUS: PASS"
            )

    else:
        print(
            "\nSAFETY CHECK:"
        )
        print(
            "STATUS: REVIEW"
        )
        print(
            "Reason: semantic result is trusted, "
            "so we inspect whether it adds useful meaning."
        )


def main():
    total_started = time.perf_counter()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_lines = []

    import contextlib
    import io

    buffer = io.StringIO()

    with contextlib.redirect_stdout(
        buffer
    ):
        for case in TEST_CASES:
            run_case(case)

    output = buffer.getvalue()

    print(output)

    OUTPUT_PATH.write_text(
        output,
        encoding="utf-8",
    )

    total_latency = (
        time.perf_counter()
        - total_started
    )

    print()
    print("=" * 100)
    print(
        "TOTAL RUNTIME:",
        round(
            total_latency,
            3,
        ),
        "seconds",
    )
    print(
        "OUTPUT:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
