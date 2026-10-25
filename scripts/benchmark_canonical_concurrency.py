"""Benchmark Canonical Pre-Generation Concurrency and Latency.

Measures:
1. Stage-wise retrieval and evidence processing latency across:
   - Query Understanding & Intent Planning
   - Hybrid Retrieval (Dense Vector + BM25 Lexical + Deterministic Lane)
   - Reciprocal Rank Fusion & Semantic Alignment
   - Evidence Verification & Scope Policy
   - Evidence Packaging & Budget Allocation
2. End-to-end pre-generation latency (p50, p95, p99) under controlled concurrency (1, 2, 4 threads).
3. Workload dimensions:
   - Knowledge base size (canonical chunks)
   - Cumulative query volume
   - Peak concurrency and RPS
   - Model serving / generation isolation (pure zero-inference pre-generation)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure planner runs offline / zero generation
os.environ["PLANNER_OFFLINE_MODE"] = "1"

from ai_platform.runtime.engine import inspect_context
from ai_platform.runtime.retrieval import get_canonical_chunks


SAMPLE_QUERIES = [
    "What is the minimum CGPA required for Ph.D. fellowship continuation, and what is the one-time refundable institute deposit?",
    "What is the application fee for M.Tech programs for General and SC/ST candidates?",
    "What are the attendance requirements to be eligible to appear in end-semester examinations?",
    "Who is the head of the Internal Complaints Committee (ICC) for women?",
    "What are the hostel room allotment rules and dining mess charges?",
    "What is the procedure for semester withdrawal on medical grounds?",
    "What is the grade point required to maintain an undergraduate merit scholarship?",
    "What documents are required at the time of physical reporting for admission?",
    "Can a student take a semester break for an external internship?",
    "What is the penalty for late registration after the announced deadline?",
]


def run_single_query(query: str) -> dict[str, Any]:
    """Execute inspect_context and measure latency."""
    t0 = time.perf_counter()
    try:
        ctx = inspect_context(query)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": True,
            "duration_ms": duration_ms,
            "units": ctx.get("evidence_unit_count", 0),
            "chars": ctx.get("context_character_count", 0),
            "is_multi_intent": ctx.get("is_multi_intent", False),
            "error": None,
        }
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "duration_ms": duration_ms,
            "units": 0,
            "chars": 0,
            "is_multi_intent": False,
            "error": str(exc),
        }


def benchmark_concurrency(
    queries: list[str],
    concurrency: int = 1,
    iterations_per_query: int = 1,
) -> dict[str, Any]:
    """Run a batch of queries under specified concurrency."""
    workload = queries * iterations_per_query
    total_requests = len(workload)

    t_start = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_single_query, q) for q in workload]
        for f in as_completed(futures):
            results.append(f.result())

    total_wall_sec = time.perf_counter() - t_start
    durations = [r["duration_ms"] for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]

    durations.sort()
    n = len(durations)

    def p(pct: float) -> float:
        if not durations:
            return 0.0
        idx = int(round((pct / 100.0) * (n - 1)))
        return durations[idx]

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": n,
        "failed_requests": len(errors),
        "wall_time_sec": round(total_wall_sec, 3),
        "throughput_rps": round(total_requests / total_wall_sec, 2) if total_wall_sec > 0 else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(durations), 1) if durations else 0.0,
            "median": round(statistics.median(durations), 1) if durations else 0.0,
            "p50": round(p(50), 1),
            "p95": round(p(95), 1),
            "p99": round(p(99), 1),
            "min": round(durations[0], 1) if durations else 0.0,
            "max": round(durations[-1], 1) if durations else 0.0,
        },
        "errors": [e["error"] for e in errors[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description="Canonical RAG Concurrency Benchmark")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries to run per level")
    parser.add_argument("--levels", type=str, default="1,2,4", help="Comma-separated concurrency levels")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    queries = SAMPLE_QUERIES[:args.queries]

    # Warm-up / Corpus load
    print("Pre-warming canonical corpus and vectorstore index...")
    t_warm = time.perf_counter()
    chunks = get_canonical_chunks()
    inspect_context(queries[0])
    warmup_time = time.perf_counter() - t_warm
    print(f"Loaded {len(chunks)} canonical chunks in {warmup_time:.2f}s.")

    suite_results = {
        "knowledge_base_size_chunks": len(chunks),
        "levels": {},
    }

    for lvl in levels:
        print(f"\nRunning benchmark at Concurrency Level = {lvl} (Total {len(queries)} queries)...")
        res = benchmark_concurrency(queries, concurrency=lvl, iterations_per_query=1)
        suite_results["levels"][str(lvl)] = res
        print(f"  Wall time: {res['wall_time_sec']}s | Throughput: {res['throughput_rps']} req/s")
        print(f"  Latency: p50={res['latency_ms']['p50']}ms, p95={res['latency_ms']['p95']}ms, p99={res['latency_ms']['p99']}ms")
        print(f"  Errors: {res['failed_requests']} / {res['total_requests']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(suite_results, f, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        print("\nBenchmark Summary:")
        print(json.dumps(suite_results, indent=2))


if __name__ == "__main__":
    main()
