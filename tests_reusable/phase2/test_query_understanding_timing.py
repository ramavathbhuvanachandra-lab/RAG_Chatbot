"""Phase 2 — Query understanding latency smoke test.

This test measures one semantic query-understanding call.

Expected runtime:
    Usually around 5–15 seconds on the current Mac.

Stop condition:
    If the test runs for more than 60 seconds, press Ctrl+C.
"""

from __future__ import annotations

import time

from backend.core.query_understanding import understand_query


QUERY = "Bhai agar SC ya ST ka student hai toh tuition fee deni padti hai kya?"


def main() -> None:
    print("=" * 100)
    print("PHASE 2 — QUERY UNDERSTANDING — LATENCY SMOKE TEST")
    print("=" * 100)
    print()
    print("Query:")
    print(QUERY)
    print()
    print("Calling query LLM...")
    print("Expected runtime: approximately 5–15 seconds.")
    print("Stop condition: if this exceeds 60 seconds, press Ctrl+C.")
    print()

    start = time.perf_counter()

    result = understand_query(QUERY)

    elapsed = time.perf_counter() - start

    print("-" * 100)
    print("RESULT")
    print("-" * 100)
    print()
    print("Search query:")
    print(result.search_query)
    print()
    print("Intent:")
    print(result.intent)
    print()
    print("List question:")
    print(result.is_list_question)
    print()
    print("Confidence:")
    print(result.confidence)
    print()
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print()

    if elapsed > 60:
        print("WARNING: Query understanding exceeded the 60-second limit.")
    elif elapsed > 30:
        print("WARNING: Query understanding is slower than expected.")
    else:
        print("PASS: Query understanding completed within the expected range.")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()