"""Phase 2 — Direct LangChain Ollama smoke test.

Purpose:
    Isolate LangChain ChatOllama from the query-understanding pipeline.

Expected runtime:
    Normally a few seconds to tens of seconds on the current Mac.

Hard stop:
    If this exceeds 60 seconds, press Ctrl+C.
"""

from __future__ import annotations

import time

from backend.llm import query_llm


def main() -> None:
    print("=" * 100)
    print("PHASE 2 — DIRECT LANGCHAIN OLLAMA SMOKE TEST")
    print("=" * 100)
    print()

    print("Model:", query_llm.model)
    print("Reasoning:", query_llm.reasoning)
    print("Context:", query_llm.num_ctx)
    print()

    prompt = "Reply with exactly: OK"

    print("Prompt:")
    print(prompt)
    print()
    print("Calling ChatOllama...")
    print("Expected runtime: a few seconds to tens of seconds.")
    print("STOP if this exceeds 60 seconds.")
    print()

    start = time.perf_counter()

    try:
        response = query_llm.invoke(prompt)

        elapsed = time.perf_counter() - start

        print("-" * 100)
        print("RAW RESPONSE")
        print("-" * 100)
        print()
        print(response)
        print()
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print()

        if elapsed > 60:
            print("FAIL: ChatOllama exceeded the 60-second limit.")
        elif elapsed > 30:
            print("WARNING: ChatOllama is slower than expected.")
        else:
            print("PASS: ChatOllama completed normally.")

    except Exception as exc:
        elapsed = time.perf_counter() - start

        print("-" * 100)
        print("EXCEPTION")
        print("-" * 100)
        print()
        print(type(exc).__name__)
        print(str(exc))
        print()
        print(f"Elapsed time before exception: {elapsed:.2f} seconds")
        print()
        print("FAIL: ChatOllama invocation failed.")


if __name__ == "__main__":
    main()