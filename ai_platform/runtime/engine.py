"""End-to-end standalone execution entry point.

This is the runtime composition root: UI or external callers can invoke
``ask`` without importing any legacy application module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from ai_platform.core.graph.graph import build_graph


@lru_cache(maxsize=1)
def get_app_graph():
    """Create and cache the standalone production graph."""
    return build_graph()


def ask(
    question: str,
    *,
    chat_history: Sequence[Any] | None = None,
) -> str:
    """Run one question through the complete reusable RAG pipeline."""
    clean = " ".join(str(question or "").strip().split())
    if not clean:
        raise ValueError("question cannot be empty")

    state = {
        "question": clean,
        "chat_history": tuple(chat_history or ()),
    }
    result = get_app_graph().invoke(state)
    answer = result.get("answer")
    if answer is None:
        raise RuntimeError("RAG graph completed without an answer")
    return str(answer).strip()


def ask_with_diagnostics(
    question: str,
    *,
    chat_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run one question and return the complete graph state for diagnostics."""
    clean = " ".join(str(question or "").strip().split())
    if not clean:
        raise ValueError("question cannot be empty")
    return dict(
        get_app_graph().invoke(
            {
                "question": clean,
                "chat_history": tuple(chat_history or ()),
            }
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the standalone AI platform RAG engine.")
    parser.add_argument("question", nargs="+", help="Question to ask")
    args = parser.parse_args()
    print(ask(" ".join(args.question)))


__all__ = ["get_app_graph", "ask", "ask_with_diagnostics"]
