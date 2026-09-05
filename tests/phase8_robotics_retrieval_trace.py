"""
Phase 8A.2 — Robotics Retrieval Trace

Purpose
-------
Trace the retrieval stages for one known failure without modifying
retrieval behavior.

This is diagnostic only.
"""

from backend.retriever import (
    normalize_text,
    detect_topics,
    detect_entities,
)


QUESTION = (
    "What research areas are related to robotics at IIT Jodhpur?"
)


def main():

    print("=" * 100)
    print("PHASE 8A.2 — ROBOTICS RETRIEVAL TRACE")
    print("=" * 100)

    print("\nQUESTION:")
    print(QUESTION)

    print("\nNORMALIZED:")
https://www.youtube.com/    print(normalize_text(QUESTION))

    print("\nTOPICS:")
    print(detect_topics(QUESTION))

    print("\nENTITIES:")
    print(detect_entities(QUESTION))

    print("=" * 100)
    print(
        "\nNow inspect backend/retriever.py output for the actual "
        "dense/BM25/RRF pipeline."
    )


if __name__ == "__main__":
    main()