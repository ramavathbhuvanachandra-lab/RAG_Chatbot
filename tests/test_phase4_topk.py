import argparse
import json
import sys
from pathlib import Path


# =========================================================
# Project Path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Imports
# =========================================================

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
)


# =========================================================
# Configuration
# =========================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "tests"
    / "regression_questions.json"
)

TOP_K_VALUES = [3, 5, 7, 10]

PREVIEW = 700


# =========================================================
# Load Questions
# =========================================================

def load_questions():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# =========================================================
# Source
# =========================================================

def source(document):
    return str(
        document.metadata.get(
            "source",
            "UNKNOWN",
        )
    ).replace(
        "\\",
        "/",
    )


# =========================================================
# Print Chunk
# =========================================================

def print_chunk(
    rank,
    document,
):
    print()
    print("-" * 90)

    print(
        f"#{rank}"
    )

    print(
        f"Source:\n{source(document)}"
    )

    text = document.page_content.strip()

    if len(text) > PREVIEW:
        text = text[:PREVIEW] + "\n..."

    print(
        "Chunk:"
    )

    print(text)


# =========================================================
# Run One Question
# =========================================================

def run_question(item):
    question = item["question"]

    print()
    print("=" * 100)
    print(
        "IITJ V1 — PHASE 4 TOP-K EVIDENCE AUDIT"
    )
    print("=" * 100)

    print(
        f"\nQUESTION:\n{question}"
    )

    # -----------------------------------------------------
    # Dense
    # -----------------------------------------------------

    dense_docs = dense_retrieve(
        question
    )

    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    bm25_docs = keyword_retrieve(
        question
    )

    # -----------------------------------------------------
    # Weighted RRF
    # -----------------------------------------------------

    fused_docs = reciprocal_rank_fusion(
        [
            dense_docs,
            bm25_docs,
        ]
    )

    print(
        f"\nTotal fused candidates: "
        f"{len(fused_docs)}"
    )

    # =====================================================
    # Compare Top-K
    # =====================================================

    for top_k in TOP_K_VALUES:

        selected = fused_docs[
            :top_k
        ]

        print()
        print("=" * 100)

        print(
            f"RRF TOP {top_k}"
        )

        print(
            "=" * 100
        )

        print(
            f"Evidence chunks: {len(selected)}"
        )

        unique_sources = len(
            {
                source(document)
                for document in selected
            }
        )

        print(
            f"Unique sources: {unique_sources}"
        )

        for rank, document in enumerate(
            selected,
            start=1,
        ):

            print_chunk(
                rank,
                document,
            )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 100)
    print(
        "PHASE 4 DECISION QUESTIONS"
    )
    print("=" * 100)

    print(
        "1. Does Top 3 contain enough answer-bearing evidence?"
    )

    print(
        "2. Does Top 5 add useful evidence without much noise?"
    )

    print(
        "3. Does Top 7 materially improve coverage?"
    )

    print(
        "4. Does Top 10 introduce unrelated/dirty chunks?"
    )

    print()
    print(
        "Do not modify production code yet."
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        required=True,
        help="Regression question ID",
    )

    args = parser.parse_args()

    questions = load_questions()

    selected = [
        item
        for item in questions
        if item["id"] == args.id
    ]

    if not selected:

        print(
            f"No question found: {args.id}"
        )

        sys.exit(1)

    run_question(
        selected[0]
    )


if __name__ == "__main__":
    main()