import argparse
import json
import re
import sys
import time
from collections import defaultdict
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
    get_document_id,
)


# =========================================================
# Configuration
# =========================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "tests"
    / "regression_questions.json"
)

TOP_K = 10
RRF_K = 60

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

PREVIEW = 900


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
# Text Helpers
# =========================================================

def normalize(text: str) -> str:
    text = str(text).lower()

    replacements = {
        "b.tech.": "btech",
        "b.tech": "btech",
        "m.tech.": "mtech",
        "m.tech": "mtech",
        "m.sc.": "msc",
        "m.sc": "msc",
        "ph.d.": "phd",
        "ph.d": "phd",
        "_": " ",
        "-": " ",
        "/": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def query_terms(question: str):
    stop_words = {
        "what",
        "are",
        "the",
        "is",
        "at",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "a",
        "an",
        "do",
        "does",
        "how",
        "which",
        "who",
        "where",
        "when",
        "why",
        "can",
        "could",
        "would",
        "should",
        "tell",
        "me",
        "about",
        "iit",
        "jodhpur",
    }

    return [
        word
        for word in normalize(
            question
        ).split()
        if len(word) > 2
        and word not in stop_words
    ]


def matched_terms(
    question,
    document,
):
    terms = query_terms(
        question
    )

    content = normalize(
        document.page_content
    )

    return [
        term
        for term in terms
        if term in content
    ]


# =========================================================
# Source Helper
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
# Diagnostic RRF
# =========================================================

def diagnostic_rrf(
    dense_docs,
    bm25_docs,
):
    """
    Weighted RRF used by the Phase 1 experiment.

    Dense = 0.7
    BM25  = 0.3

    This must stay synchronized with the production RRF
    configuration while we complete Phase 1.
    """

    ranked_lists = [
        (
            dense_docs,
            DENSE_WEIGHT,
        ),
        (
            bm25_docs,
            BM25_WEIGHT,
        ),
    ]

    scores = defaultdict(float)
    lookup = {}

    for ranked_documents, weight in ranked_lists:

        for rank, document in enumerate(
            ranked_documents,
            start=1,
        ):

            document_id = get_document_id(
                document
            )

            lookup[
                document_id
            ] = document

            scores[
                document_id
            ] += (
                weight
                / (
                    RRF_K + rank
                )
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            "document": lookup[
                document_id
            ],
            "score": score,
        }
        for document_id, score in ranked
    ]


# =========================================================
# Print Chunk
# =========================================================

def print_chunk(
    question,
    document,
    rank,
    stage,
    score=None,
    dense_rank=None,
    bm25_rank=None,
):
    print()
    print("-" * 100)

    print(
        f"{stage} #{rank}"
    )

    if score is not None:
        print(
            f"RRF score: {score:.8f}"
        )

    if dense_rank is not None:
        print(
            f"Dense rank: {dense_rank}"
        )

    if bm25_rank is not None:
        print(
            f"BM25 rank: {bm25_rank}"
        )

    print(
        f"Source: {source(document)}"
    )

    matches = matched_terms(
        question,
        document,
    )

    print(
        "Matched terms:",
        ", ".join(matches)
        if matches
        else "NONE",
    )

    text = (
        document.page_content
        .strip()
    )

    if len(text) > PREVIEW:
        text = (
            text[:PREVIEW]
            + "\n..."
        )

    print(
        "Chunk:"
    )

    print(text)


# =========================================================
# Run Phase 1
# =========================================================

def run_question(item):
    question = item[
        "question"
    ]

    print()
    print("=" * 100)
    print(
        "IITJ V1 — PHASE 1 RETRIEVAL AUDIT"
    )
    print("=" * 100)

    print(
        f"\nQUESTION:\n{question}"
    )

    print(
        "\nRRF CONFIGURATION:"
    )

    print(
        f"  Dense weight: {DENSE_WEIGHT}"
    )

    print(
        f"  BM25 weight:  {BM25_WEIGHT}"
    )

    # -----------------------------------------------------
    # Dense
    # -----------------------------------------------------

    start = time.perf_counter()

    dense_docs = dense_retrieve(
        question
    )

    dense_time = (
        time.perf_counter()
        - start
    )

    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    start = time.perf_counter()

    bm25_docs = keyword_retrieve(
        question
    )

    bm25_time = (
        time.perf_counter()
        - start
    )

    # -----------------------------------------------------
    # Weighted RRF
    # -----------------------------------------------------

    start = time.perf_counter()

    rrf_docs = diagnostic_rrf(
        dense_docs,
        bm25_docs,
    )

    rrf_time = (
        time.perf_counter()
        - start
    )

    print(
        "\nTIMINGS"
    )

    print(
        f"Dense: {dense_time:.4f}s"
    )

    print(
        f"BM25:  {bm25_time:.4f}s"
    )

    print(
        f"RRF:   {rrf_time:.4f}s"
    )

    # -----------------------------------------------------
    # Rank maps
    # -----------------------------------------------------

    dense_ranks = {
        get_document_id(doc): rank
        for rank, doc in enumerate(
            dense_docs,
            start=1,
        )
    }

    bm25_ranks = {
        get_document_id(doc): rank
        for rank, doc in enumerate(
            bm25_docs,
            start=1,
        )
    }

    # =====================================================
    # Dense
    # =====================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "A — DENSE TOP 10"
    )

    print(
        "=" * 100
    )

    for rank, doc in enumerate(
        dense_docs[:TOP_K],
        start=1,
    ):

        print_chunk(
            question,
            doc,
            rank,
            "Dense",
        )

    # =====================================================
    # BM25
    # =====================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "B — BM25 TOP 10"
    )

    print(
        "=" * 100
    )

    for rank, doc in enumerate(
        bm25_docs[:TOP_K],
        start=1,
    ):

        print_chunk(
            question,
            doc,
            rank,
            "BM25",
        )

    # =====================================================
    # Weighted RRF
    # =====================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "C — WEIGHTED RRF TOP 10"
    )

    print(
        "=" * 100
    )

    for rank, item_data in enumerate(
        rrf_docs[:TOP_K],
        start=1,
    ):

        document = item_data[
            "document"
        ]

        document_id = (
            get_document_id(
                document
            )
        )

        print_chunk(
            question,
            document,
            rank,
            "RRF",
            score=item_data[
                "score"
            ],
            dense_rank=dense_ranks.get(
                document_id
            ),
            bm25_rank=bm25_ranks.get(
                document_id
            ),
        )

    # =====================================================
    # Retention
    # =====================================================

    dense_top_ids = {
        get_document_id(doc)
        for doc in dense_docs[:TOP_K]
    }

    bm25_top_ids = {
        get_document_id(doc)
        for doc in bm25_docs[:TOP_K]
    }

    rrf_top_ids = {
        get_document_id(
            item_data["document"]
        )
        for item_data in rrf_docs[:TOP_K]
    }

    print(
        "\n" + "=" * 100
    )

    print(
        "D — RRF RETENTION"
    )

    print(
        "=" * 100
    )

    print(
        f"Dense top {TOP_K} retained in RRF: "
        f"{len(dense_top_ids & rrf_top_ids)}/{TOP_K}"
    )

    print(
        f"BM25 top {TOP_K} retained in RRF: "
        f"{len(bm25_top_ids & rrf_top_ids)}/{TOP_K}"
    )

    print(
        "\nPhase 1 is still under investigation."
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        help="Regression question ID",
    )

    args = parser.parse_args()

    questions = load_questions()

    if args.id:

        questions = [
            item
            for item in questions
            if item["id"] == args.id
        ]

        if not questions:

            print(
                f"No question found with id: {args.id}"
            )

            sys.exit(1)

    else:

        phase_ids = {
            "ee_research",
            "btech_fees",
            "hostel_facilities",
        }

        questions = [
            item
            for item in questions
            if item["id"] in phase_ids
        ]

    for item in questions:
        run_question(item)


if __name__ == "__main__":
    main()