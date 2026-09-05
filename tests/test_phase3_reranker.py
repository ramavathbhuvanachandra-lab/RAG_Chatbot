import argparse
import re
import sys
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
    reciprocal_rank_fusion,
    get_document_id,
    score_document_relevance,
    rerank_documents,
)


# =========================================================
# Configuration
# =========================================================

RRF_K = 60

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

INSPECT_TOP_K = 15
FINAL_TOP_K = 5

PREVIEW = 900


# =========================================================
# Questions
# =========================================================

QUESTIONS = {
    "ee_research":
        "What research areas are available in Electrical Engineering?",

    "btech_fees":
        "What are the fees for B.Tech students?",

    "hostel_facilities":
        "What are the hostel facilities for students?",
}


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
        term
        for term in normalize(
            question
        ).split()
        if len(term) > 2
        and term not in stop_words
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
# Weighted RRF Diagnostic
# =========================================================

def diagnostic_rrf(
    dense_docs,
    bm25_docs,
):
    """
    Reproduce the current production RRF configuration.
    """

    scores = defaultdict(float)
    lookup = {}

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

    for documents, weight in ranked_lists:

        for rank, document in enumerate(
            documents,
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
# RRF Rank Map
# =========================================================

def rrf_rank_map(
    rrf_docs,
):
    return {
        get_document_id(
            item["document"]
        ): {
            "rank": rank,
            "score": item["score"],
        }
        for rank, item in enumerate(
            rrf_docs,
            start=1,
        )
    }


# =========================================================
# Explain Reranker Signals
# =========================================================

def explain_document(
    question,
    document,
):
    """
    Reconstruct the major signals used by the current
    reranker for diagnostics.

    This does NOT change production behavior.
    """

    normalized_query = normalize(
        question
    )

    query_tokens = set(
        query_terms(
            question
        )
    )

    content = normalize(
        document.page_content
    )

    source_text = normalize(
        source(document)
    )

    content_tokens = set(
        content.split()
    )

    source_tokens = set(
        source_text.split()
    )

    matched = (
        query_tokens
        & content_tokens
    )

    content_overlap = 0.0

    if query_tokens:

        content_overlap = (
            len(matched)
            / len(query_tokens)
        )

    source_overlap = 0.0

    if query_tokens:

        source_overlap = (
            len(
                query_tokens
                & source_tokens
            )
            / len(query_tokens)
        )

    program_terms = {
        "btech",
        "mtech",
        "msc",
        "phd",
        "mba",
    }

    query_programs = {
        term
        for term in program_terms
        if term in normalized_query
    }

    document_programs = {
        term
        for term in program_terms
        if term in content
        or term in source_text
    }

    topic_terms = {
        "research",
        "fees",
        "fee",
        "hostel",
        "facilities",
        "admission",
        "eligibility",
        "department",
        "curriculum",
        "mess",
        "dining",
    }

    query_topics = {
        term
        for term in topic_terms
        if term in normalized_query
    }

    document_topics = {
        term
        for term in topic_terms
        if term in content
        or term in source_text
    }

    # Possible content-quality noise.
    url_count = (
        document.page_content.count(
            "http"
        )
    )

    source_word_count = len(
        source_text.split()
    )

    content_word_count = len(
        content_tokens
    )

    return {
        "matched_terms":
            sorted(matched),

        "content_overlap":
            content_overlap,

        "source_overlap":
            source_overlap,

        "query_programs":
            sorted(query_programs),

        "document_programs":
            sorted(document_programs),

        "query_topics":
            sorted(query_topics),

        "document_topics":
            sorted(document_topics),

        "url_count":
            url_count,

        "content_word_count":
            content_word_count,

        "source_word_count":
            source_word_count,
    }


# =========================================================
# Print Candidate
# =========================================================

def print_candidate(
    *,
    question,
    document,
    rank,
    stage,
    score,
    explanation=None,
    rrf_rank=None,
    rrf_score=None,
):
    print()
    print("-" * 105)

    print(
        f"{stage} #{rank}"
    )

    if rrf_rank is not None:
        print(
            f"RRF rank: {rrf_rank}"
        )

    if rrf_score is not None:
        print(
            f"RRF score: {rrf_score:.8f}"
        )

    print(
        f"Reranker score: {score:.6f}"
    )

    print(
        f"Source: {source(document)}"
    )

    if explanation:

        print(
            "Matched terms: "
            + (
                ", ".join(
                    explanation[
                        "matched_terms"
                    ]
                )
                if explanation[
                    "matched_terms"
                ]
                else "NONE"
            )
        )

        print(
            f"Content overlap: "
            f"{explanation['content_overlap']:.2f}"
        )

        print(
            f"Source overlap: "
            f"{explanation['source_overlap']:.2f}"
        )

        print(
            "Query programs: "
            + (
                ", ".join(
                    explanation[
                        "query_programs"
                    ]
                )
                if explanation[
                    "query_programs"
                ]
                else "NONE"
            )
        )

        print(
            "Document programs: "
            + (
                ", ".join(
                    explanation[
                        "document_programs"
                    ]
                )
                if explanation[
                    "document_programs"
                ]
                else "NONE"
            )
        )

        print(
            "Query topics: "
            + (
                ", ".join(
                    explanation[
                        "query_topics"
                    ]
                )
                if explanation[
                    "query_topics"
                ]
                else "NONE"
            )
        )

        print(
            "Document topics: "
            + (
                ", ".join(
                    explanation[
                        "document_topics"
                    ]
                )
                if explanation[
                    "document_topics"
                ]
                else "NONE"
            )
        )

        print(
            f"URL count: "
            f"{explanation['url_count']}"
        )

        print(
            f"Content token count: "
            f"{explanation['content_word_count']}"
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
# Run One Question
# =========================================================

def run_question(
    question,
):
    print()
    print("=" * 110)
    print(
        "IITJ V1 — PHASE 3 RERANKER AUDIT"
    )
    print("=" * 110)

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
    # RRF
    # -----------------------------------------------------

    rrf_docs = diagnostic_rrf(
        dense_docs,
        bm25_docs,
    )

    rrf_map = rrf_rank_map(
        rrf_docs
    )

    fused_documents = [
        item["document"]
        for item in rrf_docs
    ]

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------

    reranked_docs = rerank_documents(
        query=question,
        documents=fused_documents,
        top_k=FINAL_TOP_K,
    )

    # =====================================================
    # RRF candidates BEFORE reranking
    # =====================================================

    print()
    print("=" * 110)
    print(
        "A — TOP RRF CANDIDATES BEFORE RERANKING"
    )
    print("=" * 110)

    for rank, item in enumerate(
        rrf_docs[
            :INSPECT_TOP_K
        ],
        start=1,
    ):

        document = item[
            "document"
        ]

        document_id = get_document_id(
            document
        )

        rerank_score = (
            score_document_relevance(
                query=question,
                document=document,
                original_rank=rank,
            )
        )

        explanation = explain_document(
            question,
            document,
        )

        print_candidate(
            question=question,
            document=document,
            rank=rank,
            stage="RRF",
            score=rerank_score,
            rrf_rank=rank,
            rrf_score=item["score"],
            explanation=explanation,
        )

    # =====================================================
    # Final reranked evidence
    # =====================================================

    print()
    print("=" * 110)
    print(
        "B — FINAL RERANKED EVIDENCE"
    )
    print("=" * 110)

    for final_rank, document in enumerate(
        reranked_docs,
        start=1,
    ):

        document_id = get_document_id(
            document
        )

        rrf_info = rrf_map.get(
            document_id
        )

        if rrf_info is None:
            rrf_rank = None
            rrf_score = None
            original_rank = final_rank
        else:
            rrf_rank = rrf_info[
                "rank"
            ]
            rrf_score = rrf_info[
                "score"
            ]
            original_rank = rrf_rank

        rerank_score = (
            score_document_relevance(
                query=question,
                document=document,
                original_rank=original_rank,
            )
        )

        explanation = explain_document(
            question,
            document,
        )

        print_candidate(
            question=question,
            document=document,
            rank=final_rank,
            stage="FINAL",
            score=rerank_score,
            rrf_rank=rrf_rank,
            rrf_score=rrf_score,
            explanation=explanation,
        )

    # =====================================================
    # Rank Movement
    # =====================================================

    print()
    print("=" * 110)
    print(
        "C — RANK MOVEMENT"
    )
    print("=" * 110)

    for final_rank, document in enumerate(
        reranked_docs,
        start=1,
    ):

        document_id = get_document_id(
            document
        )

        rrf_info = rrf_map.get(
            document_id
        )

        if rrf_info is None:
            continue

        old_rank = rrf_info[
            "rank"
        ]

        delta = (
            old_rank
            - final_rank
        )

        print(
            f"Final #{final_rank} | "
            f"RRF #{old_rank} | "
            f"movement {delta:+d} | "
            f"{source(document)}"
        )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 110)
    print(
        "PHASE 3 QUESTIONS"
    )
    print("=" * 110)

    print(
        "1. Which documents did the reranker promote?"
    )

    print(
        "2. Which documents did it demote?"
    )

    print(
        "3. Did it promote answer-bearing chunks?"
    )

    print(
        "4. Did it promote metadata / URL / navigation chunks?"
    )

    print(
        "5. Did it punish useful chunks because of weak lexical signals?"
    )

    print(
        "6. Is the reranker actually improving over RRF?"
    )

    print()
    print(
        "Do NOT modify the reranker yet."
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        required=True,
        choices=QUESTIONS.keys(),
    )

    args = parser.parse_args()

    run_question(
        QUESTIONS[
            args.id
        ]
    )


if __name__ == "__main__":
    main()