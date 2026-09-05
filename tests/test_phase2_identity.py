import argparse
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
    get_document_id,
)


# =========================================================
# Configuration
# =========================================================

TOP_K = 20


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
# Source Helpers
# =========================================================

def raw_source(document):
    return str(
        document.metadata.get(
            "source",
            "",
        )
    ).replace(
        "\\",
        "/",
    )


def canonical_source(document):
    """
    Normalize the source path so equivalent absolute/relative
    representations can be compared.
    """

    return str(
        Path(
            raw_source(document)
        ).resolve()
    ).replace(
        "\\",
        "/",
    )


def content_preview(document):
    """
    Show enough chunk content to determine whether two chunks
    are actually different.
    """

    text = document.page_content.strip()

    if len(text) > 700:
        return text[:700] + " ..."

    return text


# =========================================================
# Document Inspection
# =========================================================

def inspect_documents(
    documents,
    label,
):
    """
    Inspect source identity and chunk identity.
    """

    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    print(
        f"Documents inspected: {len(documents)}"
    )

    source_groups = defaultdict(list)
    id_groups = defaultdict(list)

    for rank, document in enumerate(
        documents,
        start=1,
    ):

        doc_id = get_document_id(
            document
        )

        raw = raw_source(
            document
        )

        canonical = canonical_source(
            document
        )

        source_groups[
            canonical
        ].append(
            document
        )

        id_groups[
            doc_id
        ].append(
            document
        )

        print()
        print("-" * 100)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Raw source:\n{raw}"
        )

        print(
            f"Canonical source:\n{canonical}"
        )

        print(
            f"Document ID:\n{doc_id}"
        )

        print(
            "Chunk:"
        )

        print(
            content_preview(
                document
            )
        )

    # =====================================================
    # Same Source
    # =====================================================

    print()
    print("=" * 100)
    print("SAME-SOURCE GROUPING")
    print("=" * 100)

    repeated_sources = 0

    for source, docs in source_groups.items():

        if len(docs) <= 1:
            continue

        repeated_sources += 1

        print()
        print(
            f"{len(docs)} chunks from:"
        )

        print(
            source
        )

        for index, document in enumerate(
            docs,
            start=1,
        ):

            print(
                f"\n  Chunk {index}:"
            )

            print(
                content_preview(
                    document
                )
            )

    # =====================================================
    # Duplicate IDs
    # =====================================================

    print()
    print("=" * 100)
    print("DUPLICATE DOCUMENT IDS")
    print("=" * 100)

    duplicate_ids = 0

    for doc_id, docs in id_groups.items():

        if len(docs) <= 1:
            continue

        duplicate_ids += 1

        print()
        print(
            f"Duplicate ID appears "
            f"{len(docs)} times:"
        )

        print(
            doc_id
        )

        for index, document in enumerate(
            docs,
            start=1,
        ):

            print(
                f"\n  Duplicate chunk {index}:"
            )

            print(
                content_preview(
                    document
                )
            )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 100)
    print("IDENTITY SUMMARY")
    print("=" * 100)

    print(
        f"Unique canonical sources: "
        f"{len(source_groups)}"
    )

    print(
        f"Repeated sources: "
        f"{repeated_sources}"
    )

    print(
        f"Duplicate document IDs: "
        f"{duplicate_ids}"
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
        help="Regression question ID",
    )

    args = parser.parse_args()

    question = QUESTIONS[
        args.id
    ]

    print()
    print("=" * 100)
    print(
        "IITJ V1 — PHASE 2 DOCUMENT IDENTITY AUDIT"
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
    # Inspect Dense
    # -----------------------------------------------------

    inspect_documents(
        dense_docs[:TOP_K],
        "DENSE DOCUMENT IDENTITY",
    )

    # -----------------------------------------------------
    # Inspect BM25
    # -----------------------------------------------------

    inspect_documents(
        bm25_docs[:TOP_K],
        "BM25 DOCUMENT IDENTITY",
    )

    # -----------------------------------------------------
    # Final diagnostic instructions
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print(
        "PHASE 2 — WHAT WE ARE CHECKING"
    )
    print("=" * 100)

    print(
        "1. Same source + identical content "
        "should have the same identity."
    )

    print(
        "2. Same source + different useful chunks "
        "must remain separate."
    )

    print(
        "3. Relative and absolute source paths "
        "should not create fake identities."
    )

    print(
        "4. Repeated source does NOT automatically "
        "mean duplicate content."
    )

    print()
    print(
        "Do not change production code yet."
    )


if __name__ == "__main__":
    main()