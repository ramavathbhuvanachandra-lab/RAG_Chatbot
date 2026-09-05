"""
Phase 4 — Claim-Level Context Filter

Purpose
-------
Reduce mixed retrieved chunks into claim-compatible evidence units
before the final answer model sees them.

Important invariants
--------------------
- Deterministic.
- No additional LLM call.
- Never invents content.
- Preserves source metadata.
- Does not split common academic abbreviations such as:
      B.Tech.
      B.S.
      M.Tech.
      Ph.D.
      M.Sc.
- Removes a document only when no compatible evidence unit remains.
"""

import re
from typing import List

from langchain_core.documents import Document

from backend.claim_scope import has_scope_conflict


# =========================================================
# Configuration
# =========================================================

MIN_UNIT_CHARACTERS = 20

PROTECTED_ABBREVIATIONS = {
    "B.Tech.",
    "B.Tech",
    "B.S.",
    "B.S",
    "M.Tech.",
    "M.Tech",
    "M.S.",
    "M.S",
    "M.Sc.",
    "M.Sc",
    "Ph.D.",
    "Ph.D",
    "M.B.B.S.",
    "M.B.B.S",
    "B.V.Sc.",
    "B.V.Sc",
    "B.D.S.",
    "B.D.S",
}


# =========================================================
# Abbreviation protection
# =========================================================

def _protect_abbreviations(
    text: str,
):
    """
    Temporarily replace periods in known abbreviations so they do not
    become sentence boundaries.
    """

    protected = text

    for abbreviation in sorted(
        PROTECTED_ABBREVIATIONS,
        key=len,
        reverse=True,
    ):
        replacement = (
            abbreviation
            .replace(
                ".",
                "<DOT>",
            )
        )

        protected = protected.replace(
            abbreviation,
            replacement,
        )

    return protected


def _restore_abbreviations(
    text: str,
):
    return text.replace(
        "<DOT>",
        ".",
    )


# =========================================================
# Evidence-unit splitting
# =========================================================

def split_evidence_units(
    text: str,
) -> List[str]:
    """
    Split a retrieved chunk into semantic evidence units.

    The splitter prefers:
        1. explicit line boundaries
        2. sentence boundaries

    Academic abbreviations are protected before sentence splitting.
    """

    if not text:
        return []

    normalized = (
        str(text)
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
    )

    normalized = re.sub(
        r"[ \t]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\n{2,}",
        "\n",
        normalized,
    )

    lines = [
        line.strip()
        for line in normalized.split(
            "\n"
        )
        if line.strip()
    ]

    units = []

    for line in lines:

        protected = _protect_abbreviations(
            line
        )

        parts = re.split(
            r"(?<=[.!?])\s+",
            protected,
        )

        for part in parts:

            cleaned = _restore_abbreviations(
                part.strip()
            )

            cleaned = re.sub(
                r"^[\-\*\u2022]+\s*",
                "",
                cleaned,
            ).strip()

            if (
                len(cleaned)
                < MIN_UNIT_CHARACTERS
            ):
                continue

            units.append(
                cleaned
            )

    return units


# =========================================================
# Claim-unit compatibility
# =========================================================

def is_claim_unit_compatible(
    query: str,
    unit: str,
) -> bool:
    """
    Return True when the unit has no explicit claim-scope conflict.
    """

    return not has_scope_conflict(
        query,
        unit,
    )


# =========================================================
# Single-document filtering
# =========================================================

def filter_document_claim_units(
    query: str,
    document,
):
    """
    Keep only claim-compatible evidence units from one document.

    Returns None when no compatible unit remains.
    """

    units = split_evidence_units(
        document.page_content
    )

    if not units:
        return None

    compatible_units = [
        unit
        for unit in units
        if is_claim_unit_compatible(
            query=query,
            unit=unit,
        )
    ]

    if not compatible_units:
        return None

    filtered_text = "\n".join(
        compatible_units
    )

    return Document(
        page_content=filtered_text,
        metadata=dict(
            document.metadata
        ),
    )


# =========================================================
# Collection filtering
# =========================================================

def filter_claim_context(
    query: str,
    documents,
):
    """
    Apply claim-level filtering while preserving document order.
    """

    filtered_documents = []

    for document in documents:

        filtered_document = (
            filter_document_claim_units(
                query=query,
                document=document,
            )
        )

        if filtered_document is None:
            continue

        filtered_documents.append(
            filtered_document
        )

    return filtered_documents


# =========================================================
# Diagnostics
# =========================================================

def filter_claim_context_with_stats(
    query: str,
    documents,
):
    """
    Return filtered documents plus deterministic statistics.
    """

    filtered_documents = (
        filter_claim_context(
            query=query,
            documents=documents,
        )
    )

    return {
        "documents": filtered_documents,
        "input_documents": len(
            documents
        ),
        "output_documents": len(
            filtered_documents
        ),
        "removed_documents": (
            len(documents)
            - len(filtered_documents)
        ),
    }
