"""
Phase 4 — Final Evidence Scope Filter

Purpose
-------
Filter final ranked evidence before evidence assessment and answer
generation so that documents with explicit claim-scope conflicts do not
reach the answer model.

Important invariants
--------------------
- Deterministic.
- No additional LLM call.
- Never invents evidence.
- Removes only explicitly incompatible evidence.
- Preserves document ordering from the evidence-group ranking.
- Does not replace the entire evidence set when filtering removes
  nothing.
- If filtering removes everything, returns an empty list rather than
  silently reintroducing conflicting evidence.
"""

from typing import List

from backend.claim_scope import (
    has_scope_conflict,
)


def filter_final_evidence_scope(
    query: str,
    documents,
) -> List:
    """
    Remove documents that explicitly conflict with the requested claim
    scope.

    Examples
    --------
    Query:
        Can someone with a four-year bachelor's degree apply for
        regular Ph.D. admission?

    Kept:
        Direct admission eligibility evidence.

    Removed:
        Financial-assistance-only evidence.
        Explicit school-specific evidence.
        Explicit part-time/sponsored-only evidence.
    """

    filtered_documents = []

    for document in documents:

        if has_scope_conflict(
            query,
            document.page_content,
        ):
            continue

        filtered_documents.append(
            document
        )

    return filtered_documents


def filter_final_evidence_scope_with_stats(
    query: str,
    documents,
):
    """
    Filter evidence and return deterministic diagnostics.

    Returns
    -------
    {
        "documents": [...],
        "input_count": int,
        "kept_count": int,
        "removed_count": int,
    }
    """

    filtered_documents = (
        filter_final_evidence_scope(
            query=query,
            documents=documents,
        )
    )

    return {
        "documents": filtered_documents,
        "input_count": len(
            documents
        ),
        "kept_count": len(
            filtered_documents
        ),
        "removed_count": (
            len(documents)
            - len(filtered_documents)
        ),
    }
