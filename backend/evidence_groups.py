"""
IIT Jodhpur V1 — Evidence Groups

Purpose
-------
Represent a retrieved anchor together with its locally expanded
context as one coherent evidence unit.

Design principles
-----------------
- Deterministic only.
- No LLM calls.
- No changes to the existing document reranker.
- Each initial retrieval anchor creates one evidence group.
- The anchor always remains part of its group.
- Local neighbors remain attached to their anchor.
- Groups are scored as a whole.
- Production logic remains college-agnostic.
"""

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from backend.retriever import (
    get_source,
    normalize_text,
    detect_programs,
    detect_topics,
    detect_entities,
    score_document_relevance,
)


# =========================================================
# Data Model
# =========================================================

@dataclass
class EvidenceGroup:
    """
    One coherent evidence unit.

    The first document is always the original retrieval anchor.
    Remaining documents are optional local-context additions.
    """

    anchor: Document
    documents: List[Document]
    original_rank: int
    score: float = 0.0

    @property
    def source(self) -> str:
        return get_source(
            self.anchor
        )


# =========================================================
# Group Construction
# =========================================================

def _same_document(
    first: Document,
    second: Document,
) -> bool:
    """
    Compare documents by source and normalized content.
    """

    return (
        get_source(first)
        == get_source(second)
        and
        normalize_text(
            first.page_content
        )
        ==
        normalize_text(
            second.page_content
        )
    )


def _deduplicate_group_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Remove duplicate documents while preserving order.
    """

    result = []

    seen = set()

    for document in documents:

        key = (
            get_source(document),
            normalize_text(
                document.page_content
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            document
        )

    return result


def build_evidence_groups(
    anchors: List[Document],
) -> List[EvidenceGroup]:
    """
    Build one evidence group per initial retrieval anchor.

    The anchor is always first.
    """

    groups = []

    for rank, anchor in enumerate(
        anchors,
        start=1,
    ):

        groups.append(
            EvidenceGroup(
                anchor=anchor,
                documents=[
                    anchor
                ],
                original_rank=rank,
            )
        )

    return groups


def attach_local_context(
    groups: List[EvidenceGroup],
    expanded_documents: List[Document],
) -> List[EvidenceGroup]:
    """
    Attach expanded local-context documents back to their original
    anchors.

    A document is attached to the group whose anchor shares the same
    source and whose content is represented by the expansion result.

    Since local context expansion is generated directly from the
    initial anchors, source-aware sequential matching is sufficient
    here without introducing IDs into the ingestion pipeline.
    """

    if not groups:
        return []

    # Build lookup from original anchor source/content.
    anchor_positions = []

    for group in groups:

        anchor_positions.append(
            (
                group,
                get_source(
                    group.anchor
                ),
                normalize_text(
                    group.anchor.page_content
                ),
            )
        )

    for document in expanded_documents:

        source = get_source(
            document
        )

        normalized_content = normalize_text(
            document.page_content
        )

        attached = False

        # -----------------------------------------------------
        # Preserve original anchor membership first.
        # -----------------------------------------------------

        for group, anchor_source, anchor_content in anchor_positions:

            if (
                source
                != anchor_source
            ):
                continue

            if (
                normalized_content
                == anchor_content
            ):
                attached = True
                break

        if attached:
            continue

        # -----------------------------------------------------
        # Otherwise attach the local document to the most likely
        # same-source anchor.
        #
        # Local expansion only considers immediate neighbors,
        # so strongest lexical/topic/entity continuity is enough.
        # -----------------------------------------------------

        best_group = None
        best_score = -1.0

        for group, anchor_source, anchor_content in anchor_positions:

            if (
                source
                != anchor_source
            ):
                continue

            anchor = group.anchor

            anchor_tokens = {
                token
                for token in normalize_text(
                    anchor.page_content
                ).split()
                if len(token) > 2
            }

            document_tokens = {
                token
                for token in normalize_text(
                    document.page_content
                ).split()
                if len(token) > 2
            }

            if not anchor_tokens or not document_tokens:
                continue

            overlap = (
                len(
                    anchor_tokens
                    & document_tokens
                )
                / max(
                    len(anchor_tokens),
                    1,
                )
            )

            score = overlap

            if (
                detect_programs(
                    anchor.page_content
                )
                &
                detect_programs(
                    document.page_content
                )
            ):
                score += 0.15

            if (
                detect_topics(
                    anchor.page_content
                )
                &
                detect_topics(
                    document.page_content
                )
            ):
                score += 0.15

            if (
                detect_entities(
                    anchor.page_content
                )
                &
                detect_entities(
                    document.page_content
                )
            ):
                score += 0.10

            if score > best_score:
                best_score = score
                best_group = group

        if best_group is not None:

            best_group.documents.append(
                document
            )

    # ---------------------------------------------------------
    # Final per-group deduplication.
    # ---------------------------------------------------------

    for group in groups:

        group.documents = (
            _deduplicate_group_documents(
                group.documents
            )
        )

    return groups


# =========================================================
# Group Scoring
# =========================================================

def _combined_group_text(
    group: EvidenceGroup,
) -> str:
    """
    Combine group content for scoring only.
    """

    return "\n".join(
        document.page_content
        for document in group.documents
    )


def score_evidence_group(
    query: str,
    group: EvidenceGroup,
) -> float:
    """
    Score the evidence group as one retrieval unit.

    The anchor receives the strongest weight.
    Local context contributes supporting relevance rather than
    replacing the anchor.
    """

    anchor_score = score_document_relevance(
        query=query,
        document=group.anchor,
        original_rank=group.original_rank,
    )

    # ---------------------------------------------------------
    # Context contribution
    # ---------------------------------------------------------

    context_documents = [
        document
        for document in group.documents
        if not _same_document(
            document,
            group.anchor,
        )
    ]

    context_score = 0.0

    for document in context_documents:

        score = score_document_relevance(
            query=query,
            document=document,
            original_rank=group.original_rank,
        )

        context_score += score

    # ---------------------------------------------------------
    # Diminishing returns.
    #
    # The first useful context chunk matters most; additional
    # chunks contribute less.
    # ---------------------------------------------------------

    if context_documents:

        context_score *= (
            0.35
        )

    total_score = (
        anchor_score
        + context_score
    )

    group.score = total_score

    return total_score


def rank_evidence_groups(
    query: str,
    groups: List[EvidenceGroup],
    top_k: int,
) -> List[EvidenceGroup]:
    """
    Score and rank evidence groups.
    """

    for group in groups:

        score_evidence_group(
            query=query,
            group=group,
        )

    groups.sort(
        key=lambda group: (
            group.score,
            -group.original_rank,
        ),
        reverse=True,
    )

    return groups[
        :top_k
    ]


# =========================================================
# Flatten Final Groups
# =========================================================

def flatten_evidence_groups(
    groups: List[EvidenceGroup],
) -> List[Document]:
    """
    Convert the final evidence groups back into Documents for the
    existing evidence and answer stages.

    Group boundaries are preserved by ordering:

        anchor
        local context
        anchor
        local context
        ...
    """

    documents = []

    seen = set()

    for group in groups:

        for document in group.documents:

            key = (
                get_source(document),
                normalize_text(
                    document.page_content
                ),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            documents.append(
                document
            )

    return documents