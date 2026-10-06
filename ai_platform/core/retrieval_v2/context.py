"""
Retrieval V2 — Context Builder

Turns clean EvidenceUnits into a compact context for the final LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import EvidenceSet, QuerySpec


@dataclass(frozen=True)
class ContextBlock:
    text: str
    evidence_ids: tuple[str, ...]


def build_context(
    query: QuerySpec,
    evidence: EvidenceSet,
    *,
    max_characters: int = 16000,
) -> ContextBlock:
    """
    Build a deterministic context block.

    The final LLM receives only selected evidence, never the raw
    retrieval pool.
    """

    if evidence.empty:
        return ContextBlock(
            text="No sufficiently relevant evidence was retrieved.",
            evidence_ids=(),
        )

    sections: list[str] = []
    evidence_ids: list[str] = []

    used_characters = 0

    for index, item in enumerate(evidence.items, start=1):
        block = _format_evidence(index, item)

        if used_characters + len(block) > max_characters:
            break

        sections.append(block)
        evidence_ids.append(item.document_id)

        used_characters += len(block)

    context = "\n\n".join(sections)

    return ContextBlock(
        text=context,
        evidence_ids=tuple(evidence_ids),
    )


def _format_evidence(index: int, item) -> str:
    source = item.source or "unknown source"
    title = item.title or "untitled"

    return (
        f"[Evidence {index}]\n"
        f"Source: {source}\n"
        f"Title: {title}\n"
        f"Content:\n{item.text}"
    )