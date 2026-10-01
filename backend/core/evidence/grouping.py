"""Generic evidence grouping and conservative local-context expansion.

This module turns ranked retrieval anchors into coherent evidence groups.
It intentionally contains no institution-specific vocabulary and no LLM
calls.  The canonical chunk sequence is supplied explicitly by the caller;
there is no hidden global corpus dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Sequence


DEFAULT_MAX_PREVIOUS = 1
DEFAULT_MAX_NEXT = 1
DEFAULT_MIN_CONTEXT_SCORE = 0.18
DEFAULT_MAX_CONTEXT_PER_GROUP = 2
DEFAULT_MAX_CONTEXT_DOCUMENTS = 1_000

_NOISE_MARKERS = (
    "retrieval representation",
    "original source urls",
    "chunk id",
    "rrf score",
    "retrieval rank",
    "source path",
)

_SECTION_START_RE = re.compile(r"^(?:\d+(?:\.\d+)+\s+|#{1,6}\s+)")
_NAMED_SECTION_RE = re.compile(
    r"^(?:section|chapter|appendix|part|unit|module)\s+[A-Za-z0-9]",
    flags=re.IGNORECASE,
)
_INTERNAL_SECTION_RE = re.compile(
    r"(?<!\w)\d+(?:\.\d+)+\s+"
)
_INTERNAL_NAMED_SECTION_RE = re.compile(
    r"(?<!\w)(?:section|chapter|appendix|part|unit|module)\s+[A-Za-z0-9]",
    flags=re.IGNORECASE,
)
_LIST_START_RE = re.compile(r"^(?:[-*•]|\(?[A-Za-z0-9]+\))\s+")


def _text(document: Any) -> str:
    value = getattr(document, "page_content", None)
    if value is None and isinstance(document, dict):
        value = document.get("page_content")
    return str(value or "")


def _metadata(document: Any) -> dict[str, Any]:
    value = getattr(document, "metadata", None)
    if value is None and isinstance(document, dict):
        value = document.get("metadata")
    return dict(value or {})


def _source(document: Any) -> str:
    metadata = _metadata(document)
    for key in ("source", "url", "source_url", "path"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return ""


def _normalize(text: str) -> str:
    value = str(text or "").casefold()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b\w+\b", _normalize(text), flags=re.UNICODE)
        if len(token) > 2
    }


def _content_key(document: Any) -> tuple[str, str]:
    return _source(document), _normalize(_text(document))


def _same_document(first: Any, second: Any) -> bool:
    return _content_key(first) == _content_key(second)


def _is_noise(document: Any) -> bool:
    content = _normalize(_text(document))
    if not content:
        return True
    if any(marker in content for marker in _NOISE_MARKERS):
        return True
    if content.count("http") >= 3:
        return True
    return len(_tokens(content)) < 8


def _is_section_heading(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return False
    prefix = cleaned[:300]
    if _SECTION_START_RE.match(prefix) or _NAMED_SECTION_RE.match(prefix):
        return True
    letters = [ch for ch in cleaned if ch.isalpha()]
    if letters and cleaned.split().__len__() <= 12:
        uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if uppercase_ratio >= 0.90:
            return True
    return False


def _safe_neighbor_text(text: str, *, min_prefix_chars: int = 80) -> str | None:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned or _is_section_heading(cleaned):
        return None

    positions: list[int] = []
    for pattern in (_INTERNAL_SECTION_RE, _INTERNAL_NAMED_SECTION_RE):
        match = pattern.search(cleaned)
        if match and match.start() >= min_prefix_chars:
            positions.append(match.start())

    if positions:
        cleaned = cleaned[: min(positions)].strip()
        if len(cleaned) < min_prefix_chars:
            return None
    return cleaned


def _continuity_score(anchor: Any, neighbor_text: str) -> float:
    anchor_text = _text(anchor)
    anchor_tokens = _tokens(anchor_text)
    neighbor_tokens = _tokens(neighbor_text)
    if not anchor_tokens or not neighbor_tokens:
        return 0.0

    score = 0.0

    anchor_tail = set(list(anchor_tokens)[-40:])
    neighbor_head = set(list(neighbor_tokens)[:40])
    score += 0.45 * (len(anchor_tail & neighbor_head) / max(len(anchor_tail), 1))

    score += 0.20 * (len(anchor_tokens & neighbor_tokens) / max(len(anchor_tokens), 1))

    if _LIST_START_RE.match(neighbor_text):
        score += 0.08

    if anchor_text.rstrip().endswith((",", ":", ";")):
        score += 0.12

    lower = _normalize(neighbor_text)
    if lower.startswith(
        (
            "the applicant ",
            "the applicants ",
            "candidate ",
            "candidates ",
            "students ",
            "student ",
            "the department ",
            "the institute ",
            "the program ",
            "the programme ",
        )
    ):
        score += 0.05

    return min(score, 1.0)


def _sequence_index(anchor: Any, chunks: Sequence[Any]) -> int | None:
    metadata = _metadata(anchor)
    for key in ("chunk_index", "chunk_idx", "index"):
        value = metadata.get(key)
        if isinstance(value, int):
            source = _source(anchor)
            candidates = [
                i for i, chunk in enumerate(chunks)
                if _source(chunk) == source
            ]
            if 0 <= value < len(candidates):
                return candidates[value]

    anchor_key = _content_key(anchor)
    for i, chunk in enumerate(chunks):
        if _content_key(chunk) == anchor_key:
            return i
    return None


@dataclass(slots=True)
class EvidenceGroup:
    """One ranked anchor plus zero or more coherent local-context chunks."""

    anchor: Any
    documents: list[Any] = field(default_factory=list)
    original_rank: int = 0
    coherence_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.documents:
            self.documents = [self.anchor]
        elif not _same_document(self.documents[0], self.anchor):
            self.documents.insert(0, self.anchor)

    @property
    def source(self) -> str:
        return _source(self.anchor)


def _dedupe_documents(documents: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        key = _content_key(document)
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def build_evidence_groups(anchors: Sequence[Any]) -> list[EvidenceGroup]:
    """Create exactly one group per anchor, preserving ranked order."""
    groups: list[EvidenceGroup] = []
    seen_anchors: set[tuple[str, str]] = set()
    for rank, anchor in enumerate(anchors or (), start=1):
        key = _content_key(anchor)
        if key in seen_anchors:
            continue
        seen_anchors.add(key)
        groups.append(EvidenceGroup(anchor=anchor, original_rank=rank))
    return groups


def expand_group_context(
    groups: Sequence[EvidenceGroup],
    canonical_chunks: Sequence[Any],
    *,
    max_previous: int = DEFAULT_MAX_PREVIOUS,
    max_next: int = DEFAULT_MAX_NEXT,
    min_context_score: float = DEFAULT_MIN_CONTEXT_SCORE,
    max_context_per_group: int = DEFAULT_MAX_CONTEXT_PER_GROUP,
) -> list[EvidenceGroup]:
    """Attach conservative same-source neighboring chunks to each group.

    The anchor always remains first. At most ``max_previous`` and
    ``max_next`` neighbors are considered. Structural headings, obvious
    metadata/noise, cross-source candidates, and weak continuations are
    rejected.
    """
    if not groups or not canonical_chunks:
        return list(groups or [])

    result: list[EvidenceGroup] = []

    for group in groups:
        anchor = group.anchor
        index = _sequence_index(anchor, canonical_chunks)
        if index is None:
            result.append(group)
            continue

        candidates: list[tuple[int, Any, float]] = []
        source = _source(anchor)

        offsets: list[int] = []
        for step in range(1, max(0, max_previous) + 1):
            offsets.append(-step)
        for step in range(1, max(0, max_next) + 1):
            offsets.append(step)

        for offset in offsets:
            neighbor_index = index + offset
            if not (0 <= neighbor_index < len(canonical_chunks)):
                continue
            neighbor = canonical_chunks[neighbor_index]
            if _source(neighbor) != source or _same_document(anchor, neighbor):
                continue
            if _is_noise(neighbor):
                continue
            safe_text = _safe_neighbor_text(_text(neighbor))
            if not safe_text:
                continue
            score = _continuity_score(anchor, safe_text)
            if score < min_context_score:
                continue
            candidates.append((abs(offset), neighbor, score))

        candidates.sort(key=lambda item: (-item[2], item[0]))
        selected = [item[1] for item in candidates[: max(0, max_context_per_group)]]

        group.documents = _dedupe_documents([anchor, *selected])
        if selected:
            group.coherence_score = max(item[2] for item in candidates[: len(selected)])
        else:
            group.coherence_score = 1.0
        result.append(group)

    return result


def flatten_evidence_groups(
    groups: Sequence[EvidenceGroup],
    *,
    max_documents: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
) -> list[Any]:
    """Flatten groups in group order while removing exact duplicate content."""
    if max_documents <= 0:
        return []

    result: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for group in groups or ():
        for document in group.documents:
            key = _content_key(document)
            if key in seen:
                continue
            seen.add(key)
            result.append(document)
            if len(result) >= max_documents:
                return result
    return result


__all__ = [
    "EvidenceGroup",
    "build_evidence_groups",
    "expand_group_context",
    "flatten_evidence_groups",
]
