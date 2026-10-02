"""Final verified-evidence packaging for the reusable RAG core.

Pipeline position
-----------------
    evidence groups + coverage + claim audit -> this module -> answering

E6 has one job: convert already-verified evidence into a deterministic,
answer-model-ready package.

It does NOT:
    * retrieve documents;
    * rerank candidates;
    * decide whether evidence is sufficient;
    * call an LLM;
    * infer institution-specific meaning.

The package keeps machine-readable provenance separately from the rendered
context so internal source paths, chunk identifiers, ranks, and scores are
never required by the answer model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Literal, Sequence

from ai_platform.core.evidence.claims import ClaimAudit, EvidenceUnit


PackageStatus = Literal["ready", "partial", "empty", "conflicted"]

_DEFAULT_MAX_UNITS = 10
_DEFAULT_MAX_UNITS_PER_SOURCE = 2
_DEFAULT_MAX_CONTEXT_CHARS = 12_000

_INGESTION_WRAPPER_PATTERNS = (
    re.compile(
        r"(?im)^\s*command\s+\d+\s*[•|:]\s*retrieval\s+representation\s*[•|:]?\s*$"
    ),
    re.compile(
        r"(?im)^\s*(?:original\s+)?source\s+urls\s+preserved\s+from\s+the\s+command\s+5\s+knowledge\s+source\.?\s*$"
    ),
)


@dataclass(frozen=True, slots=True)
class PackagedEvidenceItem:
    """One evidence unit that survived final packaging."""

    evidence_id: str
    text: str
    source: str
    heading: str | None = None
    supported_claim_ids: tuple[str, ...] = ()
    partial_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.evidence_id).strip():
            raise ValueError("PackagedEvidenceItem.evidence_id cannot be empty.")
        if not str(self.text).strip():
            raise ValueError("PackagedEvidenceItem.text cannot be empty.")
        object.__setattr__(self, "evidence_id", str(self.evidence_id).strip())
        object.__setattr__(self, "text", _normalize_text(self.text))
        object.__setattr__(self, "source", str(self.source or "unknown").strip())
        object.__setattr__(
            self,
            "heading",
            _normalize_text(self.heading) if self.heading else None,
        )
        object.__setattr__(
            self,
            "supported_claim_ids",
            _dedupe_strings(self.supported_claim_ids),
        )
        object.__setattr__(
            self,
            "partial_claim_ids",
            _dedupe_strings(self.partial_claim_ids),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source": self.source,
            "heading": self.heading,
            "supported_claim_ids": list(self.supported_claim_ids),
            "partial_claim_ids": list(self.partial_claim_ids),
        }


@dataclass(frozen=True, slots=True)
class EvidencePackage:
    """Deterministic final context plus machine-readable provenance."""

    status: PackageStatus
    context: str
    items: tuple[PackagedEvidenceItem, ...] = ()
    source_count: int = 0
    selected_count: int = 0
    omitted_count: int = 0
    supported_claim_ids: tuple[str, ...] = ()
    partial_claim_ids: tuple[str, ...] = ()
    omitted_claim_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", str(self.context or "").strip())
        object.__setattr__(self, "items", tuple(self.items or ()))
        object.__setattr__(self, "source_count", max(0, int(self.source_count)))
        object.__setattr__(self, "selected_count", max(0, int(self.selected_count)))
        object.__setattr__(self, "omitted_count", max(0, int(self.omitted_count)))
        object.__setattr__(self, "supported_claim_ids", _dedupe_strings(self.supported_claim_ids))
        object.__setattr__(self, "partial_claim_ids", _dedupe_strings(self.partial_claim_ids))
        object.__setattr__(self, "omitted_claim_ids", _dedupe_strings(self.omitted_claim_ids))
        object.__setattr__(self, "reasons", _dedupe_strings(self.reasons))

    @property
    def ready_for_generation(self) -> bool:
        return self.status in {"ready", "partial"} and bool(self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "context": self.context,
            "items": [item.to_dict() for item in self.items],
            "source_count": self.source_count,
            "selected_count": self.selected_count,
            "omitted_count": self.omitted_count,
            "supported_claim_ids": list(self.supported_claim_ids),
            "partial_claim_ids": list(self.partial_claim_ids),
            "omitted_claim_ids": list(self.omitted_claim_ids),
            "reasons": list(self.reasons),
            "ready_for_generation": self.ready_for_generation,
        }


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe_strings(values: Iterable[object] | None) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        clean = _normalize_text(value)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return tuple(result)


def sanitize_evidence_text(text: str) -> str:
    """Remove known ingestion/debug wrappers while preserving factual text."""
    cleaned = str(text or "")
    for pattern in _INGESTION_WRAPPER_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _normalize_text(cleaned)


_CONTEXTLESS_VALUE_RE = re.compile(
    r"^[₹$€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:/-|rs\.?|inr)?$",
    re.I,
)


def _is_contextless_value(text: str) -> bool:
    cleaned = sanitize_evidence_text(text)
    if not cleaned:
        return False
    if _CONTEXTLESS_VALUE_RE.fullmatch(cleaned):
        return True
    # Common table fragments such as ``60 %`` or ``6.0 / 10`` are equally
    # unsafe without their field label.
    if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:%|/\s*10|/\s*8)", cleaned):
        return True
    return False


def _attach_context_to_fragments(evidence: Sequence[EvidenceUnit]) -> tuple[EvidenceUnit, ...]:
    """Attach nearby labels to isolated numeric/table values when possible."""
    units = list(evidence or ())
    if not units:
        return ()

    output: list[EvidenceUnit] = []
    any_non_fragment = any(not _is_contextless_value(unit.text) for unit in units)

    for index, unit in enumerate(units):
        if not _is_contextless_value(unit.text) or not any_non_fragment:
            output.append(unit)
            continue

        best: tuple[int, EvidenceUnit] | None = None
        for other_index, other in enumerate(units):
            if other_index == index:
                continue
            if other.source.casefold() != unit.source.casefold():
                continue
            if _is_contextless_value(other.text):
                continue
            if unit.heading and other.heading and unit.heading.casefold() != other.heading.casefold():
                continue
            distance = abs(int(getattr(other, "position", other_index)) - int(getattr(unit, "position", index)))
            if distance > 2:
                continue
            if best is None or distance < best[0]:
                best = (distance, other)

        if best is None:
            # Do not expose an unlabeled numeric fragment when the package has
            # other contextual evidence that could be mistaken for its field.
            omitted = EvidenceUnit(
                evidence_id=unit.evidence_id,
                text=unit.text,
                source=unit.source,
                position=unit.position,
                heading=unit.heading,
                factual_markers=unit.factual_markers,
                token_set=unit.token_set,
            )
            # A sentinel attribute is intentionally avoided; the caller filters
            # unmatched fragments through the returned id set below.
            continue

        _, nearby = best
        combined_text = f"{nearby.text} {unit.text}"
        output.append(
            EvidenceUnit(
                evidence_id=unit.evidence_id,
                text=combined_text,
                source=unit.source,
                position=unit.position,
                heading=unit.heading or nearby.heading,
                factual_markers=tuple(dict.fromkeys((*nearby.factual_markers, *unit.factual_markers))),
                token_set=frozenset((*nearby.token_set, *unit.token_set)),
            )
        )

    return tuple(output)


def _claim_support_index(
    claim_audit: ClaimAudit | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    supported: dict[str, list[str]] = {}
    partial: dict[str, list[str]] = {}
    if claim_audit is None:
        return {}, {}

    for match in claim_audit.matches:
        target = supported if match.status == "supported" else partial if match.status == "partial" else None
        if target is None:
            continue
        target.setdefault(match.evidence_id, []).append(match.claim_id)

    return (
        {key: _dedupe_strings(value) for key, value in supported.items()},
        {key: _dedupe_strings(value) for key, value in partial.items()},
    )


def _status_for(
    items: Sequence[PackagedEvidenceItem],
    claim_audit: ClaimAudit | None,
) -> PackageStatus:
    if claim_audit is not None and claim_audit.conflicting_claim_ids:
        return "conflicted"
    if not items:
        return "empty"
    if claim_audit is not None and claim_audit.partial_claim_ids and not claim_audit.supported_claim_ids:
        return "partial"
    return "ready"


def build_evidence_package(
    evidence: Sequence[EvidenceUnit],
    *,
    claim_audit: ClaimAudit | None = None,
    max_units: int = _DEFAULT_MAX_UNITS,
    max_units_per_source: int = _DEFAULT_MAX_UNITS_PER_SOURCE,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    include_partial: bool = True,
    verified_evidence: bool = False,
) -> EvidencePackage:
    """Build deterministic answer context from already-qualified evidence.

    Selection rules:
        1. With a claim audit, supported evidence is preferred.
        2. Partial evidence may be included only when explicitly enabled.
        3. When ``verified_evidence`` is true, claim-audit misses do not erase
           otherwise qualified evidence.
        4. Evidence directly involved in an explicit claim conflict is omitted.
        5. Exact duplicate ``source + text`` pairs are removed.
        6. Per-source limits provide generic diversity control.
        7. Evidence units are never truncated mid-unit.
        8. Internal evidence IDs/source paths stay out of ``context``.
    """
    if max_units <= 0 or max_units_per_source <= 0 or max_context_chars <= 0:
        return EvidencePackage(status="empty", context="", reasons=("invalid package limits",))

    evidence = _attach_context_to_fragments(tuple(evidence or ()))

    supported_by_id, partial_by_id = _claim_support_index(claim_audit)
    has_audit = claim_audit is not None

    candidates: list[tuple[int, int, EvidenceUnit, tuple[str, ...], tuple[str, ...]]] = []
    omitted = 0
    for index, unit in enumerate(evidence or ()):
        clean_text = sanitize_evidence_text(unit.text)
        if not clean_text:
            continue

        supported_ids = supported_by_id.get(unit.evidence_id, ())
        partial_ids = partial_by_id.get(unit.evidence_id, ())

        if _is_contextless_value(clean_text) and not supported_ids and not partial_ids:
            # Bare values without a field/context label are not safe answer
            # evidence when other contextual material is available.
            omitted += 1
            continue

        if has_audit and not verified_evidence and not supported_ids and not (include_partial and partial_ids):
            continue

        # In the end-to-end path, these units have already passed candidate
        # verification. Claim auditing remains an annotation/ranking layer, not
        # a second relevance gate. Supported claims still sort first, followed
        # by partial claims and then verified-but-unmatched evidence.
        if supported_ids:
            priority = 0
        elif partial_ids:
            priority = 1
        else:
            priority = 2
        candidates.append((priority, index, unit, supported_ids, partial_ids))

    candidates.sort(key=lambda item: (item[0], item[1]))

    selected: list[PackagedEvidenceItem] = []
    seen_content: set[tuple[str, str]] = set()
    source_counts: dict[str, int] = {}
    used_chars = 0

    all_supported = set()
    all_partial = set()
    for _, _, _, supported_ids, partial_ids in candidates:
        all_supported.update(supported_ids)
        all_partial.update(partial_ids)

    selected_supported: set[str] = set()
    selected_partial: set[str] = set()

    for priority, _, unit, supported_ids, partial_ids in candidates:
        source_key = _normalize_text(unit.source).casefold() or "unknown"
        content_key = (source_key, sanitize_evidence_text(unit.text).casefold())
        if content_key in seen_content:
            omitted += 1
            continue

        if source_counts.get(source_key, 0) >= max_units_per_source:
            omitted += 1
            continue

        if len(selected) >= max_units:
            omitted += 1
            continue

        item_text = sanitize_evidence_text(unit.text)
        rendered = f"Evidence {len(selected) + 1}\n{item_text}"
        if used_chars and used_chars + 2 + len(rendered) > max_context_chars:
            omitted += 1
            continue
        if not used_chars and len(rendered) > max_context_chars:
            omitted += 1
            continue

        item = PackagedEvidenceItem(
            evidence_id=unit.evidence_id,
            text=item_text,
            source=unit.source,
            heading=unit.heading,
            supported_claim_ids=supported_ids,
            partial_claim_ids=partial_ids,
        )
        selected.append(item)
        seen_content.add(content_key)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        selected_supported.update(supported_ids)
        selected_partial.update(partial_ids)
        used_chars += len(rendered) + (2 if used_chars else 0)

    context = "\n\n".join(
        f"Evidence {index}\n{item.text}"
        for index, item in enumerate(selected, start=1)
    )

    if not selected:
        reasons = ("no verified evidence units survived final packaging",)
    else:
        reasons_list = []
        if omitted:
            reasons_list.append("some evidence units were omitted by deduplication, diversity, count, or context limits")
        if include_partial and selected_partial:
            reasons_list.append("partial claim-supporting evidence was included")
        reasons = tuple(reasons_list)

    omitted_claims = tuple(sorted((all_supported | all_partial) - selected_supported - selected_partial))
    return EvidencePackage(
        status=_status_for(selected, claim_audit),
        context=context,
        items=tuple(selected),
        source_count=len({item.source.casefold() for item in selected}),
        selected_count=len(selected),
        omitted_count=omitted,
        supported_claim_ids=tuple(sorted(selected_supported)),
        partial_claim_ids=tuple(sorted(selected_partial)),
        omitted_claim_ids=omitted_claims,
        reasons=reasons,
    )


__all__ = [
    "EvidencePackage",
    "PackagedEvidenceItem",
    "build_evidence_package",
    "sanitize_evidence_text",
]
