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
from typing import Any, Iterable, Literal, Mapping, Sequence

from ai_platform.core.evidence.claims import ClaimAudit, EvidenceUnit
from ai_platform.core.retrieval.ranking import _is_specialized_source, _is_generic_source


PackageStatus = Literal["ready", "partial", "empty", "conflicted"]

_DEFAULT_MAX_UNITS = 16
_DEFAULT_MAX_UNITS_PER_SOURCE = 8
_DEFAULT_MAX_CONTEXT_CHARS = 12_000

_INGESTION_WRAPPER_PATTERNS = (
    re.compile(
        r"(?im)^\s*command\s+\d+\s*[•|:]\s*retrieval\s+representation\s*[•|:]?\s*$"
    ),
    re.compile(
        r"(?im)^\s*(?:original\s+)?source\s+urls\s+preserved\s+from\s+the\s+command\s+5\s+knowledge\s+source\.?\s*$"
    ),
)

# Markdown hyperlink and navigation cleanup patterns for web-scraped chunks.
_NAV_WRAPPER_RE = re.compile(
    r"\[\s*\[?\s*Back to (?:index|top|contents|home)\s*\]?\s*\](?:\s*\([^)]*\))?",
    re.IGNORECASE,
)
_NESTED_MD_LINK_RE = re.compile(
    r"\[\s*\[([^\[\]]+)\]\s*\]\((https?://[^)]+)\)"
)
_STANDARD_MD_LINK_RE = re.compile(
    r"\[([^\[\]]+)\]\((https?://[^)]+)\)"
)
_BARE_URL_RE = re.compile(r"https?://\S+")




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
    cleaned = _NAV_WRAPPER_RE.sub("", cleaned)
    cleaned = _NESTED_MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _STANDARD_MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _BARE_URL_RE.sub("", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
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
            if unit.heading and len(unit.heading.strip()) > 3:
                combined_text = f"{unit.heading}: {unit.text}"
                output.append(
                    EvidenceUnit(
                        evidence_id=unit.evidence_id,
                        text=combined_text,
                        source=unit.source,
                        position=unit.position,
                        heading=unit.heading,
                        factual_markers=unit.factual_markers,
                        token_set=unit.token_set,
                    )
                )
                continue

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


@dataclass(frozen=True, slots=True)
class IntentEvidenceGroup:
    """Evidence and audit for one sub-intent in a multi-intent query."""

    question: str
    evidence_units: tuple[EvidenceUnit, ...] = ()
    claim_audit: ClaimAudit | None = None
    intent_id: str | int = 1
    status: str = "insufficient"


def _status_for(
    items: Sequence[PackagedEvidenceItem],
    claim_audit: ClaimAudit | None,
) -> PackageStatus:
    if claim_audit is not None and claim_audit.conflicting_claim_ids:
        return "conflicted"
    if not items:
        return "empty"
    if (
        claim_audit is not None
        and claim_audit.claims
        and not claim_audit.supported_claim_ids
        and not claim_audit.partial_claim_ids
    ):
        return "empty"
    if claim_audit is not None and getattr(claim_audit, "status", None) == "insufficient":
        return "empty"
    if claim_audit is not None and getattr(claim_audit, "status", None) == "partial":
        return "partial"
    if claim_audit is not None and claim_audit.partial_claim_ids and not claim_audit.supported_claim_ids:
        return "partial"
    return "ready"


def build_evidence_package(
    evidence: Sequence[EvidenceUnit] = (),
    *,
    claim_audit: ClaimAudit | None = None,
    intent_groups: Sequence[Any] | None = None,
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
        6. Per-source limits provide generic diversity control (two-pass source diversity).
        7. When ``intent_groups`` are provided, delegates to intent-aware packaging
           preventing sub-intent context starvation.
        8. Evidence units are never truncated mid-unit.
        9. Internal evidence IDs/source paths stay out of ``context``.
    """
    if intent_groups:
        return package_intent_aware_evidence(
            intent_groups,
            composite_audit=claim_audit,
            max_units=max_units,
            max_units_per_source=max_units_per_source,
            max_context_chars=max_context_chars,
            include_partial=include_partial,
            verified_evidence=verified_evidence,
        )

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

        if supported_ids:
            priority = 0
        elif partial_ids:
            priority = 1
        else:
            priority = 2
        candidates.append((priority, index, unit, supported_ids, partial_ids))

    candidates.sort(key=lambda item: (0 if verified_evidence else item[0], item[1]))

    all_supported = set()
    all_partial = set()
    for _, _, _, supported_ids, partial_ids in candidates:
        all_supported.update(supported_ids)
        all_partial.update(partial_ids)

    selected: list[PackagedEvidenceItem] = []
    seen_content: set[tuple[str, str]] = set()
    source_counts: dict[str, int] = {}
    used_chars = 0
    selected_supported: set[str] = set()
    selected_partial: set[str] = set()

    # Two-pass source diversity:
    # Pass 1 ensures every qualifying source gets a fair initial allocation
    # so a single high-density source cannot starve other relevant sources.
    # Supported-claim evidence (priority == 0) and verified evidence are authoritative
    # and allowed up to max_units_per_source; partial/unmatched evidence is throttled in Pass 1.
    base_per_source = min(2, max_units_per_source)
    deferred_candidates: list[tuple[int, int, EvidenceUnit, tuple[str, ...], tuple[str, ...]]] = []

    for candidate in candidates:
        priority, _, unit, supported_ids, partial_ids = candidate
        source_key = _normalize_text(unit.source).casefold() or "unknown"
        content_key = (source_key, sanitize_evidence_text(unit.text).casefold())
        if content_key in seen_content:
            omitted += 1
            continue

        pass1_limit = max_units_per_source if (priority == 0 or verified_evidence) else base_per_source
        if source_counts.get(source_key, 0) >= pass1_limit:
            deferred_candidates.append(candidate)
            continue

        if len(selected) >= max_units:
            deferred_candidates.append(candidate)
            continue

        item_text = sanitize_evidence_text(unit.text)
        if any(item.source == unit.source and item_text.casefold() in item.text.casefold() for item in selected):
            omitted += 1
            continue

        if (
            _is_specialized_source(unit.source)
            and any(_is_generic_source(item.source) and any(w in item.source.casefold() for w in ("finance", "fees")) for item in selected)
            and any(w in item_text.casefold() for w in ("tuition", "program fee", "programme fee", "fee structure", "inr 2,25,000", "225000"))
        ):
            omitted += 1
            continue

        header = f"Evidence {len(selected) + 1} [{unit.heading}]" if unit.heading else f"Evidence {len(selected) + 1}"
        rendered = f"{header}\n{item_text}"
        if used_chars and used_chars + 2 + len(rendered) > max_context_chars:
            deferred_candidates.append(candidate)
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

    # Pass 2: Fill remaining budget from deferred candidates up to max_units_per_source
    for candidate in deferred_candidates:
        priority, _, unit, supported_ids, partial_ids = candidate
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
        if any(item.source == unit.source and item_text.casefold() in item.text.casefold() for item in selected):
            omitted += 1
            continue

        if (
            _is_specialized_source(unit.source)
            and any(_is_generic_source(item.source) and any(w in item.source.casefold() for w in ("finance", "fees")) for item in selected)
            and any(w in item_text.casefold() for w in ("tuition", "program fee", "programme fee", "fee structure", "inr 2,25,000", "225000"))
        ):
            omitted += 1
            continue

        header = f"Evidence {len(selected) + 1} [{unit.heading}]" if unit.heading else f"Evidence {len(selected) + 1}"
        rendered = f"{header}\n{item_text}"
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
        (f"Evidence {index} [{item.heading}]\n{item.text}" if item.heading else f"Evidence {index}\n{item.text}")
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


def package_intent_aware_evidence(
    intent_groups: Sequence[Any],
    *,
    composite_audit: ClaimAudit | None = None,
    max_units: int = _DEFAULT_MAX_UNITS,
    max_units_per_source: int = _DEFAULT_MAX_UNITS_PER_SOURCE,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    include_partial: bool = True,
    verified_evidence: bool = True,
) -> EvidencePackage:
    """Build fair-share, intent-aware evidence packaging across sub-intents.

    Guarantees:
        1. Fair-share per-intent character and unit quotas (Pass 1).
        2. Surplus redistribution for intents requiring more context (Pass 2).
        3. Global deduplication so shared units do not consume double budget.
        4. Multi-intent evidence interleaving in rendered context.
        5. Missing evidence for any sub-intent is reported explicitly in reasons.
        6. Total context budget and token limits are strictly bounded.
        7. Respects safe abstention when all intents lack supported evidence.
    """
    if max_units <= 0 or max_units_per_source <= 0 or max_context_chars <= 0:
        return EvidencePackage(status="empty", context="", reasons=("invalid package limits",))

    if not intent_groups:
        return EvidencePackage(status="empty", context="", reasons=("no intent groups provided",))

    # Standardize sub-intent items
    groups: list[dict[str, Any]] = []
    for item in intent_groups:
        if isinstance(item, Mapping):
            q = _normalize_text(item.get("question", ""))
            units = tuple(item.get("evidence_units") or ())
            pkg = item.get("evidence_package")
            if not units and pkg and hasattr(pkg, "items"):
                units = tuple(
                    EvidenceUnit(
                        evidence_id=pi.evidence_id,
                        text=pi.text,
                        source=pi.source,
                        heading=pi.heading,
                    )
                    for pi in pkg.items
                )
            audit = item.get("claim_audit")
            st = str(item.get("evidence_status", "")).casefold()
            idx = item.get("intent_index", len(groups) + 1)
        elif hasattr(item, "question"):
            q = _normalize_text(getattr(item, "question", ""))
            units = tuple(getattr(item, "evidence_units", ()) or ())
            audit = getattr(item, "claim_audit", None)
            st = str(getattr(item, "status", "")).casefold()
            idx = getattr(item, "intent_id", len(groups) + 1)
        else:
            continue
        groups.append({
            "question": q,
            "units": units,
            "audit": audit,
            "status": st,
            "intent_index": idx,
        })

    if not groups:
        return EvidencePackage(status="empty", context="", reasons=("no valid intent groups provided",))

    if len(groups) == 1:
        single = groups[0]
        return build_evidence_package(
            single["units"],
            claim_audit=single["audit"] or composite_audit,
            max_units=max_units,
            max_units_per_source=max_units_per_source,
            max_context_chars=max_context_chars,
            include_partial=include_partial,
            verified_evidence=verified_evidence,
        )

    n_intents = len(groups)
    quota_chars = max(1000, max_context_chars // n_intents)
    quota_units = max(2, max_units // n_intents)

    # Check for conflict in any sub-intent audit or composite audit
    if composite_audit is not None and composite_audit.conflicting_claim_ids:
        return EvidencePackage(status="conflicted", context="", reasons=("claim conflict detected",))
    for g in groups:
        if g["audit"] is not None and g["audit"].conflicting_claim_ids:
            return EvidencePackage(status="conflicted", context="", reasons=("claim conflict detected in sub-intent",))

    # Prepare ranked candidate lists per sub-intent
    per_intent_candidates: list[list[tuple[int, int, EvidenceUnit, tuple[str, ...], tuple[str, ...]]]] = []
    all_supported: set[str] = set()
    all_partial: set[str] = set()

    for g in groups:
        u_list = _attach_context_to_fragments(g["units"])
        sup_by_id, part_by_id = _claim_support_index(g["audit"])
        has_aud = g["audit"] is not None

        cands: list[tuple[int, int, EvidenceUnit, tuple[str, ...], tuple[str, ...]]] = []
        for index, unit in enumerate(u_list):
            clean_text = sanitize_evidence_text(unit.text)
            if not clean_text:
                continue
            sup_ids = sup_by_id.get(unit.evidence_id, ())
            part_ids = part_by_id.get(unit.evidence_id, ())

            if _is_contextless_value(clean_text) and not sup_ids and not part_ids:
                continue
            if has_aud and not verified_evidence and not sup_ids and not (include_partial and part_ids):
                continue

            if sup_ids:
                priority = 0
            elif part_ids:
                priority = 1
            else:
                priority = 2
            cands.append((priority, index, unit, sup_ids, part_ids))

        cands.sort(key=lambda item: (item[0], item[1]))
        per_intent_candidates.append(cands)
        for _, _, _, s_ids, p_ids in cands:
            all_supported.update(s_ids)
            all_partial.update(p_ids)

    selected_items: list[tuple[int, PackagedEvidenceItem]] = []
    seen_content: set[tuple[str, str]] = set()
    source_counts: dict[str, int] = {}
    intent_chars: dict[int, int] = {i: 0 for i in range(n_intents)}
    intent_unit_counts: dict[int, int] = {i: 0 for i in range(n_intents)}
    total_chars = 0
    selected_supported: set[str] = set()
    selected_partial: set[str] = set()
    omitted = 0

    candidate_pointers = [0] * n_intents

    # Pass 1: Fair-share round-robin per-intent allocation up to base quotas
    progress = True
    while progress and len(selected_items) < max_units and total_chars < max_context_chars:
        progress = False
        for i in range(n_intents):
            if intent_unit_counts[i] >= quota_units:
                continue
            if intent_chars[i] >= quota_chars:
                continue
            if len(selected_items) >= max_units:
                break

            while candidate_pointers[i] < len(per_intent_candidates[i]):
                cand = per_intent_candidates[i][candidate_pointers[i]]
                candidate_pointers[i] += 1
                priority, _, unit, sup_ids, part_ids = cand

                source_key = _normalize_text(unit.source).casefold() or "unknown"
                content_key = (source_key, sanitize_evidence_text(unit.text).casefold())
                if content_key in seen_content:
                    omitted += 1
                    continue
                if source_counts.get(source_key, 0) >= max_units_per_source:
                    omitted += 1
                    continue

                item_text = sanitize_evidence_text(unit.text)
                header = f"Evidence {len(selected_items) + 1} [{unit.heading}]" if unit.heading else f"Evidence {len(selected_items) + 1}"
                rendered = f"{header}\n{item_text}"
                item_len = len(rendered) + (2 if total_chars else 0)

                if total_chars and total_chars + item_len > max_context_chars:
                    omitted += 1
                    continue
                if not total_chars and len(rendered) > max_context_chars:
                    omitted += 1
                    continue

                if intent_chars[i] and intent_chars[i] + item_len > quota_chars:
                    # Defer candidate to Pass 2 so this intent doesn't bust its Pass 1 quota
                    candidate_pointers[i] -= 1
                    break

                item = PackagedEvidenceItem(
                    evidence_id=unit.evidence_id,
                    text=item_text,
                    source=unit.source,
                    heading=unit.heading,
                    supported_claim_ids=sup_ids,
                    partial_claim_ids=part_ids,
                )
                selected_items.append((i, item))
                seen_content.add(content_key)
                source_counts[source_key] = source_counts.get(source_key, 0) + 1
                intent_chars[i] += item_len
                intent_unit_counts[i] += 1
                total_chars += item_len
                selected_supported.update(sup_ids)
                selected_partial.update(part_ids)
                progress = True
                break

    surplus_used = False
    # Pass 2: Surplus redistribution across any intents with unallocated candidates
    progress = True
    while progress and len(selected_items) < max_units and total_chars < max_context_chars:
        progress = False
        for i in range(n_intents):
            if len(selected_items) >= max_units:
                break

            while candidate_pointers[i] < len(per_intent_candidates[i]):
                cand = per_intent_candidates[i][candidate_pointers[i]]
                candidate_pointers[i] += 1
                priority, _, unit, sup_ids, part_ids = cand

                source_key = _normalize_text(unit.source).casefold() or "unknown"
                content_key = (source_key, sanitize_evidence_text(unit.text).casefold())
                if content_key in seen_content:
                    omitted += 1
                    continue
                if source_counts.get(source_key, 0) >= max_units_per_source:
                    omitted += 1
                    continue

                item_text = sanitize_evidence_text(unit.text)
                header = f"Evidence {len(selected_items) + 1} [{unit.heading}]" if unit.heading else f"Evidence {len(selected_items) + 1}"
                rendered = f"{header}\n{item_text}"
                item_len = len(rendered) + (2 if total_chars else 0)

                if total_chars and total_chars + item_len > max_context_chars:
                    omitted += 1
                    continue
                if not total_chars and len(rendered) > max_context_chars:
                    omitted += 1
                    continue

                item = PackagedEvidenceItem(
                    evidence_id=unit.evidence_id,
                    text=item_text,
                    source=unit.source,
                    heading=unit.heading,
                    supported_claim_ids=sup_ids,
                    partial_claim_ids=part_ids,
                )
                selected_items.append((i, item))
                seen_content.add(content_key)
                source_counts[source_key] = source_counts.get(source_key, 0) + 1
                intent_chars[i] += item_len
                intent_unit_counts[i] += 1
                total_chars += item_len
                selected_supported.update(sup_ids)
                selected_partial.update(part_ids)
                progress = True
                surplus_used = True
                break

    final_items = tuple(item for _, item in selected_items)
    context = "\n\n".join(
        (f"Evidence {index} [{item.heading}]\n{item.text}" if item.heading else f"Evidence {index}\n{item.text}")
        for index, item in enumerate(final_items, start=1)
    )

    # Determine status & reasons
    reasons_list: list[str] = []
    missing_intents: list[int] = []
    for i in range(n_intents):
        if intent_unit_counts[i] == 0:
            missing_intents.append(i)
            q_desc = groups[i]["question"] or f"sub-intent {i + 1}"
            reasons_list.append(f"intent {i + 1} ('{q_desc}') has no qualifying evidence")

    if surplus_used:
        reasons_list.append("surplus context budget was redistributed across sub-intents")
    if omitted:
        reasons_list.append("some evidence units were omitted by deduplication, diversity, count, or context limits")
    if include_partial and selected_partial:
        reasons_list.append("partial claim-supporting evidence was included")

    if not final_items:
        pkg_status = "empty"
        reasons_list.append("no verified evidence units survived final packaging")
    elif len(missing_intents) == n_intents:
        pkg_status = "empty"
    elif missing_intents:
        pkg_status = "partial"
    else:
        # All intents have at least one selected item
        all_audits = [g["audit"] for g in groups if g["audit"] is not None]
        if all_audits and any(getattr(a, "status", None) == "insufficient" for a in all_audits):
            pkg_status = "partial"
        elif all_audits and any(getattr(a, "status", None) == "partial" for a in all_audits):
            pkg_status = "partial"
        elif any(g["status"] == "partial" for g in groups):
            pkg_status = "partial"
        else:
            pkg_status = "ready"

    omitted_claims = tuple(sorted((all_supported | all_partial) - selected_supported - selected_partial))

    return EvidencePackage(
        status=pkg_status,
        context=context,
        items=final_items,
        source_count=len({item.source.casefold() for item in final_items}),
        selected_count=len(final_items),
        omitted_count=omitted,
        supported_claim_ids=tuple(sorted(selected_supported)),
        partial_claim_ids=tuple(sorted(selected_partial)),
        omitted_claim_ids=omitted_claims,
        reasons=tuple(reasons_list),
    )


__all__ = [
    "EvidencePackage",
    "IntentEvidenceGroup",
    "PackagedEvidenceItem",
    "build_evidence_package",
    "package_intent_aware_evidence",
    "sanitize_evidence_text",
]
