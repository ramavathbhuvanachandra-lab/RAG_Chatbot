"""Deterministic conservative multi-intent decomposition.

This helper is not an answerer and does not use institution-specific facts.
It protects coordinated phrases, splits only plausible independent requests,
and resolves simple ordinal/anaphoric references within the same user turn.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_REQUEST_WORDS = {
    "what", "which", "who", "where", "when", "how", "why", "can", "could",
    "would", "do", "does", "did", "is", "are", "may",
}

_SINGLE_INTENT_COORDINATIONS = (
    r"\b(?:fees?|costs?)\s+and\s+(?:charges?|fees?|costs?)\b",
    r"\b(?:minimum\s+percentage|percentage)\s+and\s+eligibility\b",
    r"\b(?:application|admission)\s+(?:process|steps?)\s+and\s+(?:procedure|steps?)\b",
)

@dataclass(frozen=True, slots=True)
class IntentUnit:
    question: str
    topics: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()


def _clean(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _ensure_question(value: str) -> str:
    text = _clean(value).strip(" ,;.")
    return text if not text else (text if text.endswith("?") else text + "?")


def _starts_as_question(value: str) -> bool:
    first = _clean(value).casefold().split(maxsplit=1)
    return bool(first and first[0] in _REQUEST_WORDS)


def _protected_single_phrase(value: str) -> bool:
    return any(re.search(pattern, value, flags=re.I) for pattern in _SINGLE_INTENT_COORDINATIONS)


def _split_explicit(value: str) -> list[str]:
    if "?" in value:
        return [_ensure_question(part) for part in value.split("?") if _clean(part)]
    parts = [part for part in re.split(r"\s*;\s*", value) if _clean(part)]
    return [_ensure_question(part) for part in parts]


def _split_coordinated(value: str) -> list[str]:
    if _protected_single_phrase(value):
        return []
    normalized = _clean(value)
    matches = list(re.finditer(r"\s+(?:and|also)\s+", normalized, flags=re.I))
    for match in matches:
        left = _clean(normalized[:match.start()])
        right = _clean(normalized[match.end():])
        if not left or not right:
            continue
        if _starts_as_question(right):
            return [_ensure_question(left), _ensure_question(right)]
        if _starts_as_question(left) and len(right.split()) <= 10:
            prefix = re.match(
                r"^(what|which|where|when|how|who|why|can|could|would|is|are)\b",
                left,
                re.I,
            )
            if prefix:
                stem = prefix.group(1)
                remainder = left[len(prefix.group(0)):].strip()
                if remainder:
                    return [
                        _ensure_question(left),
                        _ensure_question(f"{stem} {right}"),
                    ]
    return []


def _resolve_local_ordinals(units: list[IntentUnit]) -> list[IntentUnit]:
    # Keep the structural questions intact. This only repairs obvious
    # references such as "what about the second one?" when a prior unit names
    # exactly two/three explicit subjects.
    subjects: list[str] = []
    for unit in units:
        match = re.search(
            r"\b(?:M\.?\s*Tech|M\.?\s*Sc\.?|MBA|B\.?\s*Tech|Ph\.?\s*D\.?)\b",
            unit.question,
            flags=re.I,
        )
        if match:
            subjects.append(match.group(0))

    if len(subjects) < 2:
        return units

    resolved: list[IntentUnit] = []
    for unit in units:
        question = unit.question
        if re.search(r"\bfirst\b", question, re.I) and len(subjects) >= 1:
            question = re.sub(r"\bfirst\s+(?:one|program)?\b", subjects[0], question, count=1, flags=re.I)
        elif re.search(r"\bsecond\b", question, re.I) and len(subjects) >= 2:
            question = re.sub(r"\bsecond\s+(?:one|program)?\b", subjects[1], question, count=1, flags=re.I)
        elif re.search(r"\bthird\b", question, re.I) and len(subjects) >= 3:
            question = re.sub(r"\bthird\s+(?:one|program)?\b", subjects[2], question, count=1, flags=re.I)
        resolved.append(IntentUnit(_ensure_question(question)))
    return resolved


def decompose_multi_intent(question: str) -> list[IntentUnit]:
    original = _clean(question)
    if not original:
        raise ValueError("question cannot be empty")

    explicit = _split_explicit(original)
    if len(explicit) > 1:
        return _resolve_local_ordinals([IntentUnit(part) for part in explicit])

    coordinated = _split_coordinated(original)
    if len(coordinated) > 1:
        return _resolve_local_ordinals([IntentUnit(part) for part in coordinated])

    return [IntentUnit(_ensure_question(original))]


def is_multi_intent(question: str) -> bool:
    return len(decompose_multi_intent(question)) > 1


def intent_questions(question: str) -> tuple[str, ...]:
    return tuple(unit.question for unit in decompose_multi_intent(question))


__all__ = ["IntentUnit", "decompose_multi_intent", "is_multi_intent", "intent_questions"]
