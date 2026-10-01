"""Final user-visible answer guard for the reusable RAG core."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class GuardResult:
    answer: str
    status: str
    fallback_used: bool
    reason: str
    removed_categories: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "removed_categories": list(self.removed_categories),
        }


_INTERNAL_PATTERNS = (
    ("evidence_enumeration", re.compile(r"\b(?:evidence|document|chunk)\s*\d+\b", re.I)),
    ("reasoning_narration", re.compile(r"\b(?:the user is asking|the user asks|let me|first[, ]+i|i need to|looking at the evidence|looking through the evidence|the key is|the problem is|hmm|step by step|my reasoning|analysis:)\b", re.I)),
    ("retrieval_metadata", re.compile(r"\b(?:rrf|bm25|dense retrieval|retrieval score|retrieval rank|vector database|embedding|source path|chunk id)\b", re.I)),
    ("internal_path", re.compile(r"\b(?:backend|tests_reusable|data)[/\\][^\s]+", re.I)),
)


def _clean(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"```(?:text|markdown)?", "", value, flags=re.I)
    value = value.replace("```", "")
    value = re.sub(r"^\s*(?:final answer|answer)\s*:\s*", "", value, flags=re.I)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _leak_categories(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _INTERNAL_PATTERNS if pattern.search(text))


def guard_answer(answer: str, *, fallback: str = "") -> GuardResult:
    """Allow only a clean user-facing answer; otherwise fail closed."""
    cleaned = _clean(answer)
    if not cleaned:
        fallback_text = _clean(fallback)
        if fallback_text:
            return GuardResult(
                answer=fallback_text,
                status="fallback",
                fallback_used=True,
                reason="empty_answer",
            )
        return GuardResult(
            answer="",
            status="empty",
            fallback_used=True,
            reason="empty_answer_no_fallback",
        )

    categories = _leak_categories(cleaned)
    if categories:
        fallback_text = _clean(fallback)
        if fallback_text:
            return GuardResult(
                answer=fallback_text,
                status="fallback",
                fallback_used=True,
                reason="internal_or_reasoning_output_rejected",
                removed_categories=categories,
            )
        return GuardResult(
            answer="",
            status="unsafe",
            fallback_used=True,
            reason="internal_or_reasoning_output_rejected",
            removed_categories=categories,
        )

    changed = cleaned != str(answer or "").strip()
    return GuardResult(
        answer=cleaned,
        status="sanitized" if changed else "clean",
        fallback_used=False,
        reason="output_cleaned" if changed else "output_clean",
    )


__all__ = ["GuardResult", "guard_answer"]