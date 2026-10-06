"""Production conversation resolver for the reusable institutional RAG core.

Responsibilities
----------------
1. Preserve standalone questions unchanged.
2. Detect genuine follow-ups conservatively.
3. Resolve references only from recent conversation evidence.
4. Resolve simple multi-intent references such as "the first one" and
   "the second one" without an extra LLM call when the antecedent is clear.
5. Use at most one lightweight LLM call when deterministic resolution is not
   sufficient.
6. Reject model rewrites that introduce an entity not supported by history.
7. Never invent institutional facts.

The resolver returns a small backward-compatible mapping consumed by the graph:
    resolved_question
    mode
    active_topic
    active_entity

Additional diagnostic fields are included for tests/observability and are safe
for callers that only consume the original four fields.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ai_platform.runtime.llm import query_understanding_llm


MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 7000
MAX_LLM_RESOLVED_CHARS = 700


_CONVERSATION_PROMPT = """
You are the conversation-resolution component of a college/institutional RAG assistant.
Your job is ONLY to resolve references in the latest user question using recent conversation.
Do not answer the question. Do not invent institutional facts.

Return JSON only:
{
  "resolved_question": "",
  "mode": "standalone|follow_up",
  "active_topic": "",
  "active_entity": "",
  "confidence": 0.0,
  "reason": ""
}

Rules:
- Keep a genuinely standalone latest question unchanged.
- Resolve only references supported by recent conversation.
- Preserve the latest question's requested attribute and constraints.
- If the latest question names a new target explicitly, do not replace it with an older target.
- Resolve "that program", "this course", "it", "the first one", "the second one", "former", and "latter" only from an unambiguous recent antecedent.
- When multiple antecedents are plausible, preserve the latest question unchanged.
- Never introduce a program, course, department, or other entity not supported by the supplied history.
""".strip()

_FOLLOW_UP_PREFIXES = (
    "what about",
    "how about",
    "how much",
    "how many",
    "when is it",
    "when are they",
    "where is it",
    "where are they",
    "what is it",
    "what are they",
    "is it",
    "are they",
    "can i",
    "can we",
    "could i",
    "could we",
    "do i",
    "do they",
    "does it",
    "tell me more",
    "more about",
    "and what",
    "and how",
    "and when",
    "and where",
)

_ANAPHOR_PATTERNS = (
    r"\bthat\s+(?:program|course|degree|department|school|option|one)\b",
    r"\bthis\s+(?:program|course|degree|department|school|option|one)\b",
    r"\bthe\s+(?:same|above|previous)\s+(?:program|course|degree|department|school|option|one)\b",
    r"\bthe\s+(?:first|second|third|last|former|latter)\s+(?:one|program|course|degree|option)?\b",
    r"\b(?:the\s+)?(?:first|second|third|last|former|latter)\s+(?:one|program|course|degree|option)\b",
    r"\b(?:it|they|them|this|that|these|those)\b",
)

_DEGREE_PATTERNS = (
    (re.compile(r"\bm\.?\s*tech\b", re.I), "M.Tech"),
    (re.compile(r"\bm\.?\s*sc\.?\b", re.I), "M.Sc"),
    (re.compile(r"\bm\.?\s*b\.?\s*a\.?\b|\bmba\b", re.I), "MBA"),
    (re.compile(r"\bb\.?\s*tech\b", re.I), "B.Tech"),
    (re.compile(r"\bph\.?\s*d\.?\b|\bphd\b", re.I), "PhD"),
)

_ENTITY_SUFFIX_RE = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9.&'()/\-]*(?:\s+[A-Z][A-Za-z0-9.&'()/\-]*){0,7})\s+"
    r"(?P<suffix>program|course|degree|department|school|institute|university|college)\b"
)

_PREPOSITION_ENTITY_RE = re.compile(
    r"\b(?:for|about|regarding|concerning|on)\s+(?:the\s+)?"
    r"(?P<name>[A-Z][A-Za-z0-9.&'()/\-]*(?:\s+[A-Z][A-Za-z0-9.&'()/\-]*){0,5})"
    r"(?:\s+(?:program|course|degree))?\b"
)

_TOPIC_PATTERNS = (
    ("application fee", "fee"),
    ("processing fee", "fee"),
    ("hostel fee", "fee"),
    ("fee", "fee"),
    ("fees", "fee"),
    ("फीस", "fee"),
    ("शुल्क", "fee"),
    ("deadline", "deadline"),
    ("last date", "deadline"),
    ("आखिरी तारीख", "deadline"),
    ("अंतिम तारीख", "deadline"),
    ("documents", "documents"),
    ("document", "documents"),
    ("docs", "documents"),
    ("दस्तावेज", "documents"),
    ("डॉक्यूमेंट", "documents"),
    ("eligibility", "eligibility"),
    ("eligible", "eligibility"),
    ("work experience", "eligibility"),
    ("experience compulsory", "eligibility"),
    ("experience necessary", "eligibility"),
    ("experience mandatory", "eligibility"),
    ("experience required", "eligibility"),
    ("experience zaroori", "eligibility"),
    ("experience chahiye", "eligibility"),
    ("work exp", "eligibility"),
    ("पात्रता", "eligibility"),
    ("योग्य", "eligibility"),
    ("काम का अनुभव", "eligibility"),
    ("नौकरी का अनुभव", "eligibility"),
    ("admission process", "process"),
    ("application process", "process"),
    ("how do i apply", "process"),
    ("how can i apply", "process"),
    ("आवेदन", "process"),
    ("प्रक्रिया", "process"),
    ("where is", "location"),
    ("located", "location"),
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _fold(value: Any) -> str:
    return _clean(value).casefold()


def _distinct(values: Sequence[str], limit: int = 16) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        item = _clean(value)
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _message_role_content(item: Any) -> tuple[str, str]:
    if isinstance(item, Mapping):
        role = _fold(item.get("role"))
        content = _clean(item.get("content") or item.get("message"))
        return role, content
    role = _fold(getattr(item, "role", ""))
    content = _clean(getattr(item, "content", item))
    return role, content


def _recent_messages(history: Sequence[Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for item in history or ():
        role, content = _message_role_content(item)
        if role not in {"user", "assistant"} or not content:
            continue
        result.append((role, content))
    return result[-MAX_HISTORY_MESSAGES:]


def _history_text(messages: Sequence[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        lines.append(f"{role.upper()}: {content}")
    text = "\n".join(lines)
    return text[-MAX_HISTORY_CHARS:]


def _extract_degree_candidates(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for pattern, canonical in _DEGREE_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), canonical))
    return found


def _extract_named_entities(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    for match in _ENTITY_SUFFIX_RE.finditer(text):
        name = _clean(match.group("name"))
        name = re.sub(
            r"^(?:the|a|an)\s+",
            "",
            name,
            flags=re.I,
        )
        if name:
            found.append((match.start("name"), name))

    for match in _PREPOSITION_ENTITY_RE.finditer(text):
        name = _clean(match.group("name"))
        name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I)
        # Do not treat a generic lowercase continuation as an entity.
        if name and re.search(r"[A-Z]", name):
            found.append((match.start("name"), name))

    return found


def _anchor_candidates(messages: Sequence[tuple[str, str]]) -> list[str]:
    """Extract ordered likely subjects from recent user turns.

    Degree/program markers are preferred because they are precise and portable.
    Generic named entities are a fallback for arbitrary institutions.
    """
    occurrences: list[tuple[int, int, str]] = []
    for message_index, (role, content) in enumerate(messages):
        if role != "user":
            continue
        for position, value in _extract_degree_candidates(content):
            occurrences.append((message_index, position, value))
        for position, value in _extract_named_entities(content):
            if not any(_fold(value) == _fold(existing) for _, _, existing in occurrences):
                occurrences.append((message_index, position, value))

    occurrences.sort(key=lambda item: (item[0], item[1]))
    result: list[str] = []
    seen: set[str] = set()
    for _, _, value in occurrences:
        key = _fold(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result[-8:]


def _latest_user_question(messages: Sequence[tuple[str, str]]) -> str:
    for role, content in reversed(messages):
        if role == "user":
            return content
    return ""


def _explicit_subject_in_question(question: str, anchors: Sequence[str]) -> bool:
    low = _fold(question)
    for anchor in anchors:
        if _fold(anchor) in low:
            return True
    # Degree recognition is independent of the exact history candidate list.
    return bool(_extract_degree_candidates(question))


def _has_anaphor(question: str) -> bool:
    low = _fold(question)
    return any(re.search(pattern, low, flags=re.I) for pattern in _ANAPHOR_PATTERNS)


def _looks_like_follow_up(question: str) -> bool:
    low = _fold(question)
    if not low:
        return False
    if _has_anaphor(question):
        return True
    if low.startswith(_FOLLOW_UP_PREFIXES):
        return True
    # Natural conversational wrappers should not hide a follow-up prefix:
    # "Okay, now what about the deadline?" / "So, how much is it?"
    stripped = re.sub(r"^(?:(?:okay|ok|now|so|then)[,\s]+)+", "", low, flags=re.I)
    if stripped.startswith(_FOLLOW_UP_PREFIXES):
        return True
    return len(low.split()) <= 5


def _topic_from_text(text: str) -> str:
    low = _fold(text)
    for phrase, topic in _TOPIC_PATTERNS:
        if phrase in low:
            return topic
    return ""


def _replace_reference(question: str, replacement: str) -> str:
    result = question
    patterns = (
        r"\bthat\s+(?:program|course|degree|department|school|option|one)\b",
        r"\bthis\s+(?:program|course|degree|department|school|option|one)\b",
        r"\bthe\s+(?:same|above|previous)\s+(?:program|course|degree|department|school|option|one)\b",
        r"\b(?:the\s+)?(?:first|second|third|last|former|latter)\s+(?:one|program|course|degree|option)\b",
    )
    for pattern in patterns:
        result = re.sub(pattern, replacement, result, count=1, flags=re.I)
    if _fold(replacement):
        result = re.sub(r"\b(?:that|this|it|they|them)\b", replacement, result, count=1, flags=re.I)
    return _clean(result)


def _append_subject(question: str, subject: str) -> str:
    value = _clean(question).rstrip(" ?.!;")
    if not subject or _fold(subject) in _fold(value):
        return _clean(question)
    if re.search(r"\b(?:what about|how about)\b", _fold(value)):
        return f"{value} for {subject}?"
    if re.match(r"^(?:how much|how many|when|where)\b", _fold(value)):
        return f"{value} for {subject}?"
    if re.fullmatch(r"(?:\d+(?:\.\d+)?)(?:\s*%|\s*(?:cgpa|gpa))?", _fold(value)):
        return f"{value} for {subject}?"
    return f"{value} for {subject}?"


def _ordinal_index(question: str) -> int | None:
    """Resolve ordinal references only when they modify an entity placeholder."""
    low = _fold(question)
    if re.search(r"\b(?:first|former)\s+(?:one|program|course|degree|option)\b", low):
        return 0
    if re.search(r"\b(?:second|latter)\s+(?:one|program|course|degree|option)\b", low):
        return 1
    if re.search(r"\bthird\s+(?:one|program|course|degree|option)\b", low):
        return 2
    if re.search(r"\blast\s+(?:one|program|course|degree|option)\b", low):
        return -1
    return None


def _candidate_for_reference(question: str, anchors: Sequence[str]) -> tuple[str | None, str]:
    if not anchors:
        return None, "no_candidate"

    index = _ordinal_index(question)
    if index is not None:
        if index == -1:
            return anchors[-1], "ordinal_last"
        if index < len(anchors):
            return anchors[index], "ordinal"
        return None, "ordinal_unavailable"

    low = _fold(question)
    if re.search(r"\b(?:that|this|same|previous|above)\s+(?:program|course|degree|department|school|option|one)\b", low):
        if len(anchors) == 1:
            return anchors[0], "single_clear_anaphor"
        return None, "ambiguous_anaphor"

    if re.search(r"\b(?:it|they|them|this|that)\b", low):
        if len(anchors) == 1:
            return anchors[0], "single_clear_pronoun"
        return None, "ambiguous_pronoun"

    # Permit natural discourse prefixes without making them a reference signal
    # themselves: "Okay, now what about the deadline?"
    ellipsis = re.sub(r"^(?:(?:okay|ok|now|so|then)[,\s]+)+", "", low, flags=re.I)
    if ellipsis.startswith(("what about", "how about", "when", "where", "how much", "how many")):
        if len(anchors) == 1:
            return anchors[0], "single_clear_ellipsis"
        return None, "ambiguous_ellipsis"

    if _topic_from_text(question) and len(anchors) == 1:
        return anchors[0], "single_clear_topic_ellipsis"

    # Questions such as "Is 7.1 CGPA enough?" remain contextual when there is
    # exactly one clear active entity.
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|cgpa|gpa)\b", low) and len(anchors) == 1:
        return anchors[0], "single_clear_numeric_context"

    # Numeric-only follow-ups such as "7.1?" or "60%?" can still be resolved
    # when exactly one explicit subject is active.
    numeric = low.rstrip(" ?.!;")
    if re.fullmatch(r"(?:\d+(?:\.\d+)?)(?:\s*%|\s*(?:cgpa|gpa))?", numeric):
        if len(anchors) == 1:
            return anchors[0], "single_clear_numeric_ellipsis"
        return None, "ambiguous_numeric_ellipsis"

    return None, "no_reference"


def _safe_fallback(question: str) -> dict[str, str]:
    return {
        "resolved_question": question,
        "mode": "standalone",
        "active_topic": "",
        "active_entity": "",
        "resolution_source": "fallback",
        "resolution_confidence": "0.0",
        "resolution_reason": "preserved_original",
    }


def _parse_model_response(response: Any, original: str) -> dict[str, str]:
    content = getattr(response, "content", response)
    text = _clean(content)
    if not text:
        return _safe_fallback(original)

    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return _safe_fallback(original)
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return _safe_fallback(original)

    if not isinstance(payload, Mapping):
        return _safe_fallback(original)

    resolved = _clean(payload.get("resolved_question")) or original
    if len(resolved) > MAX_LLM_RESOLVED_CHARS:
        return _safe_fallback(original)

    mode = _fold(payload.get("mode"))
    if mode not in {"standalone", "follow_up"}:
        mode = "follow_up" if _fold(resolved) != _fold(original) else "standalone"

    return {
        "resolved_question": resolved,
        "mode": mode,
        "active_topic": _clean(payload.get("active_topic")),
        "active_entity": _clean(payload.get("active_entity")),
        "resolution_source": "llm",
        "resolution_confidence": _clean(payload.get("confidence")) or "0.0",
        "resolution_reason": _clean(payload.get("reason")),
    }


def _model_rewrite_is_supported(
    original: str,
    resolved: str,
    history: Sequence[tuple[str, str]],
    anchors: Sequence[str],
) -> bool:
    """Reject rewrites that introduce unsupported subject matter.

    We only police newly introduced capitalized/degree-like anchors. Ordinary
    connective wording may be rephrased by the model.
    """
    if _fold(original) == _fold(resolved):
        return True

    resolved_degrees = [value for _, value in _extract_degree_candidates(resolved)]
    original_degrees = [value for _, value in _extract_degree_candidates(original)]
    history_degrees = [value for _, value in _extract_degree_candidates(" ".join(text for _, text in history))]
    allowed_degrees = {_fold(value) for value in [*anchors, *history_degrees]}

    for degree in resolved_degrees:
        if _fold(degree) not in allowed_degrees and _fold(degree) not in {_fold(x) for x in original_degrees}:
            return False

    # If the model inserted a known history anchor, it is allowed.
    for anchor in anchors:
        if _fold(anchor) in _fold(resolved):
            continue

    return True


def resolve_conversation(
    *,
    question: str,
    chat_history: Sequence[Any] | None = None,
) -> dict[str, str]:
    """Resolve a possible follow-up conservatively and deterministically first."""
    original = _clean(question)
    if not original:
        raise ValueError("question cannot be empty")

    messages = _recent_messages(tuple(chat_history or ()))
    if not messages:
        return _safe_fallback(original)

    if not _looks_like_follow_up(original):
        return {
            **_safe_fallback(original),
            "active_entity": _anchor_candidates(messages)[-1] if _anchor_candidates(messages) else "",
            "active_topic": _topic_from_text(_latest_user_question(messages)),
        }

    anchors = _anchor_candidates(messages)
    latest_question = _latest_user_question(messages)
    latest_anchors = _anchor_candidates([("user", latest_question)]) if latest_question else []
    # Reference resolution must consider all recent explicit entities for
    # anaphora/ordinals; otherwise the latest mention can incorrectly hide a
    # second plausible antecedent (e.g. M.Tech + MBA -> "that program").
    if _has_anaphor(original) or _ordinal_index(original) is not None:
        reference_anchors = anchors
    else:
        reference_anchors = latest_anchors or anchors
    latest_topic = _topic_from_text(latest_question)

    # Explicit subject means the latest question is already self-contained.
    if _explicit_subject_in_question(original, anchors) and not _has_anaphor(original):
        return {
            "resolved_question": original,
            "mode": "standalone",
            "active_topic": _topic_from_text(original) or latest_topic,
            "active_entity": (anchors[-1] if len(anchors) == 1 else ""),
            "resolution_source": "explicit_subject",
            "resolution_confidence": "1.0",
            "resolution_reason": "latest_question_is_self_contained",
        }

    candidate, reason = _candidate_for_reference(original, reference_anchors)

    # High-confidence deterministic resolution: no LLM cost.
    if candidate:
        resolved = _replace_reference(original, candidate)
        if resolved.casefold() == original.casefold() and reason in {
            "single_clear_ellipsis",
            "single_clear_topic_ellipsis",
            "single_clear_numeric_ellipsis",
            "single_clear_numeric_context",
        }:
            resolved = _append_subject(original, candidate)
        return {
            "resolved_question": resolved,
            "mode": "follow_up",
            "active_topic": _topic_from_text(original) or latest_topic,
            "active_entity": candidate,
            "resolution_source": "deterministic",
            "resolution_confidence": "0.98",
            "resolution_reason": reason,
        }

    # Ambiguous multi-intent reference: never guess. Preserve the question and
    # let downstream planning handle it conservatively.
    if reason.startswith("ambiguous_") or reason == "ordinal_unavailable":
        return {
            "resolved_question": original,
            "mode": "standalone",
            "active_topic": _topic_from_text(original) or latest_topic,
            "active_entity": "",
            "resolution_source": "ambiguous_preserved",
            "resolution_confidence": "0.0",
            "resolution_reason": reason,
        }

    # One lightweight LLM call is reserved for genuinely contextual but
    # non-deterministic cases.
    history_text = _history_text(messages)
    prompt = (
        f"<HISTORY>\n{history_text}\n</HISTORY>\n"
        f"<LATEST_QUESTION>\n{original}\n</LATEST_QUESTION>\n"
        f"<KNOWN_ANCHORS>\n{', '.join(reference_anchors) or 'none'}\n</KNOWN_ANCHORS>"
    )

    try:
        response = query_understanding_llm.invoke(
            [
                ("system", _CONVERSATION_PROMPT),
                ("human", prompt),
            ]
        )
        parsed = _parse_model_response(response, original)
        active_entity = _fold(parsed.get("active_entity"))
        history_blob = _fold(" ".join(text for _, text in messages))
        original_blob = _fold(original)
        if active_entity and active_entity not in history_blob and active_entity not in original_blob:
            return {
                "resolved_question": original,
                "mode": "standalone",
                "active_topic": latest_topic,
                "active_entity": "",
                "resolution_source": "llm_rejected",
                "resolution_confidence": "0.0",
                "resolution_reason": "unsupported_model_entity",
            }
        if not _model_rewrite_is_supported(original, parsed["resolved_question"], messages, anchors):
            return {
                "resolved_question": original,
                "mode": "standalone",
                "active_topic": latest_topic,
                "active_entity": "",
                "resolution_source": "llm_rejected",
                "resolution_confidence": "0.0",
                "resolution_reason": "unsupported_model_rewrite",
            }
        if parsed["mode"] == "follow_up" and parsed["resolved_question"]:
            return parsed
    except Exception:
        pass

    return {
        **_safe_fallback(original),
        "active_topic": latest_topic,
        "resolution_reason": "llm_failed_or_invalid",
    }


__all__ = ["MAX_HISTORY_MESSAGES", "resolve_conversation"]
