from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import query_understanding_llm


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryUnderstanding:
    """
    Institution-independent semantic representation of a user query.
    """

    original_query: str
    search_query: str
    intent: str
    is_list_question: bool
    confidence: float


MAX_SEARCH_QUERY_CHARS = 512


SYSTEM_PROMPT = """
You are the semantic query parser of a reusable multi-institution RAG system.

Your task is ONLY to understand the user's request and convert it into
one concise retrieval query.

DO NOT answer the user.
DO NOT provide institutional facts.
DO NOT invent missing details.
DO NOT broaden a narrow request.

Preserve every explicit qualifier that affects retrieval, including:
- person/category
- student/staff/visitor
- program/degree
- occupancy
- bedding
- duration
- date/year
- location
- admission mode
- any other explicit constraint

Understand:
- English
- casual English
- incomplete questions
- misspellings
- abbreviations
- Hinglish
- conversational wording

when the intended meaning is reasonably clear.

The search_query must:
- preserve the user's actual meaning
- make the request clearer for retrieval
- retain important qualifiers
- remove unnecessary conversational wording
- remain conservative
- never invent facts

The intent must be a short generic label such as:
fees, tuition_fees, hostel_fees, admissions,
admission_categories, programs, course_registration,
hostel, academics, unknown

Set is_list_question to TRUE when the user asks for multiple
items, categories, options, types, programs, documents,
facilities, steps, or another collection.

Examples:
- what are the programs?
- which categories are available?
- what all documents are required?
- list the hostel facilities
- kaun kaun se programs hain?
- kya kya documents chahiye?
- different admission categories kaun kaun se hain?

Return ONLY valid JSON:

{
  "search_query": "...",
  "intent": "...",
  "is_list_question": false,
  "confidence": 0.0
}

Do not include explanations.
Do not include markdown.
Do not include comments.
Do not include additional fields.

/no_think
""".strip()


def _extract_text(response) -> str:
    if response is None:
        return ""

    content = getattr(
        response,
        "content",
        response,
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text")
                if value:
                    parts.append(str(value))

        return "\n".join(parts).strip()

    return str(content).strip()


def _remove_thinking_blocks(text: str) -> str:
    cleaned = str(text or "").strip()

    if not cleaned:
        return ""

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    cleaned = re.sub(
        r"</think>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"<think>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()


def _extract_json(text: str) -> dict:
    cleaned = _remove_thinking_blocks(text)

    if not cleaned:
        raise ValueError(
            "Model returned empty output."
        )

    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            value, _ = decoder.raw_decode(
                cleaned[index:]
            )

            if isinstance(value, dict):
                return value

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "No valid JSON object found in model output."
    )


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
            "y",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "n",
        }:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return False


def _normalise_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if 1.0 < confidence <= 100.0:
        confidence /= 100.0

    return max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )


def _normalise_intent(value) -> str:
    intent = str(
        value or "unknown"
    ).strip().lower()

    if not intent:
        return "unknown"

    intent = re.sub(
        r"[^a-z0-9]+",
        "_",
        intent,
    )

    intent = re.sub(
        r"_+",
        "_",
        intent,
    ).strip("_")

    return intent or "unknown"


def _safe_search_query(
    search_query: str,
    original: str,
) -> tuple[str, float]:
    cleaned = str(
        search_query or ""
    ).strip()

    if not cleaned:
        return original, 0.3

    if len(cleaned) > MAX_SEARCH_QUERY_CHARS:
        logger.warning(
            "Oversized semantic search query. "
            "Falling back to original query."
        )

        return original, 0.2

    return cleaned, 1.0


# ------------------------------------------------------------
# Generic linguistic list detection
# ------------------------------------------------------------

_LIST_PATTERNS = (
    r"\bwhat\s+are\b",
    r"\bwhich\s+are\b",
    r"\bwhat\s+all\b",
    r"\blist\b",
    r"\bwhat\s+types?\b",
    r"\bwhat\s+categories?\b",
    r"\bwhich\s+options?\b",
    r"\bdifferent\s+\w+(?:\s+\w+){0,3}\s+(?:are|available|exist)\b",
    r"\bkaun\s+kaun\b",
    r"\bkaun\s+kaun\s+se\b",
    r"\bkaun\s+kaun\s+si\b",
    r"\bkon\s+kon\b",
    r"\bkon\s+kon\s+se\b",
    r"\bkon\s+kon\s+si\b",
    r"\bkya\s+kya\b",
    r"\bkaun\s+se\b",
    r"\bkaun\s+si\b",
    r"\bkon\s+se\b",
    r"\bkon\s+si\b",
)


_COMPILED_LIST_PATTERNS = tuple(
    re.compile(
        pattern,
        flags=re.IGNORECASE,
    )
    for pattern in _LIST_PATTERNS
)


def _has_strong_list_signal(text: str) -> bool:
    normalized = re.sub(
        r"\s+",
        " ",
        str(text or "").strip(),
    )

    if not normalized:
        return False

    return any(
        pattern.search(normalized)
        for pattern in _COMPILED_LIST_PATTERNS
    )


def _finalize_list_question(
    original_query: str,
    model_value: bool,
) -> bool:
    if model_value:
        return True

    if _has_strong_list_signal(
        original_query
    ):
        return True

    return False


# ------------------------------------------------------------
# Cached semantic inference
# ------------------------------------------------------------

@lru_cache(maxsize=256)
def _understand_query_cached(
    original: str,
) -> QueryUnderstanding:

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=(
                "USER QUERY:\n"
                + original
                + "\n\n/no_think"
            )
        ),
    ]

    try:
        response = query_understanding_llm.invoke(
            messages
        )

        raw_output = _extract_text(
            response
        )

        parsed = _extract_json(
            raw_output
        )

        search_query, confidence_multiplier = (
            _safe_search_query(
                parsed.get(
                    "search_query",
                    "",
                ),
                original,
            )
        )

        intent = _normalise_intent(
            parsed.get(
                "intent",
                "unknown",
            )
        )

        model_is_list_question = _parse_bool(
            parsed.get(
                "is_list_question",
                False,
            )
        )

        is_list_question = _finalize_list_question(
            original_query=original,
            model_value=model_is_list_question,
        )

        confidence = _normalise_confidence(
            parsed.get(
                "confidence",
                0.0,
            )
        )

        confidence *= confidence_multiplier

        return QueryUnderstanding(
            original_query=original,
            search_query=search_query,
            intent=intent,
            is_list_question=is_list_question,
            confidence=confidence,
        )

    except Exception:
        logger.exception(
            "Semantic query understanding failed. "
            "Falling back to original query."
        )

        return QueryUnderstanding(
            original_query=original,
            search_query=original,
            intent="unknown",
            is_list_question=False,
            confidence=0.0,
        )


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def understand_query(
    query: str,
) -> QueryUnderstanding:

    original = str(
        query or ""
    ).strip()

    if not original:
        return QueryUnderstanding(
            original_query="",
            search_query="",
            intent="unknown",
            is_list_question=False,
            confidence=0.0,
        )

    return _understand_query_cached(
        original
    )


def understand_query_dict(
    query: str,
) -> dict:

    return asdict(
        understand_query(query)
    )