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
- When resolving a category follow-up (e.g., "What about sponsored students?"), retain the antecedent attribute (e.g., tuition fee, eligibility criteria) along with the active program and new category.
- When a follow-up asks a new question about an entity mentioned earlier (e.g., leave rules for scholars in that program), resolve the program reference while identifying the new topic.
- Respect user corrections and topic changes (e.g., "No, I meant MBA, not M.Tech").
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
    "is there",
    "are there",
    "are married",
    "does this",
    "can a",
    "can an",
    "can i",
    "can we",
    "could i",
    "could we",
    "do i",
    "do they",
    "does it",
    "what is the fine",
    "what fine",
    "is campus",
    "tell me more",
    "more about",
    "and what",
    "and how",
    "and when",
    "and where",
)

_ANAPHOR_PATTERNS = (
    r"\bthat\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
    r"\bthis\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
    r"\b(?:is|iss|in)\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
    r"\b(?:us|uss|un)\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
    r"\b(?:usmein|ismein|unmein)\b",
    r"\b(?:usse|isse)\b",
    r"\b(?:isi|usi)\s+(?:document|curriculum|table|list|programme|program|department|facility|place|shop)\b",
    r"\bthe\s+(?:same|above|previous)\s+(?:programme|program|course|degree|department|school|facility|place|option|one)\b",
    r"\bthe\s+(?:first|second|third|last|former|latter)\s+(?:one|programme|program|course|degree|facility|option)?\b",
    r"\b(?:the\s+)?(?:first|second|third|last|former|latter)\s+(?:one|programme|program|course|degree|facility|option)\b",
    r"\b(?:it|they|them|this|that|these|those)\b",
    r"\b(?:there|their\s+timings?|its\s+timings?)\b",
    r"\b(?:yeh|woh|ye|wo)\b",
    r"\b(?:next\s+session|another\s+session|agla\s+session|naya\s+session)\b",
)

_DEGREE_PATTERNS = (
    (re.compile(r"\bb\.?\s*s\.?\b|\bbs\b", re.I), "B.S."),
    (re.compile(r"\bm\.?\s*tech\b", re.I), "M.Tech"),
    (re.compile(r"\bm\.?\s*sc\.?\b", re.I), "M.Sc"),
    (re.compile(r"\bm\.?\s*b\.?\s*a\.?\b|\bmba\b", re.I), "MBA"),
    (re.compile(r"\bb\.?\s*tech\b", re.I), "B.Tech"),
    (re.compile(r"\bph\.?\s*d\.?\b|\bphd\b", re.I), "PhD"),
)

_FULL_PROGRAM_RE = re.compile(
    r"\b(?P<deg>B\.?S\.?|B\.?Tech\.?|M\.?Tech\.?|M\.?Sc\.?|MBA|Ph\.?D\.?)\s+(?:in|of)\s+"
    r"(?P<spec>[A-Z][A-Za-z0-9.&'()/\-]*(?:\s+(?:and|&|[A-Z][A-Za-z0-9.&'()/\-]*))*)\b"
)

_ENTITY_SUFFIX_RE = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9.&'()/\-]*(?:\s+[A-Z][A-Za-z0-9.&'()/\-]*){0,7})\s+"
    r"(?P<suffix>program|programme|course|degree|department|school|institute|university|college|curriculum|workshop|workshops|training|trainings)\b",
    re.I,
)

_PREPOSITION_ENTITY_RE = re.compile(
    r"\b(?:for|about|regarding|concerning|on|in|at)\s+(?:the\s+)?"
    r"(?P<name>[A-Z][A-Za-z0-9.&'()/\-]*(?:\s+(?:and|&|[A-Z][A-Za-z0-9.&'()/\-]*)){0,5})"
    r"(?:\s+(?:program|programme|course|degree|department|school|curriculum|workshop|workshops|training|trainings))?\b"
)

_TOPIC_PATTERNS = (
    ("casual leave", "leave"),
    ("medical leave", "leave"),
    ("maternity leave", "leave"),
    ("duty leave", "leave"),
    ("leave rules", "leave"),
    ("days of leave", "leave"),
    ("leave", "leave"),
    ("leaves", "leave"),
    ("vacation", "leave"),
    ("holiday", "leave"),
    ("छुट्टी", "leave"),
    ("अवकाश", "leave"),
    ("course registration", "registration"),
    ("credit limit", "registration"),
    ("add drop", "registration"),
    ("late registration", "registration"),
    ("placement", "placement"),
    ("placements", "placement"),
    ("internship", "placement"),
    ("internships", "placement"),
    ("placement package", "placement"),
    ("noc", "placement"),
    ("grade", "academics"),
    ("grading", "academics"),
    ("cgpa", "academics"),
    ("sgpa", "academics"),
    ("cpi", "academics"),
    ("spi", "academics"),
    ("fine", "discipline"),
    ("penalty", "discipline"),
    ("tuition fee", "fee"),
    ("application fee", "fee"),
    ("processing fee", "fee"),
    ("hostel fee", "fee"),
    ("hostel charges", "fee"),
    ("hostel rent", "fee"),
    ("room rent", "fee"),
    ("fee", "fee"),
    ("fees", "fee"),
    ("फीस", "fee"),
    ("शुल्क", "fee"),
    ("fellowship", "fellowship"),
    ("scholarship", "fellowship"),
    ("stipend", "fellowship"),
    ("financial assistance", "fellowship"),
    ("financial support", "fellowship"),
    ("hostel", "facility"),
    ("accommodation", "facility"),
    ("mess", "facility"),
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
    ("chip design", "research"),
    ("semiconductor", "research"),
    ("research area", "research"),
    ("research areas", "research"),
    ("research", "research"),
    ("graded credits", "curriculum"),
    ("audit credits", "curriculum"),
    ("curriculum", "curriculum"),
    ("credits", "curriculum"),
    ("credit", "curriculum"),
    ("syllabus", "curriculum"),
    ("course structure", "curriculum"),
    ("departments", "departments"),
    ("department", "departments"),
    ("centres", "departments"),
    ("centers", "departments"),
    ("schools", "departments"),
    ("school", "departments"),
    ("programmes", "programmes"),
    ("programs", "programmes"),
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
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return _fold(item[0]), _clean(item[1])
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
        if isinstance(item, (tuple, list)) and len(item) == 2:
            first = _fold(item[0])
            if first in {"user", "assistant"}:
                content = _clean(item[1])
                if content:
                    result.append((first, content))
            else:
                user_msg = _clean(item[0])
                asst_msg = _clean(item[1])
                if user_msg:
                    result.append(("user", user_msg))
                if asst_msg:
                    result.append(("assistant", asst_msg))
            continue
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
    full_spans: list[tuple[int, int]] = []
    for match in _FULL_PROGRAM_RE.finditer(text):
        found.append((match.start(), _clean(match.group(0))))
        full_spans.append((match.start(), match.end()))
    for pattern, canonical in _DEGREE_PATTERNS:
        for match in pattern.finditer(text):
            if not any(s <= match.start() < e for s, e in full_spans):
                found.append((match.start(), canonical))
    return found


_CATEGORY_WORDS = {
    "general", "gen", "obc", "sc", "st", "ews", "gen-ews", "pwd", "creamy", "non-creamy",
    "category", "categories", "applicant", "applicants", "student", "students", "candidate", "candidates"
}


_FACILITY_PATTERNS = (
    (re.compile(r"\bhostel\s+(?:accommodation|room\s+rent|room|charges|fee)?\b", re.I), "hostel"),
    (re.compile(r"\bhostels?\b", re.I), "hostel"),
    (re.compile(r"\bold\s+mess\b", re.I), "Old Mess"),
    (re.compile(r"\bnew\s+mess\b", re.I), "New Mess"),
    (re.compile(r"\bmess\b", re.I), "mess"),
    (re.compile(r"\bdining\b", re.I), "mess"),
    (re.compile(r"\bcanteen\b", re.I), "canteen"),
    (re.compile(r"\blibrary\b", re.I), "library"),
    (re.compile(r"\bgymkhana\b", re.I), "gymkhana"),
    (re.compile(r"\bsports?\b", re.I), "sports"),
    (re.compile(r"\b(?:health\s+cent(?:er|re)|hospital|dispensary|medical|clinic)\b", re.I), "health_centre"),
    (re.compile(r"\b(?:transport|transportation|bus|shuttle)\b", re.I), "transport"),
)


def _extract_facility_candidates(text: str) -> list[tuple[int, str]]:
    raw: list[tuple[int, int, str]] = []
    for pattern, canonical in _FACILITY_PATTERNS:
        for match in pattern.finditer(text):
            raw.append((match.start(), match.end(), canonical))
    # Deduplicate overlapping matches: preserve the longer span (e.g. 'Old Mess' over 'mess')
    raw.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    filtered: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, canonical in raw:
        if any(not (end <= occ_start or start >= occ_end) for occ_start, occ_end in occupied):
            continue
        occupied.append((start, end))
        filtered.append((start, canonical))
    return filtered


_STOPWORDS = {"what", "does", "the", "a", "an", "i'm", "im", "looking", "at", "tell", "me", "about", "i", "missed", "interested", "in"}


def _extract_named_entities(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    for match in _ENTITY_SUFFIX_RE.finditer(text):
        name = _clean(match.group("name"))
        words = name.split()
        while words and (words[0].lower().rstrip(".,;:") in _STOPWORDS or not words[0][0].isupper()):
            words.pop(0)
        name = " ".join(words)
        if name:
            words_set = {w.lower().rstrip(".,;:") for w in name.split()}
            if words_set and not words_set.issubset(_CATEGORY_WORDS) and name.casefold() not in {"iit jodhpur", "iit", "jodhpur"}:
                found.append((match.start("name"), name))

    for match in _PREPOSITION_ENTITY_RE.finditer(text):
        name = _clean(match.group("name"))
        words = name.split()
        while words and (words[0].lower().rstrip(".,;:") in _STOPWORDS or not words[0][0].isupper()):
            words.pop(0)
        name = " ".join(words)
        # Do not treat a generic lowercase continuation as an entity.
        if name and re.search(r"[A-Z]", name):
            words_set = {w.lower().rstrip(".,;:") for w in name.split()}
            if words_set and not words_set.issubset(_CATEGORY_WORDS) and name.casefold() not in {"iit jodhpur", "iit", "jodhpur"}:
                found.append((match.start("name"), name))

    return found


def _score_degree_context(content: str, position: int) -> int:
    prec = content[max(0, position - 40):position].lower()
    score = 0
    if re.search(r"\b(?:apply\s+(?:directly\s+)?(?:for|to)|admission\s+(?:to|in|for)|eligible\s+for|for\s+(?:regular\s+)?)\b", prec):
        score += 50
    return score


def _anchor_candidates(messages: Sequence[tuple[str, str]]) -> list[str]:
    """Extract ordered likely subjects from recent user turns.

    Degree/program markers are preferred because they are precise and portable.
    Generic named entities are a fallback for arbitrary institutions.
    """
    deg_occurrences: list[tuple[int, int, int, str]] = []
    other_occurrences: list[tuple[int, int, str]] = []
    for message_index, (role, content) in enumerate(messages):
        if role != "user":
            continue
        for position, value in _extract_degree_candidates(content):
            score = _score_degree_context(content, position)
            deg_occurrences.append((message_index, -score, position, value))
        for position, value in _extract_facility_candidates(content):
            other_occurrences.append((message_index, position, value))
        for position, value in _extract_named_entities(content):
            if not any(_fold(value) == _fold(existing[-1]) for existing in deg_occurrences):
                if not any(_fold(value) == _fold(existing[2]) for existing in other_occurrences):
                    other_occurrences.append((message_index, position, value))

    deg_occurrences.sort(key=lambda item: (item[0], item[1], item[2]))
    other_occurrences.sort(key=lambda item: (item[0], item[1]))
    all_occurrences: list[tuple[int, int, str]] = []
    for msg_idx, score, pos, val in deg_occurrences:
        all_occurrences.append((msg_idx, 10, val))
    for msg_idx, pos, val in other_occurrences:
        all_occurrences.append((msg_idx, 5, val))
    all_occurrences.sort(key=lambda item: (item[0], item[1]))
    occurrences = [x[2] for x in all_occurrences]
    result: list[str] = []
    seen: set[str] = set()
    for value in occurrences:
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
    if bool(_extract_degree_candidates(question)):
        return True
    if bool(_extract_facility_candidates(question)) and not _has_anaphor(question):
        return True
    return False


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
    stripped = re.sub(r"^(?:(?:okay|ok|now|so|then|aur|ab|kya|aur\s+bhi)[,\s]+)+", "", low, flags=re.I)
    if stripped.startswith(_FOLLOW_UP_PREFIXES):
        return True
    if re.search(r"\b(?:next\s+session|another\s+session|agla\s+session|naya\s+session)\b", low):
        return True
    if re.search(r"\b(?:total\s+kaise\s+banta\s+hai|numbers\s+add\s+karke|numbers\s+add)\b", low):
        return True
    # If the question contains an explicit campus facility/amenity and lacks anaphoric markers,
    # it is an independent topic change, not a follow-up to an academic degree/department.
    if _extract_facility_candidates(question) and not _has_anaphor(question):
        return False
    return len(low.split()) <= 5


def _topic_from_text(text: str) -> str:
    low = _fold(text)
    for phrase, topic in _TOPIC_PATTERNS:
        if phrase in low:
            return topic
    return ""


def _extract_query_attribute(question: str) -> str:
    """Extract asked attribute phrase (e.g. tuition fee, eligibility criteria, leave rules)."""
    patterns = (
        r"\b(?:what is the|what are the)\s+([a-z\s]+?)\s+(?:for|of|in)\b",
        r"\b(tuition fee|fee structure|hostel fee|application fee|processing fee|eligibility criteria|leave rules|admission process|selection process|scholarship amount|stipend)\b",
    )
    low = _fold(question)
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            val = _clean(m.group(1))
            if val and len(val.split()) <= 4:
                return val
    return ""


def _replace_reference(question: str, replacement: str) -> str:
    result = question
    patterns = (
        r"\bthat\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
        r"\bthis\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
        r"\b(?:is|iss)\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
        r"\b(?:us|uss)\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b",
        r"\bthe\s+(?:same|above|previous)\s+(?:programme|program|course|degree|department|school|facility|place|option|one)\b",
        r"\b(?:the\s+)?(?:first|second|third|last|former|latter)\s+(?:one|programme|program|course|degree|facility|option)\b",
    )
    for pattern in patterns:
        result = re.sub(pattern, replacement, result, count=1, flags=re.I)
    if _fold(replacement):
        result = re.sub(r"\b(?:paying|pay)\s+(?:it|that|this)\b", f"paying the fee for {replacement}", result, count=1, flags=re.I)
        result = re.sub(r"\b(?:usmein|ismein)\b", f"in {replacement}", result, count=1, flags=re.I)
        result = re.sub(r"\b(?:usse|isse)\b", f"from {replacement}", result, count=1, flags=re.I)
        result = re.sub(r"\b(?:isi|usi)\s+document\b", f"the {replacement} document", result, count=1, flags=re.I)
        result = re.sub(r"\b(?:how do i get|how to get|how to reach|how do i reach)\s+there\b", f"how do I get to {replacement}", result, count=1, flags=re.I)
        result = re.sub(r"\bthere\b", f"to {replacement}", result, count=1, flags=re.I)
        result = re.sub(r"\b(?:what are\s+)?their\s+timings\b", f"timings for {replacement}", result, count=1, flags=re.I)
        result = re.sub(r"\b(?:where can i buy|where can i get|where to get)\s+this\b", f"where can I get {replacement}", result, count=1, flags=re.I)
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
    if re.search(r"\b(?:that|this|is|iss|us|uss|same|previous|above)\s+(?:programme|program|course|degree|department|school|facility|place|spot|shop|outlet|option|one)\b", low):
        if len(anchors) == 1:
            return anchors[0], "single_clear_anaphor"
        return anchors[-1] if anchors else None, "latest_anaphor"

    if re.search(r"\b(?:there|their\s+timings?|its\s+timings?)\b", low):
        if anchors:
            return anchors[-1], "single_clear_locative_anaphor"

    if re.search(r"\b(?:how do i get|how to get|how do i reach|how to reach)\b", low):
        if anchors:
            return anchors[-1], "direction_followup"

    if re.search(r"\b(?:usmein|ismein|unmein|usse|isse)\b", low):
        if len(anchors) == 1:
            return anchors[0], "single_clear_hinglish_pronoun"
        return anchors[0] if anchors else None, "primary_target_hinglish_pronoun"

    if re.search(r"\b(?:isi|usi)\s+(?:document|curriculum|table|list)\b", low):
        if anchors:
            return anchors[0], "single_clear_document_anaphor"

    if re.search(r"\b(?:next\s+session|another\s+session|agla\s+session|naya\s+session)\b", low):
        if anchors:
            return anchors[0], "next_session_temporal_followup"

    if re.search(r"\b(?:credits?|credit numbers?|credit table|total kaise banta hai|numbers add)\b", low):
        if anchors:
            return anchors[0], "credits_calculation_followup"

    if re.search(r"\b(?:it|they|them|this|that)\b", low):
        if len(anchors) == 1:
            return anchors[0], "single_clear_pronoun"
        if len(anchors) > 1:
            return None, "ambiguous_pronoun"
        return None, "ambiguous_pronoun"

    if re.search(r"\b(?:fee waiver|family quarters|married|late registration|register \d+ days late|fine if i register|campus placement|fellowship|scholarship|stipend|financial assistance|financial support)\b", low):
        if anchors:
            return anchors[0], "domain_policy_ellipsis"


    # Permit natural discourse prefixes without making them a reference signal
    # themselves: "Okay, now what about the deadline?"
    ellipsis = re.sub(r"^(?:(?:okay|ok|now|so|then)[,\s]+)+", "", low, flags=re.I)
    if ellipsis.startswith(("what about", "how about", "when", "where", "how much", "how many")):
        if len(anchors) == 1:
            return anchors[0], "single_clear_ellipsis"
        return None, "ambiguous_ellipsis"

    if _topic_from_text(question) and len(anchors) == 1:
        if _extract_facility_candidates(question) and not _has_anaphor(question):
            return None, "facility_standalone_topic"
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
    has_named_program = bool(_extract_degree_candidates(original))

    if not messages:
        is_standalone_deictic = (not has_named_program) and bool(
            re.search(r"\b(?:this|that)\s+(?:programme|program|course|degree)\b", original, re.I)
        ) and not bool(
            re.search(r"\b(?:document|curriculum|table|structure|department|overview|elective|group|general\s+information)\b", original, re.I)
        )
        return {
            **_safe_fallback(original),
            "unanchored_reference": "true" if is_standalone_deictic else "false",
        }

    latest_question = _latest_user_question(messages)
    latest_topic = _topic_from_text(latest_question)

    is_deictic_program = (not has_named_program) and bool(
        re.search(r"\b(?:this|the|that)\s+(?:programme|program|course|degree)\b", original, re.I)
    )

    # If the user makes a deictic reference to a programme/course/degree ("this programme"),
    # verify whether a plausible degree antecedent exists in recent conversational context.
    # If the recent turns were purely departmental, research, dining, or general,
    # do NOT guess or jump back to distant historical degrees.
    if is_deictic_program:
        recent_context_text = " ".join(text for _, text in messages[-2:])
        recent_has_degree = bool(_extract_degree_candidates(recent_context_text))
        if not recent_has_degree:
            return {
                **_safe_fallback(original),
                "unanchored_reference": "true",
                "active_topic": _topic_from_text(original) or latest_topic,
            }

    # Explicit user correction (e.g., "No, I meant MBA, not M.Tech") overrides previous context.
    corr_match = re.search(
        r"\b(?:no[, ]+i meant|actually[, ]+(?:i meant|asking about)|i mean|meant)\s+([A-Za-z0-9.&'()/\-]+(?:\s+[A-Za-z0-9.&'()/\-]+)?)\b",
        original,
        flags=re.I,
    )
    if corr_match:
        corrected_raw = corr_match.group(1).strip()
        corr_degrees = [v for _, v in _extract_degree_candidates(corrected_raw)]
        corrected_entity = corr_degrees[0] if corr_degrees else corrected_raw
        attr = _extract_query_attribute(latest_question) or latest_topic or "information"
        return {
            "resolved_question": f"What is the {attr} for {corrected_entity}?",
            "mode": "follow_up",
            "active_topic": latest_topic,
            "active_entity": corrected_entity,
            "resolution_source": "explicit_correction",
            "resolution_confidence": "1.0",
            "resolution_reason": "user_explicit_correction",
        }

    # Check if the assistant recently requested clarification (e.g. asking which programme/department/topic)
    # and the user is now answering with the clarifying entity/degree.
    if messages and messages[-1][0] == "assistant":
        last_assistant_msg = messages[-1][1].casefold()
        is_clarification_prompt = any(
            phrase in last_assistant_msg
            for phrase in (
                "specify which programme",
                "specify which program",
                "which programme are you",
                "which program are you",
                "could you please specify which",
                "please specify which",
                "which department",
                "which course",
            )
        )
        if is_clarification_prompt and latest_question:
            corr_degrees = [v for _, v in _extract_degree_candidates(original)]
            clarifying_entity = corr_degrees[0] if corr_degrees else original.strip().rstrip(".?! ")
            if clarifying_entity:
                deictic_pattern = re.compile(
                    r"\b(?:this|the|that)\s+(?:programme|program|course|degree|department)\b",
                    re.I,
                )
                if deictic_pattern.search(latest_question):
                    resolved = deictic_pattern.sub(clarifying_entity, latest_question)
                else:
                    clean_pending = latest_question.rstrip("?. ")
                    resolved = f"{clean_pending} for {clarifying_entity}?"

                return {
                    "resolved_question": resolved,
                    "mode": "follow_up",
                    "active_topic": _topic_from_text(latest_question) or _topic_from_text(resolved) or latest_topic,
                    "active_entity": clarifying_entity,
                    "resolution_source": "clarification_resolution",
                    "resolution_confidence": "1.0",
                    "resolution_reason": "resolved_pending_clarification",
                }

    if not _looks_like_follow_up(original):
        orig_topic = _topic_from_text(original)
        orig_degrees = [v for _, v in _extract_degree_candidates(original)]
        return {
            **_safe_fallback(original),
            "active_entity": orig_degrees[-1] if orig_degrees else "",
            "active_topic": orig_topic,
        }

    anchors = _anchor_candidates(messages)
    latest_anchors = _anchor_candidates([("user", latest_question)]) if latest_question else []
    # Reference resolution must consider all recent explicit entities for
    # anaphora/ordinals; otherwise the latest mention can incorrectly hide a
    # second plausible antecedent (e.g. M.Tech + MBA -> "that program").
    if _has_anaphor(original) or _ordinal_index(original) is not None:
        reference_anchors = anchors
    else:
        reference_anchors = latest_anchors or anchors

    # Explicit subject means the latest question is already self-contained.
    if _explicit_subject_in_question(original, anchors) and not _has_anaphor(original):
        return {
            "resolved_question": original,
            "mode": "standalone",
            "active_topic": _topic_from_text(original),
            "active_entity": (anchors[-1] if len(anchors) == 1 else ""),
            "resolution_source": "explicit_subject",
            "resolution_confidence": "1.0",
            "resolution_reason": "latest_question_is_self_contained",
        }

    candidate, reason = _candidate_for_reference(original, reference_anchors)

    has_anaphor_in_q = _has_anaphor(original)
    is_ordinal_in_q = _ordinal_index(original) is not None
    ellipsis_match = re.match(r"^(?:what about|how about|and for|for)\s+(.+?)\??$", original, flags=re.I)
    include_match = re.match(r"^(?:does\s+(?:that|it|this)\s+include\s+(.+?))\??$", original, flags=re.I)

    if ellipsis_match and not (has_anaphor_in_q or is_ordinal_in_q):
        target_cat = ellipsis_match.group(1).strip()
        attr = _extract_query_attribute(latest_question) or latest_topic or ""
        if candidate:
            resolved = f"What is the {attr} for {target_cat} in {candidate}?" if attr else f"{original.rstrip('?. ')} in {candidate}?"
        elif latest_question:
            clean_lq = latest_question.rstrip("?. ")
            if attr:
                resolved = f"What is the {attr} for {target_cat}?"
            else:
                resolved = f"{clean_lq} for {target_cat}?"
        else:
            resolved = original

        if resolved != original:
            return {
                "resolved_question": resolved,
                "mode": "follow_up",
                "active_topic": latest_topic,
                "active_entity": candidate or target_cat,
                "resolution_source": "deterministic_category_ellipsis",
                "resolution_confidence": "0.98",
                "resolution_reason": "category_ellipsis_over_latest_turn",
            }

    # High-confidence deterministic resolution: no LLM cost.
    if candidate:
        if ellipsis_match and not (has_anaphor_in_q or is_ordinal_in_q):
            pass
        elif include_match:
            items_to_check = include_match.group(1).strip()
            attr = _extract_query_attribute(latest_question)
            topic_str = f"the {attr}" if attr else (f"the {latest_topic}" if latest_topic and latest_topic != "general" else "the fee")
            resolved = f"Does {topic_str} for {candidate} include {items_to_check}?"
        else:
            resolved = _replace_reference(original, candidate)
            if _fold(candidate) not in _fold(resolved):
                resolved = _append_subject(resolved, candidate)

        curr_topic = _topic_from_text(original)
        effective_topic = curr_topic or latest_topic
        return {
            "resolved_question": resolved,
            "mode": "follow_up",
            "active_topic": effective_topic,
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
            "active_topic": _topic_from_text(original),
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
        entity_tokens = [w for w in re.split(r"\W+", active_entity) if len(w) > 2]
        if entity_tokens and not any(w in history_blob or w in original_blob for w in entity_tokens):
            return {
                "resolved_question": original,
                "mode": "standalone",
                "active_topic": _topic_from_text(original),
                "active_entity": "",
                "resolution_source": "llm_rejected",
                "resolution_confidence": "0.0",
                "resolution_reason": "unsupported_model_entity",
            }
        if not _model_rewrite_is_supported(original, parsed["resolved_question"], messages, anchors):
            return {
                "resolved_question": original,
                "mode": "standalone",
                "active_topic": _topic_from_text(original),
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
        "active_topic": _topic_from_text(original),
        "resolution_reason": "llm_failed_or_invalid",
    }


__all__ = ["MAX_HISTORY_MESSAGES", "resolve_conversation"]
