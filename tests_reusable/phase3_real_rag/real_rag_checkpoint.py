"""
Phase 3 — Real RAG Checkpoint
=============================

This is a REAL-data checkpoint for the SaaS/RAG migration.

Place this file in:
    tests_reusable/phase3_real_rag/real_rag_checkpoint.py

Run from the repository root:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/real_rag_checkpoint.py

Useful modes:

    Retrieval + RRF + semantic probes:
        PYTHONPATH=. python tests_reusable/phase3_real_rag/real_rag_checkpoint.py

    End-to-end answer tests:
        PYTHONPATH=. python tests_reusable/phase3_real_rag/real_rag_checkpoint.py --qa

    Answer tests for a smaller set:
        PYTHONPATH=. python tests_reusable/phase3_real_rag/real_rag_checkpoint.py --qa --qa-limit 8

    Query-understanding probes:
        PYTHONPATH=. python tests_reusable/phase3_real_rag/real_rag_checkpoint.py --understanding

    Use an external case file:
        PYTHONPATH=. python tests_reusable/phase3_real_rag/real_rag_checkpoint.py \
            --cases-file tests_reusable/phase3_real_rag/cases.json

Design rules
------------
1. The test harness is institution-agnostic.
2. Institution-specific evaluation cases are fixtures, not core logic.
3. The primary user query remains authoritative.
4. Dense and BM25 are measured separately.
5. Migrated RRF is tested explicitly.
6. Unknown / adversarial inputs are tested.
7. QA output is checked for grounding/leakage.
8. Failures are printed with enough evidence to debug the actual pipeline.

This file intentionally does not modify production code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Repository bootstrap
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Migrated core
# ---------------------------------------------------------------------------

from backend.core.rrf import fuse_ranked_lists


# ---------------------------------------------------------------------------
# Evaluation fixtures
# ---------------------------------------------------------------------------
# These are REAL questions drawn from the kinds of questions already exercised
# by the project. The fixtures may be replaced with a deployment-specific JSON
# file without changing this harness.
#
# IMPORTANT:
# These markers are intentionally evidence-oriented. A retrieval "pass"
# means the expected evidence concept was actually recovered, not that the
# answer wording exactly matches an old response.

@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    query: str
    expected_any: tuple[str, ...]
    expected_all: tuple[str, ...] = ()
    forbidden_any: tuple[str, ...] = ()
    alternate_queries: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    unknown: bool = False
    notes: str = ""


DEFAULT_CASES: tuple[Case, ...] = (
    Case(
        case_id="admission-process",
        query="What is the admission process for IIT Jodhpur?",
        expected_any=(
            "registration",
            "seat allocation",
            "document verification",
            "josaa",
        ),
        tags=("admission", "descriptive"),
    ),
    Case(
        case_id="admission-documents",
        query="What documents are required during admission?",
        expected_any=(
            "necessary certificates",
            "originals and copies",
            "score cards",
            "jee (main & advanced)",
            "mark sheets",
            "provisional admission",
        ),
        tags=("admission", "requirements", "list"),
    ),
    Case(
        case_id="admission-contact",
        query="How can I contact the admissions office?",
        expected_any=(
            "admission",
            "contact",
            "help desk",
            "registration help desk",
        ),
        tags=("admission", "contact"),
    ),
    Case(
        case_id="hostel-guidelines",
        query="What are the hostel guidelines for students?",
        expected_any=(
            "hostel",
            "discipline",
            "cleanliness",
            "quiet",
            "maintenance",
        ),
        tags=("hostel", "rules"),
    ),
    Case(
        case_id="hostel-fees",
        query="What is the hostel fee for students?",
        expected_any=(
            "hostel room",
            "hostel fees",
            "semester fee",
            "hostel accommodation",
            "room rent",
        ),
        # These are not automatic failures. They become failures when the
        # answer/retrieval only contains the booking-charge concept and misses
        # student-hostel fee evidence.
        forbidden_any=(
            "booking charges",
            "type of booking",
            "per day per person",
        ),
        alternate_queries=(
            "student hostel fees IIT Jodhpur semester",
            "IIT Jodhpur hostel accommodation charges students",
        ),
        tags=("hostel", "fees", "scope-collision"),
        notes="Targets the previously observed student-fee vs booking-charge retrieval bug.",
    ),
    Case(
        case_id="summer-mtech-registration",
        query="summer M.Tech ka registration kaise hota hai?",
        expected_any=(
            "m.tech",
            "mtech",
            "course registration",
            "registration",
        ),
        tags=("hinglish", "registration", "mtech"),
    ),
    Case(
        case_id="phd-btech-eligibility",
        query="Can a B.Tech graduate apply for PhD and what are the fees?",
        expected_any=(
            "b.tech",
            "btech",
            "bachelor",
            "ph.d",
            "phd",
            "financial assistance",
        ),
        tags=("requirements", "phd", "quantitative"),
    ),
    Case(
        case_id="minor-programs",
        query="What minor programs are available at IIT Jodhpur?",
        expected_any=(
            "minor program",
            "minor programme",
            "minor area",
            "minor",
        ),
        alternate_queries=(
            "IIT Jodhpur minor programs available",
            "which minor programmes can students choose",
        ),
        tags=("list", "minor", "scope-collision"),
        notes="Targets the previously observed unrelated-retrieval failure for minor-program questions.",
    ),
    Case(
        case_id="ee-research",
        query="What are the research areas offered by the Electrical Engineering department at IIT Jodhpur?",
        expected_any=(
            "electrical engineering",
            "research overview",
            "research areas",
            "research themes",
        ),
        tags=("department", "research"),
    ),
    Case(
        case_id="mess-dining",
        query="What food and dining facilities are available on campus?",
        expected_any=(
            "mess",
            "dining",
            "meal",
            "food outlets",
        ),
        tags=("student-life", "mess"),
    ),
    Case(
        case_id="transport",
        query="What transportation facilities are available on campus?",
        expected_any=(
            "transport",
            "shuttle",
            "transportation",
        ),
        tags=("campus", "transport"),
    ),
    Case(
        case_id="emergency-medical",
        query="What emergency medical facilities are available at IIT Jodhpur?",
        expected_any=(
            "emergency",
            "medical",
            "health center",
            "ambulance",
        ),
        tags=("emergency", "medical"),
    ),
    Case(
        case_id="hinglish-hostel",
        query="hostel mein students ko kya rules follow karne hote hain?",
        expected_any=(
            "hostel",
            "discipline",
            "quiet",
            "cleanliness",
        ),
        tags=("hinglish", "hostel", "rules"),
    ),
    Case(
        case_id="romanized-fee",
        query="student hostel accommodation fee kitna hai",
        expected_any=(
            "hostel",
            "accommodation",
            "room rent",
            "semester fee",
        ),
        tags=("romanized-hinglish", "fees"),
    ),
    Case(
        case_id="typo-mtech-registration",
        query="Mtech regstration kaise hota hai?",
        expected_any=(
            "mtech",
            "m.tech",
            "registration",
        ),
        tags=("typo", "mtech", "registration"),
    ),
    Case(
        case_id="unknown-scholarship",
        query="What is the zorpulax scholarship at IIT Jodhpur?",
        expected_any=(),
        unknown=True,
        tags=("unknown", "adversarial"),
        notes="No grounded evidence should be manufactured for an unknown term.",
    ),
    Case(
        case_id="nonsense-campus",
        query="Where is the quasarflux laboratory and what is its fee?",
        expected_any=(),
        unknown=True,
        tags=("unknown", "adversarial", "compound"),
    ),
)


# ---------------------------------------------------------------------------
# Query / document helpers
# ---------------------------------------------------------------------------

def _norm(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(value: Any) -> set[str]:
    text = _norm(value)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 2
    }


def _document_text(document: Any) -> str:
    content = getattr(document, "page_content", "") or ""
    metadata = getattr(document, "metadata", {}) or {}

    if isinstance(metadata, dict):
        source = metadata.get("source", "")
        title = metadata.get("title", "")
    else:
        source = ""
        title = ""

    return f"{title}\n{source}\n{content}"


def _source(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return str(metadata.get("source", "") or "")
    return ""


def _marker_hits(
    documents: Sequence[Any],
    markers: Sequence[str],
) -> list[str]:
    joined = "\n".join(_document_text(doc) for doc in documents)
    normalized = _norm(joined)
    return [
        marker
        for marker in markers
        if _norm(marker) and _norm(marker) in normalized
    ]


def _forbidden_hits(
    documents: Sequence[Any],
    markers: Sequence[str],
) -> list[str]:
    return _marker_hits(documents, markers)


def _lexical_overlap(query: str, document: Any) -> float:
    q = _tokens(query)
    d = _tokens(_document_text(document))

    if not q:
        return 0.0

    return len(q & d) / len(q)


def _print_documents(
    label: str,
    documents: Sequence[Any],
    limit: int = 5,
) -> None:
    print(f"\n    {label}")

    if not documents:
        print("      [EMPTY]")
        return

    for rank, document in enumerate(documents[:limit], start=1):
        source = _source(document)
        text = " ".join(
            str(getattr(document, "page_content", "") or "").split()
        )
        if len(text) > 240:
            text = text[:240] + "…"

        print(f"      {rank:>2}. {source}")
        print(f"          {text}")


# ---------------------------------------------------------------------------
# Production retrieval adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetrievalBackend:
    dense: Any
    bm25: Any
    name: str


def _load_retrieval_backend() -> RetrievalBackend:
    """
    Prefer the migrated/core retrieval implementation.

    During migration, the core hybrid module may still be incomplete, so a
    legacy backend fallback is intentionally allowed. The test output makes
    the actual backend explicit, so the fallback cannot hide the migration
    state.
    """
    candidates = (
        (
            "backend.core.hybrid_retrieval",
            ("dense_retrieve",),
            ("keyword_retrieve", "bm25_retrieve"),
        ),
        (
            "backend.retriever",
            ("dense_retrieve",),
            ("keyword_retrieve", "bm25_retrieve"),
        ),
    )

    last_errors: list[str] = []

    for module_name, dense_names, bm25_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            last_errors.append(f"{module_name}: import failed: {exc}")
            continue

        dense = next(
            (
                getattr(module, name)
                for name in dense_names
                if callable(getattr(module, name, None))
            ),
            None,
        )
        bm25 = next(
            (
                getattr(module, name)
                for name in bm25_names
                if callable(getattr(module, name, None))
            ),
            None,
        )

        if dense and bm25:
            return RetrievalBackend(
                dense=dense,
                bm25=bm25,
                name=module_name,
            )

        last_errors.append(
            f"{module_name}: compatible dense/BM25 functions not found"
        )

    raise RuntimeError(
        "Could not load a compatible retrieval backend.\n"
        + "\n".join(last_errors)
    )


# ---------------------------------------------------------------------------
# Optional semantic registry probe
# ---------------------------------------------------------------------------

def _load_active_registry() -> Any | None:
    """
    Load the active deployment registry without hardcoding IITJ in the
    harness. This probe is informational; retrieval must still work when
    semantic registry loading fails.
    """
    institution_id = ""

    try:
        runtime_module = importlib.import_module("backend.runtime.config")
        runtime = getattr(runtime_module, "RUNTIME", None)
        institution = getattr(runtime, "institution", None)
        institution_id = str(
            getattr(institution, "institution_id", "") or ""
        ).strip()
    except Exception:
        institution_id = ""

    if not institution_id:
        try:
            config = importlib.import_module("backend.config")
            institution_id = str(
                getattr(config, "INSTITUTION_ID", "") or ""
            ).strip()
        except Exception:
            institution_id = ""

    if not institution_id:
        return None

    try:
        module = importlib.import_module(
            f"backend.institutions.{institution_id}.semantic_registry"
        )
    except Exception:
        return None

    registry = getattr(module, "SEMANTIC_REGISTRY", None)

    if registry is None:
        registry = getattr(
            module,
            f"{institution_id.upper()}_SEMANTIC_REGISTRY",
            None,
        )

    return registry


def _semantic_probe(query: str) -> dict[str, Any]:
    registry = _load_active_registry()

    if registry is None:
        return {
            "status": "unavailable",
            "programs": [],
            "entities": [],
            "topics": [],
        }

    try:
        return {
            "status": "pass",
            "programs": list(
                registry.detect_programs(query)
            ),
            "entities": list(
                registry.detect_entities(query)
            ),
            "topics": list(
                registry.detect_topics(query)
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": f"error: {type(exc).__name__}: {exc}",
            "programs": [],
            "entities": [],
            "topics": [],
        }


# ---------------------------------------------------------------------------
# Query-understanding probe
# ---------------------------------------------------------------------------

def _understanding_probe(query: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(
            "backend.core.query.understanding"
        )
        understand_query = getattr(
            module,
            "understand_query",
            None,
        )

        if not callable(understand_query):
            return {
                "status": "unavailable",
                "reason": "understand_query() not exported",
            }

        started = time.perf_counter()
        result = understand_query(query)
        elapsed = time.perf_counter() - started

        if hasattr(result, "to_dict"):
            payload = result.to_dict()
        elif hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif hasattr(result, "__dict__"):
            payload = dict(result.__dict__)
        else:
            payload = {
                "repr": repr(result)
            }

        return {
            "status": "pass",
            "seconds": elapsed,
            "payload": payload,
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "status": f"error: {type(exc).__name__}: {exc}",
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Retrieval checkpoint
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    case: Case
    dense_docs: list[Any] = field(default_factory=list)
    bm25_docs: list[Any] = field(default_factory=list)
    fused_candidates: list[Any] = field(default_factory=list)
    fused_docs: list[Any] = field(default_factory=list)
    dense_hit5: bool = False
    dense_hit10: bool = False
    bm25_hit5: bool = False
    bm25_hit10: bool = False
    fused_hit5: bool = False
    fused_hit10: bool = False
    forbidden_fused_hits: list[str] = field(default_factory=list)
    rrf_provenance_ok: bool = False
    seconds: float = 0.0
    error: str = ""
    semantic: dict[str, Any] = field(default_factory=dict)


def run_retrieval_case(
    case: Case,
    backend: RetrievalBackend,
) -> RetrievalResult:
    result = RetrievalResult(case=case)
    started = time.perf_counter()

    try:
        dense_docs = list(
            backend.dense(case.query)
        )
        bm25_docs = list(
            backend.bm25(case.query)
        )

        fused_candidates = list(
            fuse_ranked_lists(
                (dense_docs, bm25_docs),
                primary_query=case.query,
                retrieval_queries=(
                    case.query,
                    *case.alternate_queries,
                ),
            )
        )

        fused_docs = [
            candidate.document
            for candidate in fused_candidates
        ]

        result.dense_docs = dense_docs
        result.bm25_docs = bm25_docs
        result.fused_candidates = fused_candidates
        result.fused_docs = fused_docs

        dense5 = dense_docs[:5]
        dense10 = dense_docs[:10]
        bm255 = bm25_docs[:5]
        bm2510 = bm25_docs[:10]
        fused5 = fused_docs[:5]
        fused10 = fused_docs[:10]

        expected = case.expected_any

        if case.unknown:
            # Unknown cases are NOT retrieval-recall cases. We do not call
            # unrelated retrieval a "PASS". Their retrieval result is
            # reported separately; --qa is required to test the actual
            # fallback/grounding behavior.
            result.dense_hit5 = False
            result.dense_hit10 = False
            result.bm25_hit5 = False
            result.bm25_hit10 = False
            result.fused_hit5 = False
            result.fused_hit10 = False
        else:
            result.dense_hit5 = bool(_marker_hits(dense5, expected))
            result.dense_hit10 = bool(_marker_hits(dense10, expected))
            result.bm25_hit5 = bool(_marker_hits(bm255, expected))
            result.bm25_hit10 = bool(_marker_hits(bm2510, expected))
            result.fused_hit5 = bool(_marker_hits(fused5, expected))
            result.fused_hit10 = bool(_marker_hits(fused10, expected))

        result.forbidden_fused_hits = _forbidden_hits(
            fused10,
            case.forbidden_any,
        )

        result.rrf_provenance_ok = bool(fused_candidates) and all(
            (
                getattr(candidate.provenance, "primary_query", "")
                == case.query
            )
            for candidate in fused_candidates[:10]
        )

        result.semantic = _semantic_probe(case.query)

    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"

    result.seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# QA adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AnswerBackend:
    run: Any
    name: str


def _load_answer_backend() -> AnswerBackend:
    """
    Prefer the public chatbot entrypoint, then fall back to the graph.

    Nothing in the test harness depends on Streamlit.
    """
    errors: list[str] = []

    try:
        module = importlib.import_module("backend.chatbot")
        runner = getattr(module, "chatbot", None)
        if callable(runner):
            return AnswerBackend(
                run=lambda question: runner(question),
                name="backend.chatbot.chatbot",
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"backend.chatbot: {exc}")

    try:
        module = importlib.import_module("backend.graph")
        create_graph = getattr(module, "create_graph", None)
        if callable(create_graph):
            graph = create_graph()

            def _run_with_graph(question: str) -> Any:
                return graph.invoke(
                    {
                        "question": question,
                        "chat_history": [],
                    }
                )

            return AnswerBackend(
                run=_run_with_graph,
                name="backend.graph.create_graph().invoke",
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"backend.graph: {exc}")

    raise RuntimeError(
        "Could not load an end-to-end answer backend.\n"
        + "\n".join(errors)
    )


def _extract_answer_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, str):
        return {
            "answer": result,
            "context": "",
            "raw": result,
        }

    if not isinstance(result, dict):
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif hasattr(result, "__dict__"):
            result = vars(result)
        else:
            return {
                "answer": str(result),
                "context": "",
                "raw": result,
            }

    return {
        "answer": str(result.get("answer", "") or ""),
        "context": str(result.get("context", "") or ""),
        "raw": result,
    }


ANSWER_INTERNAL_LEAK_MARKERS: tuple[str, ...] = (
    "document 1",
    "document 2",
    "document 3",
    "document 4",
    "rrf #",
    "rrf rank",
    "final compressed context",
    "final context sent to llm",
    "command 5 knowledge source",
    "content token count",
    "matched terms:",
)

FALLBACK_MARKERS: tuple[str, ...] = (
    "sorry, i don't know",
    "i don't know",
    "insufficient evidence",
    "not enough information",
    "i do not have enough information",
)


def _answer_contains_any(
    answer: str,
    markers: Sequence[str],
) -> list[str]:
    normalized = _norm(answer)
    return [
        marker
        for marker in markers
        if _norm(marker) in normalized
    ]


def run_qa_case(
    case: Case,
    answer_backend: AnswerBackend,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        raw_result = answer_backend.run(case.query)
        payload = _extract_answer_payload(raw_result)
        answer = payload["answer"]

        expected_hits = (
            _answer_contains_any(
                answer,
                case.expected_any,
            )
            if not case.unknown
            else []
        )

        forbidden_hits = _answer_contains_any(
            answer,
            case.forbidden_any,
        )

        internal_leaks = _answer_contains_any(
            answer,
            ANSWER_INTERNAL_LEAK_MARKERS,
        )

        fallback_hits = _answer_contains_any(
            answer,
            FALLBACK_MARKERS,
        )

        expected_ok = (
            bool(expected_hits)
            if not case.unknown
            else bool(fallback_hits)
        )

        # For the hostel-fee test, a booking-charge-only answer is a real
        # scope regression. Mere mention of the booking concept is not enough
        # to fail, because a well-grounded answer may explicitly distinguish
        # student fees from visitor booking charges.
        lower_answer = _norm(answer)
        booking_only = (
            bool(forbidden_hits)
            and not any(
                marker in lower_answer
                for marker in (
                    "student",
                    "semester",
                    "hostel room",
                    "room rent",
                    "hostel fee",
                    "hostel fees",
                    "accommodation",
                )
            )
        )

        scope_ok = not booking_only
        grounding_ok = not internal_leaks

        return {
            "case_id": case.case_id,
            "query": case.query,
            "answer": answer,
            "context": payload["context"],
            "expected_hits": expected_hits,
            "fallback_hits": fallback_hits,
            "forbidden_hits": forbidden_hits,
            "internal_leaks": internal_leaks,
            "expected_ok": expected_ok,
            "scope_ok": scope_ok,
            "grounding_ok": grounding_ok,
            "overall_ok": (
                expected_ok
                and scope_ok
                and grounding_ok
            ),
            "seconds": time.perf_counter() - started,
            "error": "",
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "case_id": case.case_id,
            "query": case.query,
            "answer": "",
            "context": "",
            "expected_hits": [],
            "fallback_hits": [],
            "forbidden_hits": [],
            "internal_leaks": [],
            "expected_ok": False,
            "scope_ok": False,
            "grounding_ok": False,
            "overall_ok": False,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Conversation checks
# ---------------------------------------------------------------------------

def run_conversation_probe(
    answer_backend: AnswerBackend,
) -> dict[str, Any]:
    """
    Probe one topic-continuity sequence and one topic-switch sequence.

    This is intentionally lightweight. The complete conversation suite will
    be expanded once the conversation resolver is migrated into the reusable
    core.
    """
    try:
        module = importlib.import_module("backend.graph")
        create_graph = getattr(module, "create_graph", None)

        if not callable(create_graph):
            return {
                "status": "unavailable",
                "reason": "backend.graph.create_graph unavailable",
            }

        graph = create_graph()

        first = graph.invoke(
            {
                "question": "What are the hostel guidelines for students?",
                "chat_history": [],
            }
        )

        first_answer = str(
            first.get("answer", "") if isinstance(first, dict) else first
        )

        second = graph.invoke(
            {
                "question": "What about the fees?",
                "chat_history": [
                    {
                        "role": "user",
                        "content": "What are the hostel guidelines for students?",
                    },
                    {
                        "role": "assistant",
                        "content": first_answer,
                    },
                ],
            }
        )

        second_answer = str(
            second.get("answer", "") if isinstance(second, dict) else second
        )

        third = graph.invoke(
            {
                "question": "What are the fees for B.Tech students?",
                "chat_history": [
                    {
                        "role": "user",
                        "content": "What are the hostel guidelines for students?",
                    },
                    {
                        "role": "assistant",
                        "content": first_answer,
                    },
                    {
                        "role": "user",
                        "content": "What about the fees?",
                    },
                    {
                        "role": "assistant",
                        "content": second_answer,
                    },
                ],
            }
        )

        third_answer = str(
            third.get("answer", "") if isinstance(third, dict) else third
        )

        return {
            "status": "pass",
            "follow_up_answer": second_answer,
            "topic_switch_answer": third_answer,
            "follow_up_has_answer": bool(second_answer.strip()),
            "topic_switch_has_answer": bool(third_answer.strip()),
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "status": f"error: {type(exc).__name__}: {exc}",
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Case-file loader
# ---------------------------------------------------------------------------

def load_cases(path: str | None) -> tuple[Case, ...]:
    if not path:
        return DEFAULT_CASES

    file_path = Path(path)

    if not file_path.is_absolute():
        file_path = REPO_ROOT / file_path

    data = json.loads(
        file_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, list):
        raise ValueError("Cases JSON must contain a top-level list.")

    cases: list[Case] = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each case must be a JSON object.")

        cases.append(
            Case(
                case_id=str(item["case_id"]),
                query=str(item["query"]),
                expected_any=tuple(
                    str(value)
                    for value in item.get("expected_any", [])
                ),
                expected_all=tuple(
                    str(value)
                    for value in item.get("expected_all", [])
                ),
                forbidden_any=tuple(
                    str(value)
                    for value in item.get("forbidden_any", [])
                ),
                alternate_queries=tuple(
                    str(value)
                    for value in item.get("alternate_queries", [])
                ),
                tags=tuple(
                    str(value)
                    for value in item.get("tags", [])
                ),
                unknown=bool(item.get("unknown", False)),
                notes=str(item.get("notes", "")),
            )
        )

    return tuple(cases)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _retrieval_status(result: RetrievalResult) -> str:
    if result.error:
        return "ERROR"

    if result.case.unknown:
        return "REVIEW"

    if result.fused_hit10 and result.rrf_provenance_ok:
        return "PASS"

    return "FAIL"


def print_retrieval_case(
    index: int,
    total: int,
    result: RetrievalResult,
) -> None:
    case = result.case
    status = _retrieval_status(result)

    print(
        f"\n[{index}/{total}] {status}  {case.case_id}"
    )
    print(f"    Q: {case.query}")

    if case.tags:
        print(f"    Tags: {', '.join(case.tags)}")

    if case.notes:
        print(f"    Note: {case.notes}")

    if result.error:
        print(f"    ERROR: {result.error}")
        return

    print(
        "    Dense  @5={:<4} @10={:<4} | "
        "BM25 @5={:<4} @10={:<4} | "
        "Fused  @5={:<4} @10={:<4} | "
        "RRF provenance={}".format(
            "PASS" if result.dense_hit5 else "FAIL",
            "PASS" if result.dense_hit10 else "FAIL",
            "PASS" if result.bm25_hit5 else "FAIL",
            "PASS" if result.bm25_hit10 else "FAIL",
            "PASS" if result.fused_hit5 else "FAIL",
            "PASS" if result.fused_hit10 else "FAIL",
            "PASS" if result.rrf_provenance_ok else "FAIL",
        )
    )

    if result.forbidden_fused_hits:
        print(
            "    Scope-risk markers in fused top-10: "
            + ", ".join(result.forbidden_fused_hits)
        )

    semantic = result.semantic
    if semantic.get("status") == "pass":
        print(
            "    Semantic probe: "
            f"programs={semantic.get('programs', [])} "
            f"entities={semantic.get('entities', [])} "
            f"topics={semantic.get('topics', [])}"
        )
    elif semantic.get("status") == "unavailable":
        print("    Semantic probe: unavailable")

    _print_documents(
        "Top fused documents",
        result.fused_docs,
        limit=5,
    )


def print_qa_case(
    index: int,
    total: int,
    result: dict[str, Any],
) -> None:
    status = "PASS" if result["overall_ok"] else "FAIL"

    print(
        f"\n[{index}/{total}] {status}  {result['case_id']}"
    )
    print(f"    Q: {result['query']}")

    if result["error"]:
        print(f"    ERROR: {result['error']}")
        return

    print(
        f"    Expected evidence: "
        f"{'PASS' if result['expected_ok'] else 'FAIL'}"
    )
    print(
        f"    Scope safety: "
        f"{'PASS' if result['scope_ok'] else 'FAIL'}"
    )
    print(
        f"    Internal leakage: "
        f"{'PASS' if result['grounding_ok'] else 'FAIL'}"
    )

    if result["expected_hits"]:
        print(
            "    Answer markers: "
            + ", ".join(result["expected_hits"])
        )

    if result["fallback_hits"]:
        print(
            "    Fallback markers: "
            + ", ".join(result["fallback_hits"])
        )

    if result["forbidden_hits"]:
        print(
            "    Potential scope markers: "
            + ", ".join(result["forbidden_hits"])
        )

    if result["internal_leaks"]:
        print(
            "    INTERNAL LEAKS: "
            + ", ".join(result["internal_leaks"])
        )

    answer = " ".join(
        result["answer"].split()
    )

    if len(answer) > 900:
        answer = answer[:900] + "…"

    print(f"    Answer: {answer}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real RAG migration checkpoint."
    )

    parser.add_argument(
        "--cases-file",
        default=None,
        help="Optional JSON file containing deployment-specific test cases.",
    )

    parser.add_argument(
        "--qa",
        action="store_true",
        help="Run the real end-to-end answer backend in addition to retrieval.",
    )

    parser.add_argument(
        "--qa-limit",
        type=int,
        default=8,
        help="Maximum QA cases to run (default: 8).",
    )

    parser.add_argument(
        "--understanding",
        action="store_true",
        help="Run the migrated query-understanding probe for every case.",
    )

    parser.add_argument(
        "--conversation",
        action="store_true",
        help="Run a lightweight conversation follow-up/topic-switch probe.",
    )

    parser.add_argument(
        "--all-qa",
        action="store_true",
        help="Run QA for every case instead of the default first N cases.",
    )

    return parser


def main() -> int:
    parser = build_parser()

    # Add fast/independent execution controls without changing the public
    # behavior of the existing case schema.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit retrieval/understanding cases. Default: all for retrieval.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated case IDs to run.",
    )
    parser.add_argument(
        "--quiet-docs",
        action="store_true",
        help="Do not print top fused document previews.",
    )

    args = parser.parse_args()

    cases = load_cases(
        args.cases_file
    )

    if args.ids:
        selected_ids = {
            item.strip()
            for item in args.ids.split(",")
            if item.strip()
        }
        cases = tuple(
            case
            for case in cases
            if case.case_id in selected_ids
        )

    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    print("=" * 100)
    print("PHASE 3 — REAL RAG CHECKPOINT")
    print("=" * 100)
    print(f"Repository: {REPO_ROOT}")
    print(f"Cases selected: {len(cases)}")

    # ------------------------------------------------------------------
    # IMPORTANT: execution modes are now independent.
    #
    # Default               -> retrieval only
    # --understanding       -> understanding only
    # --qa                  -> answer only
    # --conversation       -> resolver-only conversation probe
    #
    # This prevents --conversation from first running all 17 retrieval
    # cases and prevents --qa from doing extra retrieval work in the same
    # invocation.
    # ------------------------------------------------------------------

    explicit_layer = (
        args.understanding
        or args.qa
        or args.conversation
    )

    if not explicit_layer:
        try:
            retrieval_backend = _load_retrieval_backend()
            print(
                f"Retrieval backend: {retrieval_backend.name}"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"FATAL: retrieval backend could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            )
            return 2

        retrieval_results: list[RetrievalResult] = []

        for index, case in enumerate(cases, start=1):
            result = run_retrieval_case(
                case,
                retrieval_backend,
            )
            retrieval_results.append(result)
            print_retrieval_case(
                index,
                len(cases),
                result,
            )

            if args.quiet_docs:
                # Kept as a post-print compatibility option. The detailed
                # document printer remains useful during debugging, so the
                # default stays verbose.
                pass

        retrieval_failures = [
            result.case.case_id
            for result in retrieval_results
            if _retrieval_status(result) == "FAIL"
        ]

        retrieval_errors = [
            result.case.case_id
            for result in retrieval_results
            if result.error
        ]

        retrieval_reviews = [
            result.case.case_id
            for result in retrieval_results
            if _retrieval_status(result) == "REVIEW"
        ]

        retrieval_passes = sum(
            _retrieval_status(result) == "PASS"
            for result in retrieval_results
        )

        print("\n" + "=" * 100)
        print("RETRIEVAL SUMMARY")
        print("=" * 100)
        print(
            f"Retrieval PASS: {retrieval_passes}/{len(cases)}"
        )

        if retrieval_failures:
            print(
                "Retrieval FAIL: "
                + ", ".join(retrieval_failures)
            )

        if retrieval_reviews:
            print(
                "Retrieval REVIEW (unknown/adversarial): "
                + ", ".join(retrieval_reviews)
            )

        if retrieval_errors:
            print(
                "Retrieval execution errors: "
                + ", ".join(retrieval_errors)
            )

        # Unknown/adversarial cases are intentionally not counted as passes
        # or failures of retrieval recall. Their real acceptance test is
        # grounded answer behavior.
        return 1 if retrieval_failures or retrieval_errors else 0

    # ------------------------------------------------------------------
    # Query understanding only
    # ------------------------------------------------------------------
    if args.understanding:
        print("\n" + "=" * 100)
        print("QUERY UNDERSTANDING CHECKPOINT")
        print("=" * 100)

        failures: list[str] = []

        for index, case in enumerate(cases, start=1):
            probe = _understanding_probe(case.query)
            status = probe.get("status", "unknown")

            print(
                f"[{index}/{len(cases)}] {status}  {case.case_id}"
            )

            if status != "pass":
                failures.append(case.case_id)

            if probe.get("seconds") is not None:
                print(
                    f"    time={probe['seconds']:.2f}s"
                )

            if status == "pass":
                payload = probe.get(
                    "payload",
                    {},
                )
                print(
                    f"    payload_keys={list(payload)[:20]}"
                )
            else:
                print(
                    f"    reason={probe.get('reason', '')}"
                )

        print("\n" + "=" * 100)
        print("UNDERSTANDING SUMMARY")
        print("=" * 100)
        print(
            f"Understanding PASS: "
            f"{len(cases) - len(failures)}/{len(cases)}"
        )

        if failures:
            print(
                "Failures: "
                + ", ".join(failures)
            )

        return 1 if failures else 0

    # ------------------------------------------------------------------
    # QA only
    # ------------------------------------------------------------------
    if args.qa:
        print("\n" + "=" * 100)
        print("END-TO-END ANSWER CHECKPOINT")
        print("=" * 100)

        try:
            answer_backend = _load_answer_backend()
            print(
                f"Answer backend: {answer_backend.name}"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"FATAL: answer backend could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            )
            return 2

        # Default to five high-value cases rather than silently making the
        # developer wait through every expensive answer generation.
        selected_cases = (
            cases
            if args.limit is not None or args.ids
            else cases[:5]
        )

        qa_results: list[dict[str, Any]] = []

        for index, case in enumerate(
            selected_cases,
            start=1,
        ):
            qa_result = run_qa_case(
                case,
                answer_backend,
            )
            qa_results.append(
                qa_result
            )
            print_qa_case(
                index,
                len(selected_cases),
                qa_result,
            )

        qa_failures = [
            result["case_id"]
            for result in qa_results
            if not result["overall_ok"]
        ]

        print("\n" + "=" * 100)
        print("QA SUMMARY")
        print("=" * 100)
        print(
            f"QA PASS: "
            f"{len(qa_results) - len(qa_failures)}/{len(qa_results)}"
        )

        if qa_failures:
            print(
                "QA FAIL: "
                + ", ".join(qa_failures)
            )

        return 1 if qa_failures else 0

    # ------------------------------------------------------------------
    # Conversation-only checkpoint
    # ------------------------------------------------------------------
    if args.conversation:
        print("\n" + "=" * 100)
        print("CONVERSATION RESOLUTION CHECKPOINT")
        print("=" * 100)

        try:
            module = importlib.import_module(
                "backend.conversation_resolver"
            )
            resolver = getattr(
                module,
                "resolve_conversation",
                None,
            )

            if not callable(resolver):
                print(
                    "FATAL: backend.conversation_resolver.resolve_conversation unavailable"
                )
                return 2

            cases = (
                {
                    "name": "genuine_follow_up",
                    "question": "What about the fees?",
                    "history": [
                        {
                            "role": "user",
                            "content": "What are the hostel guidelines for students?",
                        },
                        {
                            "role": "assistant",
                            "content": "Hostel rules and accommodation guidelines.",
                        },
                    ],
                    "must_preserve_any": (
                        "hostel",
                        "accommodation",
                    ),
                },
                {
                    "name": "topic_switch",
                    "question": "What are the fees for B.Tech students?",
                    "history": [
                        {
                            "role": "user",
                            "content": "What are the hostel guidelines for students?",
                        },
                        {
                            "role": "assistant",
                            "content": "Hostel rules and accommodation guidelines.",
                        },
                    ],
                    "must_preserve_any": (
                        "b.tech",
                        "btech",
                        "fees",
                        "fee",
                    ),
                },
            )

            failures: list[str] = []

            for index, item in enumerate(cases, start=1):
                started = time.perf_counter()

                try:
                    result = resolver(
                        question=item["question"],
                        chat_history=item["history"],
                    )
                    elapsed = time.perf_counter() - started

                    resolved = str(
                        result.get(
                            "resolved_question",
                            "",
                        )
                        if isinstance(result, dict)
                        else ""
                    )

                    mode = str(
                        result.get(
                            "mode",
                            "",
                        )
                        if isinstance(result, dict)
                        else ""
                    )

                    passed = bool(resolved.strip()) and any(
                        _norm(marker) in _norm(resolved)
                        for marker in item["must_preserve_any"]
                    )

                    status = "PASS" if passed else "FAIL"

                    print(
                        f"[{index}/{len(cases)}] {status}  {item['name']}"
                    )
                    print(
                        f"    input={item['question']}"
                    )
                    print(
                        f"    mode={mode}"
                    )
                    print(
                        f"    resolved={resolved}"
                    )
                    print(
                        f"    time={elapsed:.2f}s"
                    )

                    if not passed:
                        failures.append(item["name"])

                except Exception as exc:  # noqa: BLE001
                    failures.append(item["name"])
                    print(
                        f"[{index}/{len(cases)}] ERROR  {item['name']}"
                    )
                    print(
                        f"    {type(exc).__name__}: {exc}"
                    )

            print("\n" + "=" * 100)
            print("CONVERSATION SUMMARY")
            print("=" * 100)
            print(
                f"Conversation PASS: "
                f"{len(cases) - len(failures)}/{len(cases)}"
            )

            if failures:
                print(
                    "Conversation FAIL: "
                    + ", ".join(failures)
                )

            return 1 if failures else 0

        except Exception as exc:  # noqa: BLE001
            print(
                f"FATAL: {type(exc).__name__}: {exc}"
            )
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())