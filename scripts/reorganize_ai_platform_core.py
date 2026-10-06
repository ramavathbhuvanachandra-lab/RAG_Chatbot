#!/usr/bin/env python3
"""Reorganize the existing ai_platform core without changing its behavior.

This migration deliberately keeps the current core implementations intact.
It only:
  1. moves root-level modules into semantic subpackages;
  2. rewrites imports/importlib targets inside ai_platform/;
  3. creates package __init__.py files;
  4. refuses to overwrite conflicting destination files.

It never reads, imports, edits, or deletes backend/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "ai_platform"
CORE = PLATFORM / "core"

MOVE_MAP = {
    CORE / "conversation.py": CORE / "conversation" / "conversation.py",
    CORE / "multi_intent.py": CORE / "conversation" / "multi_intent.py",
    CORE / "nodes.py": CORE / "graph" / "nodes.py",
    CORE / "institution.py": CORE / "institution" / "profile.py",
    CORE / "query_frame.py": CORE / "query" / "frame.py",
    CORE / "query_interpreter.py": CORE / "query" / "interpreter.py",
    CORE / "query_pipeline.py": CORE / "query" / "pipeline.py",
    CORE / "retrieval_contracts.py": CORE / "retrieval" / "contracts.py",
    CORE / "retrieval_query.py": CORE / "retrieval" / "query.py",
    CORE / "hybrid_retrieval.py": CORE / "retrieval" / "hybrid.py",
    CORE / "rrf.py": CORE / "retrieval" / "rrf.py",
    CORE / "semantic_alignment.py": CORE / "retrieval" / "semantic_alignment.py",
    CORE / "semantic_registry.py": CORE / "retrieval" / "semantic_registry.py",
    CORE / "candidate_verification.py": CORE / "retrieval" / "verification.py",
}

IMPORT_MAP = {
    # Old ai_platform paths -> final ai_platform paths
    "ai_platform.core.conversation": "ai_platform.core.conversation.conversation",
    "ai_platform.core.multi_intent": "ai_platform.core.conversation.multi_intent",
    "ai_platform.core.nodes": "ai_platform.core.graph.nodes",
    "ai_platform.core.institution": "ai_platform.core.institution.profile",
    "ai_platform.core.query_frame": "ai_platform.core.query.frame",
    "ai_platform.core.query_interpreter": "ai_platform.core.query.interpreter",
    "ai_platform.core.query_pipeline": "ai_platform.core.query.pipeline",
    "ai_platform.core.retrieval_contracts": "ai_platform.core.retrieval.contracts",
    "ai_platform.core.retrieval_query": "ai_platform.core.retrieval.query",
    "ai_platform.core.hybrid_retrieval": "ai_platform.core.retrieval.hybrid",
    "ai_platform.core.rrf": "ai_platform.core.retrieval.rrf",
    "ai_platform.core.semantic_alignment": "ai_platform.core.retrieval.semantic_alignment",
    "ai_platform.core.semantic_registry": "ai_platform.core.retrieval.semantic_registry",
    "ai_platform.core.candidate_verification": "ai_platform.core.retrieval.verification",

    # Legacy backend targets -> standalone platform targets.
    "backend.conversation_resolver": "ai_platform.core.conversation.conversation",
    "backend.multi_intent": "ai_platform.core.conversation.multi_intent",
    "backend.retriever": "ai_platform.runtime.retrieval",
    "backend.ingestion": "ai_platform.runtime.corpus",
    "backend.embedding": "ai_platform.runtime.embedding",
    "backend.vectorstore": "ai_platform.runtime.vectorstore",
    "backend.llm": "ai_platform.runtime.llm",
    "backend.config": "ai_platform.runtime.config",
    "backend.runtime.config": "ai_platform.runtime.config",
    "backend.runtime.embedding": "ai_platform.runtime.embedding",
    "backend.runtime.vectorstore": "ai_platform.runtime.vectorstore",
    "backend.runtime.llm": "ai_platform.runtime.llm",
    "backend.institutions.loader": "ai_platform.institutions.loader",
    "backend.institutions.iitj": "ai_platform.institutions.iitj",

    "backend.core.answering.generator": "ai_platform.core.answering.generator",
    "backend.core.answering.grounding": "ai_platform.core.answering.grounding",
    "backend.core.answering.guard": "ai_platform.core.answering.guard",
    "backend.core.candidate_verification": "ai_platform.core.retrieval.verification",
    "backend.core.evidence.claims": "ai_platform.core.evidence.claims",
    "backend.core.evidence.coverage": "ai_platform.core.evidence.coverage",
    "backend.core.evidence.evidence": "ai_platform.core.evidence.evidence",
    "backend.core.evidence.grouping": "ai_platform.core.evidence.grouping",
    "backend.core.evidence.packaging": "ai_platform.core.evidence.packaging",
    "backend.core.evidence.scope": "ai_platform.core.evidence.scope",
    "backend.core.query.models": "ai_platform.core.query.models",
    "backend.core.query.understanding": "ai_platform.core.query.understanding",
    "backend.core.query_frame": "ai_platform.core.query.frame",
    "backend.core.query_interpreter": "ai_platform.core.query.interpreter",
    "backend.core.query_pipeline": "ai_platform.core.query.pipeline",
    "backend.core.retrieval.lexical_recall": "ai_platform.core.retrieval.lexical_recall",
    "backend.core.retrieval.ranking": "ai_platform.core.retrieval.ranking",
    "backend.core.retrieval_contracts": "ai_platform.core.retrieval.contracts",
    "backend.core.retrieval_query": "ai_platform.core.retrieval.query",
    "backend.core.rrf": "ai_platform.core.retrieval.rrf",
    "backend.core.semantic_alignment": "ai_platform.core.retrieval.semantic_alignment",
    "backend.core.semantic_registry": "ai_platform.core.retrieval.semantic_registry",
    "backend.core.institution": "ai_platform.core.institution.profile",
    "backend.core.nodes": "ai_platform.core.graph.nodes",
}


SEED_FILES = {
    CORE / "conversation" / "conversation.py": r'''"""Standalone conversation-resolution adapter."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from ai_platform.runtime.llm import query_understanding_llm


MAX_HISTORY_MESSAGES = 8

_REFERENCE_STARTS = (
    "what about", "how about", "how much", "how many", "where is it",
    "where are they", "what is it", "what are they", "is it", "are they",
    "can i", "can we", "how do i", "when is it", "when are they", "and ",
)

_CONVERSATION_PROMPT = """
You are a generic conversation-resolution component for a retrieval system.
Resolve only references supported by recent conversation. Never invent
institutional facts. Never answer the user.

Return JSON only:
{
  "resolved_question": "",
  "mode": "standalone|follow_up",
  "active_topic": "",
  "active_entity": ""
}

Rules:
- Preserve the latest question's information need.
- Use history only for references such as it, they, what about, how much,
  or a clearly omitted subject.
- Do not replace a named target with a different target.
- When uncertain, preserve the latest question unchanged.
""".strip()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _history_users(history: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if isinstance(item, Mapping):
            role = _clean(item.get("role")).casefold()
            content = _clean(item.get("content") or item.get("message"))
            if role == "user" and content:
                result.append(content)
        else:
            content = _clean(getattr(item, "content", item))
            if content:
                result.append(content)
    return result


def _looks_like_follow_up(question: str) -> bool:
    normalized = _clean(question).casefold()
    return bool(normalized) and (len(normalized.split()) <= 6 or normalized.startswith(_REFERENCE_STARTS))


def _safe_fallback(question: str) -> dict[str, str]:
    return {"resolved_question": question, "mode": "standalone", "active_topic": "", "active_entity": ""}


def _parse_model_response(response: Any, original: str) -> dict[str, str]:
    content = response.content if hasattr(response, "content") else response
    text = _clean(content)
    if not text:
        return _safe_fallback(original)
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return _safe_fallback(original)
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return _safe_fallback(original)
    if not isinstance(payload, Mapping):
        return _safe_fallback(original)
    resolved = _clean(payload.get("resolved_question")) or original
    mode = _clean(payload.get("mode")).casefold()
    if mode not in {"standalone", "follow_up"}:
        mode = "follow_up" if resolved.casefold() != original.casefold() else "standalone"
    return {
        "resolved_question": resolved,
        "mode": mode,
        "active_topic": _clean(payload.get("active_topic")),
        "active_entity": _clean(payload.get("active_entity")),
    }


def resolve_conversation(*, question: str, chat_history: Sequence[Any] | None = None) -> dict[str, str]:
    original = _clean(question)
    if not original:
        raise ValueError("question cannot be empty")
    users = _history_users(tuple(chat_history or ()))
    if not users or not _looks_like_follow_up(original):
        return _safe_fallback(original)
    history_text = "\n".join(f"USER: {value}" for value in users[-MAX_HISTORY_MESSAGES:])
    try:
        response = query_understanding_llm.invoke([
            ("system", _CONVERSATION_PROMPT),
            ("human", f"<HISTORY>\n{history_text}\n</HISTORY>\n<LATEST_QUESTION>\n{original}\n</LATEST_QUESTION>"),
        ])
        return _parse_model_response(response, original)
    except Exception:
        return _safe_fallback(original)


__all__ = ["MAX_HISTORY_MESSAGES", "resolve_conversation"]
''',
    CORE / "conversation" / "multi_intent.py": r'''"""Standalone conservative multi-intent decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import re


_REQUEST_WORDS = {
    "what", "which", "who", "where", "when", "how", "why", "can",
    "could", "would", "do", "does", "did", "is", "are",
}


@dataclass(frozen=True, slots=True)
class IntentUnit:
    question: str
    topics: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()


def _clean(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _ensure_question(text: str) -> str:
    value = _clean(text).strip(" ,;.")
    return value if not value else (value if value.endswith("?") else value + "?")


def _starts_as_question(text: str) -> bool:
    first = _clean(text).casefold().split(maxsplit=1)
    return bool(first and first[0] in _REQUEST_WORDS)


def _split_explicit(question: str) -> list[str]:
    if "?" in question:
        return [p for p in (_ensure_question(x) for x in question.split("?")) if p]
    return [p for p in (_ensure_question(x) for x in re.split(r"\s*;\s*", question)) if p]


def _split_coordinated(question: str) -> list[str]:
    normalized = _clean(question)
    matches = list(re.finditer(r"\s+(?:and|also)\s+", normalized, flags=re.I))
    for match in matches:
        left = _clean(normalized[:match.start()])
        right = _clean(normalized[match.end():])
        if not left or not right:
            continue
        if _starts_as_question(right):
            return [_ensure_question(left), _ensure_question(right)]
        if _starts_as_question(left) and len(right.split()) <= 7:
            prefix = re.match(r"^(what|which|where|when|how|who|can|could|would|is|are)\b", left, re.I)
            if prefix:
                return [_ensure_question(left), _ensure_question(prefix.group(1) + " " + right)]
    return []


def decompose_multi_intent(question: str) -> list[IntentUnit]:
    original = _clean(question)
    if not original:
        raise ValueError("question cannot be empty")
    explicit = _split_explicit(original)
    if len(explicit) > 1:
        return [IntentUnit(x) for x in explicit]
    coordinated = _split_coordinated(original)
    if len(coordinated) > 1:
        return [IntentUnit(x) for x in coordinated]
    return [IntentUnit(_ensure_question(original))]


def is_multi_intent(question: str) -> bool:
    return len(decompose_multi_intent(question)) > 1


def intent_questions(question: str) -> tuple[str, ...]:
    return tuple(x.question for x in decompose_multi_intent(question))


__all__ = ["IntentUnit", "decompose_multi_intent", "is_multi_intent", "intent_questions"]
''',
    CORE / "graph" / "state.py": r'''"""LangGraph state contract for the reusable AI platform."""

from __future__ import annotations

from typing import Any, Sequence, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    chat_history: Sequence[Any]
    resolved_question: str
    conversation_mode: str
    active_topic: str
    active_entity: str
    is_multi_intent: bool
    intent_count: int
    intent_units: Sequence[Any]
    intent_questions: Sequence[str]
    query: Any
    query_frame: Any
    retrieval_queries: Sequence[str]
    retrieval_results: Sequence[Any]
    retrieval_weights: Sequence[float]
    fused_candidates: Sequence[Any]
    verified_candidates: Sequence[Any]
    uncertain_candidates: Sequence[Any]
    rejected_candidates: Sequence[Any]
    ranked_candidates: Sequence[Any]
    evidence_groups: Sequence[Any]
    evidence_documents: Sequence[Any]
    evidence_candidates: Sequence[Any]
    evidence_assessment: Any
    evidence_coverage: Any
    claim_audit: Any
    evidence_package: Any
    answer: str
    grounding_assessment: Any
    guard_result: Any
    error: str


__all__ = ["RAGState"]
''',
    CORE / "graph" / "graph.py": r'''"""Canonical LangGraph wiring for the standalone reusable AI platform."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ai_platform.core.graph.nodes import CoreNodes
from ai_platform.core.graph.state import RAGState


def _route_after_intent_planning(state: dict[str, Any]) -> str:
    return "multi_intent" if bool(state.get("is_multi_intent")) else "single_intent"


def build_graph(nodes: CoreNodes | None = None):
    runtime_nodes = nodes or CoreNodes()
    workflow = StateGraph(RAGState)
    workflow.add_node("resolve_conversation", runtime_nodes.resolve_conversation_node)
    workflow.add_node("plan_multi_intent", runtime_nodes.plan_multi_intent_node)
    workflow.add_node("retrieve_and_qualify", runtime_nodes.retrieve_and_qualify_node)
    workflow.add_node("process_multi_intent", runtime_nodes.process_multi_intent_node)
    workflow.add_node("answer", runtime_nodes.answer_node)
    workflow.add_edge(START, "resolve_conversation")
    workflow.add_edge("resolve_conversation", "plan_multi_intent")
    workflow.add_conditional_edges("plan_multi_intent", _route_after_intent_planning, {"single_intent": "retrieve_and_qualify", "multi_intent": "process_multi_intent"})
    workflow.add_edge("retrieve_and_qualify", "answer")
    workflow.add_edge("process_multi_intent", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


__all__ = ["build_graph"]
''',
    CORE / "institution" / "profile.py": r'''"""Generic institution deployment contract for the reusable RAG core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InstitutionProfile:
    institution_id: str
    display_name: str
    data_path: Path
    vectorstore_path: Path
    vectorstore_collection: str | None = None
    structured_data_path: Path | None = None
    default_language: str = "English"
    supported_languages: tuple[str, ...] = ("English",)
    answer_tone: str = "student_friendly"
    answer_verbosity: str = "moderate"
    fallback_message: str = "I'm sorry, I don't know based on the available information."
    enable_semantic_query_understanding: bool = True
    enable_multi_intent: bool = True
    enable_hybrid_retrieval: bool = True
    enable_local_context: bool = True
    enable_evidence_guard: bool = True
    environment: str = "development"
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    prompt_additions: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.institution_id.strip(): raise ValueError("institution_id cannot be empty")
        if not self.display_name.strip(): raise ValueError("display_name cannot be empty")
        if not self.data_path: raise ValueError("data_path cannot be empty")
        if not self.vectorstore_path: raise ValueError("vectorstore_path cannot be empty")
        if not self.supported_languages: raise ValueError("supported_languages cannot be empty")
        if not self.default_language.strip(): raise ValueError("default_language cannot be empty")
        if not self.answer_tone.strip(): raise ValueError("answer_tone cannot be empty")
        if not self.answer_verbosity.strip(): raise ValueError("answer_verbosity cannot be empty")
        if not self.fallback_message.strip(): raise ValueError("fallback_message cannot be empty")
        if not self.environment.strip(): raise ValueError("environment cannot be empty")
        if self.vectorstore_collection is not None and not self.vectorstore_collection.strip():
            raise ValueError("vectorstore_collection cannot be empty when provided")
        if any(not language.strip() for language in self.supported_languages):
            raise ValueError("supported_languages cannot contain empty values")
        if any(not key.strip() or not value.strip() for key, value in self.metadata):
            raise ValueError("metadata keys and values cannot be empty")

    @property
    def institution_data_root(self) -> Path:
        return self.data_path

    @property
    def vector_db_root(self) -> Path:
        return self.vectorstore_path

    @property
    def has_structured_data(self) -> bool:
        return self.structured_data_path is not None


__all__ = ["InstitutionProfile"]
''',
}

INIT_FILES = {
    CORE / "__init__.py": '"""Reusable AI platform core package."""\n',
    CORE / "conversation" / "__init__.py": '"""Conversation resolution and intent decomposition."""\n',
    CORE / "graph" / "__init__.py": '"""LangGraph orchestration package."""\n',
    CORE / "institution" / "__init__.py": '"""Generic institution deployment contracts."""\n',
    CORE / "query" / "__init__.py": '"""Query contracts, interpretation, and pipeline."""\n',
    CORE / "retrieval" / "__init__.py": '"""Reusable retrieval primitives and contracts."""\n',
}



def _replace_import_paths(text: str) -> str:
    # Longest-first avoids partial replacement collisions.
    for old, new in sorted(IMPORT_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(old, new)
    return text


def _safe_move(source: Path, destination: Path, *, dry_run: bool) -> None:
    if not source.exists():
        return
    if destination.exists():
        source_text = source.read_text(encoding="utf-8", errors="replace")
        dest_text = destination.read_text(encoding="utf-8", errors="replace")
        if source_text != dest_text:
            raise RuntimeError(
                f"Refusing to overwrite different file:\n  source: {source}\n  destination: {destination}"
            )
        if not dry_run:
            source.unlink()
        print(f"OK identical: {source} -> {destination} (source removed)")
        return
    print(f"MOVE: {source} -> {destination}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def _rewrite_platform_python(*, dry_run: bool) -> None:
    for path in sorted(PLATFORM.rglob("*.py")):
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8", errors="replace")
        updated = _replace_import_paths(original)
        if updated != original:
            print(f"REWRITE: {path}")
            if not dry_run:
                path.write_text(updated, encoding="utf-8")


def _seed_final_files(*, dry_run: bool) -> None:
    for destination, content in SEED_FILES.items():
        if destination.exists():
            continue
        print(f"CREATE SEED: {destination}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")


def _install_init_files(*, dry_run: bool) -> None:
    for path, content in INIT_FILES.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # The root __init__.py is intentionally normalized; subpackage init
            # files are replaced only if empty so existing exports are retained.
            if path == CORE / "__init__.py" or not path.read_text(encoding="utf-8", errors="replace").strip():
                print(f"REWRITE INIT: {path}")
                if not dry_run:
                    path.write_text(content, encoding="utf-8")
            else:
                print(f"KEEP INIT: {path}")
        else:
            print(f"CREATE INIT: {path}")
            if not dry_run:
                path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show changes without modifying files")
    args = parser.parse_args()

    if not PLATFORM.exists():
        print(f"Missing {PLATFORM}", file=sys.stderr)
        return 2

    for source, destination in MOVE_MAP.items():
        _safe_move(source, destination, dry_run=args.dry_run)

    _seed_final_files(dry_run=args.dry_run)
    _install_init_files(dry_run=args.dry_run)
    _rewrite_platform_python(dry_run=args.dry_run)

    print("\nStandalone core reorganization complete." if not args.dry_run else "\nDry run complete.")
    print("No backend/ files were modified by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
