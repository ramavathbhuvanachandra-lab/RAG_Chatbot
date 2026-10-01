"""Core LangGraph node orchestration for the reusable institutional RAG engine.

This module is the canonical orchestration layer for the new architecture.

Design
------
Conversation
    -> query understanding
    -> multi-intent planning
    -> bounded hybrid retrieval
    -> weighted RRF
    -> semantic candidate construction
    -> candidate verification
    -> deterministic ranking
    -> local context + evidence grouping
    -> scope/conflict protection
    -> evidence sufficiency
    -> evidence coverage
    -> claim audit
    -> final evidence packaging
    -> one answer generation call
    -> answer grounding
    -> final output guard

Important boundaries
--------------------
* This file orchestrates; domain facts remain in institution data/config.
* Retrieval services are imported lazily through dependency adapters so the
  core remains testable without Ollama/Chroma/LangChain startup side effects.
* The user/resolved question remains authoritative. Derived queries are
  bounded recovery signals only.
* Candidate provenance is retained, while internal provenance never becomes
  user-facing answer context.
* The final answer path uses only the new E7.1/E7.2/E8 answering modules.
* No institution-specific vocabulary, source paths, fallback text, or model
  names are embedded in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import re
from typing import Any, Callable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Small contracts
# ---------------------------------------------------------------------------

State = dict[str, Any]
Document = Any


@dataclass(frozen=True, slots=True)
class NodeDependencies:
    """Explicit service ports used by the core node layer.

    Every callable has a narrow role. Production defaults are resolved lazily
    by ``default_dependencies``; tests can replace any service independently.
    """

    conversation_resolver: Callable[..., Mapping[str, Any]]
    multi_intent_decomposer: Callable[[str], Sequence[Any]]
    query_understander: Callable[..., Any]

    dense_retrieve: Callable[[str], Sequence[Document]]
    keyword_retrieve: Callable[[str], Sequence[Document]]
    fuse_ranked_lists: Callable[..., Sequence[Any]]

    align_query_to_document: Callable[..., Any]
    rank_candidates: Callable[..., Sequence[Any]]
    verify_candidates: Callable[..., Any]

    build_evidence_groups: Callable[[Sequence[Document]], Sequence[Any]]
    expand_group_context: Callable[..., Sequence[Any]]
    flatten_evidence_groups: Callable[..., Sequence[Document]]

    filter_scope_conflicts: Callable[..., Sequence[Document]]
    assess_evidence: Callable[..., Any]
    assess_coverage: Callable[..., Any]

    extract_claims: Callable[[Any], Sequence[Any]]
    extract_evidence_units: Callable[[Sequence[Any]], Sequence[Any]]
    audit_claims: Callable[..., Any]
    build_evidence_package: Callable[..., Any]

    answer_generator: Callable[[Any, Any], Any]
    answer_request_type: type
    assess_answer_grounding: Callable[..., Any]
    guard_answer: Callable[..., Any]

    institution_provider: Callable[[State], Any]
    answer_model_provider: Callable[[State], Any]
    fallback_provider: Callable[[State, Any], str]

    canonical_chunks_provider: Callable[[State], Sequence[Document]]

    # Optional institution adapter for explicit claim-scope conflicts. The
    # adapter owns institution vocabulary; the core only applies its
    # generic conflict contract.
    scope_policy_provider: Callable[[State], Any] | None = None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _casefold(value: Any) -> str:
    return _clean(value).casefold()


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return ()


def _document_text(document: Any) -> str:
    if isinstance(document, Mapping):
        return _clean(document.get("page_content"))
    return _clean(getattr(document, "page_content", ""))


def _document_source(document: Any) -> str:
    if isinstance(document, Mapping):
        metadata = document.get("metadata") or {}
    else:
        metadata = getattr(document, "metadata", {}) or {}

    if not isinstance(metadata, Mapping):
        return ""

    for key in ("source", "source_path", "path", "url"):
        value = _clean(metadata.get(key))
        if value:
            return value
    return ""


def _document_id(document: Any) -> str:
    value = _clean(getattr(document, "document_id", ""))
    if value:
        return value

    metadata = (
        document.get("metadata", {})
        if isinstance(document, Mapping)
        else getattr(document, "metadata", {})
    ) or {}

    if isinstance(metadata, Mapping):
        for key in ("document_id", "id", "chunk_id"):
            value = _clean(metadata.get(key))
            if value:
                return value

    # Stable fallback is intentionally delegated to the canonical candidate
    # contract in _to_candidate(), not guessed from display metadata here.
    return ""


def _distinct_queries(values: Sequence[str], limit: int = 3) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = _clean(value)
        if not text:
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(text)

        if len(result) >= max(1, int(limit)):
            break

    return tuple(result)


def _answer_text(result: Any) -> str:
    if result is None:
        return ""

    value = getattr(result, "answer", None)
    if value is not None:
        return _clean(value)

    if isinstance(result, Mapping):
        for key in ("answer", "text", "content"):
            if key in result:
                return _clean(result[key])

    return _clean(getattr(result, "content", result))


def _status(result: Any, default: str = "insufficient") -> str:
    if result is None:
        return default
    value = getattr(result, "status", None)
    if value is not None:
        return _clean(value).casefold() or default
    if isinstance(result, Mapping):
        value = result.get("status")
        return _clean(value).casefold() or default if value is not None else default
    return default


def _fallback_from_state(state: State, institution: Any) -> str:
    explicit = _clean(state.get("fallback_response"))
    if explicit:
        return explicit

    if institution is not None:
        for name in (
            "fallback_response",
            "unknown_response",
            "fallback_message",
            "unknown_message",
        ):
            value = _clean(getattr(institution, name, ""))
            if value:
                return value

    raise ValueError(
        "No institution-configured fallback response is available."
    )


def _query_search_variants(query: Any, resolved_question: str) -> tuple[str, ...]:
    """Build bounded retrieval formulations without replacing the primary."""
    candidates = [resolved_question]

    semantic = _clean(getattr(query, "search_query", ""))
    if semantic:
        candidates.append(semantic)

    target = getattr(query, "target", None)
    request_type = _clean(getattr(query, "request_type", ""))
    target_text = _clean(getattr(target, "text", ""))

    compact_parts = [target_text, request_type]
    compact = _clean(" ".join(part for part in compact_parts if part))
    if compact:
        candidates.append(compact)

    return _distinct_queries(candidates, limit=3)


def _with_alignment(candidate: Any, alignment: Any, meaning: Any) -> Any:
    """Attach semantic meaning/alignment without mutating frozen candidates."""
    try:
        return replace(
            candidate,
            alignment=alignment,
            meaning=meaning,
        )
    except TypeError:
        return candidate


def _document_identity_key(document: Any) -> tuple[str, str]:
    return (
        _document_source(document).casefold(),
        _document_text(document).casefold(),
    )


def _lexical_recall_candidates(
    state: State,
    query_text: str,
    query_frame: Any,
    existing: Sequence[Any],
    *,
    limit: int = 40,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    """Scan the canonical corpus with a deterministic lexical rescue lane."""
    documents = tuple(state.get("canonical_chunks", ()) or ())

    if not documents:
        return (), ()

    from backend.core.retrieval.lexical_recall import retrieve_lexical
    from backend.core.retrieval_contracts import (
        CandidateAlignment,
        DocumentMeaning,
        RetrievalCandidate,
        RetrievalProvenance,
        RetrievalSignal,
    )

    hits = retrieve_lexical(
        query_text,
        documents,
        query_frame=query_frame,
        limit=limit,
        min_score=0.28,
    )

    existing_keys = {
        _document_identity_key(getattr(candidate, "document", candidate))
        for candidate in existing
    }

    candidates: list[Any] = []
    trace: list[dict[str, Any]] = []
    rank = 0

    for hit in hits:
        key = _document_identity_key(hit.document)
        if key in existing_keys:
            continue
        rank += 1
        source = _document_source(hit.document) or "unknown"
        provenance = RetrievalProvenance(
            primary_query=query_text,
            retrieval_queries=(query_text,),
            signals=(
                RetrievalSignal(
                    channel="lexical_fallback",
                    rank=rank,
                    score=hit.score,
                    weight=1.0,
                    query=query_text,
                ),
            ),
        )
        candidate = RetrievalCandidate.from_document(
            hit.document,
            source=source,
            provenance=provenance,
            meaning=DocumentMeaning(),
            alignment=CandidateAlignment(),
        )
        candidates.append(candidate)
        trace.append({
            "lexical_rank": rank,
            "score": round(float(hit.score), 6),
            "query_overlap": round(float(hit.query_overlap), 6),
            "target_overlap": round(float(hit.target_overlap), 6),
            "facet_overlap": round(float(hit.facet_overlap), 6),
            "phrase_hit": bool(hit.phrase_hit),
            "matched_terms": list(hit.matched_terms),
            "source": source,
            "text_preview": _clean(_document_text(hit.document))[:280],
        })

        existing_keys.add(key)
        if len(candidates) >= limit:
            break

    return tuple(candidates), tuple(trace)


def _candidate_trace(candidate: Any, rank: int, stage: str = "ranked") -> dict[str, Any]:
    document = getattr(candidate, "document", candidate)
    provenance = getattr(candidate, "provenance", None)
    alignment = getattr(candidate, "alignment", None)
    return {
        "rank": int(rank),
        "stage": stage,
        "document_id": _clean(getattr(candidate, "document_id", "")),
        "source": _document_source(document),
        "final_score": round(float(getattr(candidate, "final_score", 0.0) or 0.0), 6),
        "text_preview": _document_text(document)[:280],
        "retrieval": provenance.to_dict() if hasattr(provenance, "to_dict") else {},
        "alignment": alignment.to_dict() if hasattr(alignment, "to_dict") else {},
    }


def _to_candidate(document: Document, deps: NodeDependencies, query_text: str) -> Any:
    from backend.core.retrieval_contracts import RetrievalCandidate

    _, meaning, alignment = deps.align_query_to_document(
        query_text,
        document,
        _load_semantic_registry(document, deps),
    )

    candidate = RetrievalCandidate.from_document(
        document,
        meaning=meaning,
        alignment=alignment,
        source=_document_source(document) or None,
    )
    return candidate


def _load_semantic_registry(_document: Any, _deps: NodeDependencies) -> Any:
    """Load the active deployment registry lazily.

    The deployment registry is institution-specific configuration, not core
    vocabulary. Tests normally inject ``align_query_to_document`` and never
    reach this adapter.
    """
    from backend.institutions.loader import load_institution_profile

    profile = load_institution_profile()
    module = importlib.import_module(
        f"backend.institutions.{profile.institution_id}.semantic_registry"
    )

    registry = getattr(module, "SEMANTIC_REGISTRY", None)
    if registry is None:
        legacy_name = f"{profile.institution_id.upper()}_SEMANTIC_REGISTRY"
        registry = getattr(module, legacy_name, None)

    if registry is None:
        raise AttributeError(
            "Active institution semantic registry must export SEMANTIC_REGISTRY."
        )

    validate = getattr(registry, "validate", None)
    if callable(validate):
        validate()

    return registry


# ---------------------------------------------------------------------------
# Lazy production adapters
# ---------------------------------------------------------------------------

def _resolve_callable(module_name: str, attribute: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"{module_name}.{attribute} must be callable")
    return value


def default_dependencies() -> NodeDependencies:
    """Construct production dependencies without eager service startup."""
    query_understanding = _resolve_callable(
        "backend.core.query.understanding",
        "understand_query",
    )
    conversation_resolver = _resolve_callable(
        "backend.conversation_resolver",
        "resolve_conversation",
    )
    multi_intent = _resolve_callable(
        "backend.multi_intent",
        "decompose_multi_intent",
    )

    dense = _resolve_callable("backend.retriever", "dense_retrieve")
    keyword = _resolve_callable("backend.retriever", "keyword_retrieve")
    fuse = _resolve_callable("backend.core.rrf", "fuse_ranked_lists")

    align = _resolve_callable(
        "backend.core.semantic_alignment",
        "align_query_to_document",
    )
    rank = _resolve_callable(
        "backend.core.retrieval.ranking",
        "rank_candidates",
    )
    verify = _resolve_callable(
        "backend.core.candidate_verification",
        "verify_candidates",
    )

    build_groups = _resolve_callable(
        "backend.core.evidence.grouping",
        "build_evidence_groups",
    )
    expand_groups = _resolve_callable(
        "backend.core.evidence.grouping",
        "expand_group_context",
    )
    flatten_groups = _resolve_callable(
        "backend.core.evidence.grouping",
        "flatten_evidence_groups",
    )

    scope_filter = _resolve_callable(
        "backend.core.evidence.scope",
        "filter_scope_conflicts",
    )
    evidence_assess = _resolve_callable(
        "backend.core.evidence.evidence",
        "assess_evidence",
    )
    coverage_assess = _resolve_callable(
        "backend.core.evidence.coverage",
        "assess_coverage",
    )

    extract_claims = _resolve_callable(
        "backend.core.evidence.claims",
        "extract_claims",
    )
    extract_units = _resolve_callable(
        "backend.core.evidence.claims",
        "extract_evidence_units",
    )
    audit_claims = _resolve_callable(
        "backend.core.evidence.claims",
        "audit_claims",
    )
    package = _resolve_callable(
        "backend.core.evidence.packaging",
        "build_evidence_package",
    )

    generator = _resolve_callable(
        "backend.core.answering.generator",
        "generate_answer",
    )
    request_type = getattr(
        importlib.import_module("backend.core.answering.generator"),
        "AnswerGenerationRequest",
    )

    grounding = _resolve_callable(
        "backend.core.answering.grounding",
        "assess_answer_grounding",
    )
    guard = _resolve_callable(
        "backend.core.answering.guard",
        "guard_answer",
    )

    def institution_provider(state: State) -> Any:
        explicit = state.get("institution_profile")
        if explicit is not None:
            return explicit
        return _resolve_callable(
            "backend.institutions.loader",
            "load_institution_profile",
        )()

    def answer_model_provider(_state: State) -> Any:
        try:
            return getattr(
                importlib.import_module("backend.runtime.llm"),
                "answer_llm",
            )
        except (ImportError, AttributeError):
            return getattr(
                importlib.import_module("backend.llm"),
                "answer_llm",
            )

    def fallback_provider(state: State, institution: Any) -> str:
        return _fallback_from_state(state, institution)

    def canonical_chunks_provider(_state: State) -> Sequence[Document]:
        try:
            module = importlib.import_module("backend.retriever")
            return tuple(getattr(module, "chunks", ()) or ())
        except Exception:
            return ()

    def scope_policy_provider(_state: State) -> Any:
        return _load_scope_policy()

    return NodeDependencies(
        conversation_resolver=conversation_resolver,
        multi_intent_decomposer=multi_intent,
        query_understander=query_understanding,
        dense_retrieve=dense,
        keyword_retrieve=keyword,
        fuse_ranked_lists=fuse,
        align_query_to_document=align,
        rank_candidates=rank,
        verify_candidates=verify,
        build_evidence_groups=build_groups,
        expand_group_context=expand_groups,
        flatten_evidence_groups=flatten_groups,
        filter_scope_conflicts=scope_filter,
        assess_evidence=evidence_assess,
        assess_coverage=coverage_assess,
        extract_claims=extract_claims,
        extract_evidence_units=extract_units,
        audit_claims=audit_claims,
        build_evidence_package=package,
        answer_generator=generator,
        answer_request_type=request_type,
        assess_answer_grounding=grounding,
        guard_answer=guard,
        institution_provider=institution_provider,
        answer_model_provider=answer_model_provider,
        fallback_provider=fallback_provider,
        canonical_chunks_provider=canonical_chunks_provider,
        scope_policy_provider=scope_policy_provider,
    )


# ---------------------------------------------------------------------------
# Institution scope policy integration
# ---------------------------------------------------------------------------

def _load_scope_policy() -> Any:
    """Load the active deployment's optional claim-scope policy."""
    from backend.institutions.loader import load_institution_profile

    profile = load_institution_profile()
    module = importlib.import_module(
        f"backend.institutions.{profile.institution_id}.scope_policy"
    )

    policy = getattr(module, "SCOPE_POLICY", None)
    if policy is None:
        getter = getattr(module, "get_scope_policy", None)
        if callable(getter):
            policy = getter()

    if policy is None:
        raise AttributeError(
            "Active institution scope policy must export SCOPE_POLICY or get_scope_policy()."
        )

    validate = getattr(module, "validate_scope_policy", None)
    if callable(validate):
        validate(policy)

    return policy


def _scope_policy_conflicts(
    policy: Any,
    question: str,
    source: str,
    document_text: str,
) -> tuple[str, ...]:
    """Return hard scope conflicts using source scope first, content second.

    Source scope is evaluated independently because a document may mention
    another program merely as a prerequisite/qualifier. A program-specific
    source path, by contrast, expresses the document's authoritative scope.
    Content-level conflicts are used only as a fallback when the source does
    not establish that dimension.
    """
    detect = getattr(policy, "detect", None)
    conflict_fn = getattr(policy, "conflicting_dimensions", None)
    if not callable(detect) or not callable(conflict_fn):
        return ()

    try:
        query_scopes = detect(question)
        source_scopes = detect(source)
        document_scopes = detect(document_text)

        conflicts = list(conflict_fn(query_scopes, source_scopes))

        # Content may expose an explicit conflict when the source path is
        # generic. Do not let content mentions override an already established
        # source scope; source scope is the stronger signal.
        for dimension in conflict_fn(query_scopes, document_scopes):
            if dimension not in source_scopes and dimension not in conflicts:
                conflicts.append(dimension)

        return tuple(
            dict.fromkeys(
                str(item) for item in conflicts if str(item)
            )
        )
    except Exception:
        return ()


def _apply_scope_conflicts(
    candidate: Any,
    policy: Any,
    question: str,
) -> Any:
    """Attach institution scope conflicts to a retrieval candidate."""
    if policy is None:
        return candidate

    document = getattr(candidate, "document", candidate)
    source = _document_source(document)
    text = _document_text(document)
    conflicts = _scope_policy_conflicts(
        policy,
        question,
        source,
        text,
    )
    if not conflicts:
        return candidate

    alignment = getattr(candidate, "alignment", None)
    if alignment is None:
        return candidate

    existing = tuple(getattr(alignment, "conflicts", ()) or ())
    merged = tuple(dict.fromkeys((*existing, *conflicts)))
    try:
        return replace(
            candidate,
            alignment=replace(alignment, conflicts=merged),
        )
    except TypeError:
        return candidate


# ---------------------------------------------------------------------------
# Institution semantic enrichment and protected retrieval queries
# ---------------------------------------------------------------------------

def _registry_description(registry: Any, text: str) -> Mapping[str, Any]:
    describe=getattr(registry,"describe",None)
    if not callable(describe): return {}
    try: value=describe(text)
    except Exception: return {}
    return value if isinstance(value,Mapping) else {}

def _enrich_query_contract(
    query: Any,
    frame: Any,
    question: str,
    registry: Any,
) -> tuple[Any, Any, tuple[str, ...]]:
    """Enrich the generic query contract from the active institution registry.

    Registry data supplies the institution vocabulary; this function only
    applies generic mechanics: detect explicit programs/entities, promote a
    unique concrete target when the LLM omitted one, and preserve protected
    terms across retrieval variants.
    """
    from dataclasses import replace
    from backend.core.query.models import Entity, EntityMention, Target

    desc = _registry_description(registry, question)
    programs = tuple(_clean(value) for value in (desc.get("programs", ()) or ()) if _clean(value))
    topic_names = { _clean(value).casefold() for value in (desc.get("topics", ()) or ()) if _clean(value) }
    raw_entities = tuple(_clean(value) for value in (desc.get("entities", ()) or ()) if _clean(value))
    concrete_entities = tuple(value for value in raw_entities if value.casefold() not in topic_names)

    protected: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    for name in (*programs, *concrete_entities):
        kind = "program" if name in programs else "entity"
        key = (kind, name.casefold())
        if key in seen:
            continue
        seen.add(key)
        protected.append(
            Entity(
                name=name,
                entity_type=kind,
                entity_id=name,
                confidence=1.0,
                resolution_state="resolved",
            )
        )

    target = getattr(query, "target", None)
    target_text = _clean(getattr(target, "text", "")) if target else ""
    action_target = bool(
        target_text.casefold().startswith((
            "apply ", "how do i ", "how can i ", "how to ",
            "where do i ", "where can i ",
        ))
    )
    if action_target and not any(
        target_text.casefold() == _clean(getattr(entity, "name", "")).casefold()
        for entity in protected
    ):
        target = None

    protected_programs = [entity for entity in protected if getattr(entity, "entity_type", "") == "program"]
    protected_concrete = [entity for entity in protected if getattr(entity, "entity_type", "") == "entity"]

    # The registry may recover the exact target even when the semantic model
    # emitted only the request type (e.g. "M.Tech eligibility").
    if target is None:
        promoted = None
        if len(protected_programs) == 1:
            promoted = protected_programs[0]
        elif len(protected_concrete) == 1:
            promoted = protected_concrete[0]
        if promoted is not None:
            target = Target(
                text=getattr(promoted, "name"),
                entity_id=getattr(promoted, "entity_id", None),
                entity_type=getattr(promoted, "entity_type", None),
                confidence=1.0,
                resolution_state="resolved",
            )
    elif target is not None:
        target_name = target_text.casefold()
        for entity in protected:
            entity_name = _clean(getattr(entity, "name", ""))
            if entity_name and entity_name.casefold() == target_name:
                target = Target(
                    text=target_text,
                    entity_id=getattr(entity, "entity_id", None),
                    entity_type=getattr(entity, "entity_type", None),
                    confidence=1.0,
                    resolution_state="resolved",
                )
                break

    mentions = list(getattr(query, "entity_mentions", ()) or ())
    mentioned = {_clean(getattr(item, "normalized_text", None) or getattr(item, "text", "")).casefold() for item in mentions}
    for entity in protected:
        name = _clean(getattr(entity, "name", ""))
        if name and name.casefold() not in mentioned:
            mentions.append(
                EntityMention(
                    text=name,
                    normalized_text=name,
                    entity_id=getattr(entity, "entity_id", None),
                    entity_type=getattr(entity, "entity_type", None),
                    confidence=1.0,
                    resolution_state="resolved",
                )
            )
            mentioned.add(name.casefold())

    preserved = list(getattr(query, "preserved_terms", ()) or ())
    preserved.extend(getattr(entity, "name", "") for entity in protected)

    q = replace(
        query,
        target=target,
        entities=tuple(protected),
        entity_mentions=tuple(mentions),
        preserved_terms=tuple(dict.fromkeys(_clean(value) for value in preserved if _clean(value))),
    )

    f = None
    if frame is not None:
        f = replace(
            frame,
            target=getattr(target, "text", None) if target else None,
            entities=tuple(protected),
            entity_mentions=tuple(mentions),
            semantic_query=getattr(q, "search_query", ""),
        )

    aliases: list[str] = []
    mapping = getattr(registry, "program_terms", {}) or {}
    for canonical, variants in mapping.items():
        aliases.append(str(canonical))
        aliases.extend(str(value) for value in (variants or ()))

    return q, f, tuple(dict.fromkeys(_clean(value) for value in aliases if _clean(value)))

def _protected_queries(query: Any, primary: str) -> tuple[str,...]:
    protected=[]; target=getattr(query,"target",None)
    if target is not None and _clean(getattr(target,"text","")): protected.append(_clean(getattr(target,"text","")))
    protected.extend(_clean(getattr(e,"name","")) for e in (getattr(query,"entities",()) or ()) if _clean(getattr(e,"name","")))
    protected=tuple(dict.fromkeys(protected)); out=[primary]
    semantic=_clean(getattr(query,"search_query",""))
    if semantic and all(p.casefold() in semantic.casefold() for p in protected): out.append(semantic)
    compact=_clean(" ".join(dict.fromkeys((*protected,_clean(getattr(query,"request_type",""))))))
    if compact and all(p.casefold() in compact.casefold() for p in protected): out.append(compact)
    return _distinct_queries(out,limit=3)

# ---------------------------------------------------------------------------
# Core node implementation
# ---------------------------------------------------------------------------

class CoreNodes:
    """Reusable orchestration nodes for one institutional deployment."""

    def __init__(self, dependencies: NodeDependencies | None = None) -> None:
        self.deps = dependencies or default_dependencies()

    # ------------------------------------------------------------------
    # Query/conversation
    # ------------------------------------------------------------------

    def resolve_conversation_node(self, state: State) -> State:
        question = _clean(state.get("question"))
        if not question:
            raise ValueError("state['question'] cannot be empty")

        history = _sequence(state.get("chat_history"))
        result = self.deps.conversation_resolver(
            question=question,
            chat_history=history,
        )

        resolved = _clean(
            result.get("resolved_question", question)
            if isinstance(result, Mapping)
            else question
        )

        return {
            "resolved_question": resolved or question,
            "conversation_mode": (
                _clean(result.get("mode"))
                if isinstance(result, Mapping)
                else ""
            ),
            "active_topic": (
                _clean(result.get("active_topic"))
                if isinstance(result, Mapping)
                else ""
            ),
            "active_entity": (
                _clean(result.get("active_entity"))
                if isinstance(result, Mapping)
                else ""
            ),
        }

    def understand_query_node(self, state: State) -> State:
        question = _clean(
            state.get("resolved_question")
            or state.get("question")
        )
        if not question:
            raise ValueError("No query is available for understanding.")

        query = self.deps.query_understander(question)

        query_frame = None
        try:
            frame_builder = _resolve_callable(
                "backend.core.query.understanding",
                "understand_query_frame",
            )
            query_frame = frame_builder(question)
        except Exception:
            # The canonical Query remains valid even if the migration frame
            # helper is temporarily unavailable.
            query_frame = None

        registry = None
        try:
            registry = _load_semantic_registry(None, self.deps)
        except Exception:
            pass
        known_program_aliases = ()
        if registry is not None:
            try:
                query, query_frame, known_program_aliases = _enrich_query_contract(query, query_frame, question, registry)
            except Exception:
                pass
        retrieval_queries = _protected_queries(query, question)

        return {
            "query": query,
            "query_frame": query_frame,
            "retrieval_queries": retrieval_queries,
            "known_program_aliases": known_program_aliases,
            "semantic_query": _clean(getattr(query, "search_query", "")),
            "query_confidence": float(
                getattr(query, "confidence", 0.0) or 0.0
            ),
            "query_resolution_state": _clean(
                getattr(query, "resolution_state", "")
            ),
        }

    def plan_multi_intent_node(self, state: State) -> State:
        question = _clean(
            state.get("resolved_question")
            or state.get("question")
        )
        units = tuple(
            self.deps.multi_intent_decomposer(question)
        )

        normalized_units: list[dict[str, Any]] = []
        for unit in units:
            unit_question = _clean(
                getattr(unit, "question", "")
                or (
                    unit.get("question", "")
                    if isinstance(unit, Mapping)
                    else ""
                )
            )
            if not unit_question:
                continue

            normalized_units.append(
                {
                    "question": unit_question,
                    "topics": list(
                        getattr(unit, "topics", ())
                        or (
                            unit.get("topics", ())
                            if isinstance(unit, Mapping)
                            else ()
                        )
                    ),
                    "entities": list(
                        getattr(unit, "entities", ())
                        or (
                            unit.get("entities", ())
                            if isinstance(unit, Mapping)
                            else ()
                        )
                    ),
                }
            )

        if not normalized_units:
            normalized_units = [{"question": question, "topics": [], "entities": []}]

        return {
            "is_multi_intent": len(normalized_units) > 1,
            "intent_count": len(normalized_units),
            "intent_units": normalized_units,
            "intent_questions": [
                item["question"] for item in normalized_units
            ],
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def hybrid_retrieve_node(self, state: State) -> State:
        primary = _clean(
            state.get("resolved_question")
            or state.get("question")
        )
        queries = tuple(
            _sequence(state.get("retrieval_queries"))
        ) or (primary,)

        queries = _distinct_queries(queries, limit=3)

        retrieval_results: list[Sequence[Document]] = []
        weights: list[float] = []
        lexical_recall_counts: list[int] = []

        # Primary query stays dominant. Additional formulations only recover
        # recall. A deterministic lexical lane is added to the primary query
        # so exact/well-known wording remains recoverable even when semantic
        # understanding or BM25 ranking is imperfect.
        decay = (1.0, 0.65, 0.45)

        canonical_chunks = tuple(self.deps.canonical_chunks_provider(state) or ())

        for index, query in enumerate(queries):
            dense_docs = tuple(self.deps.dense_retrieve(query))
            bm25_docs = tuple(self.deps.keyword_retrieve(query))

            retrieval_results.extend((dense_docs, bm25_docs))
            factor = decay[min(index, len(decay) - 1)]
            weights.extend((0.55 * factor, 0.30 * factor))

            lexical_docs: tuple[Document, ...] = ()
            if index == 0 and canonical_chunks:
                from backend.core.retrieval.lexical_recall import retrieve_lexical

                lexical_hits = retrieve_lexical(
                    query,
                    canonical_chunks,
                    query_frame=state.get("query_frame"),
                    limit=30,
                    min_score=0.28,
                )
                lexical_docs = tuple(hit.document for hit in lexical_hits)

            retrieval_results.append(lexical_docs)
            lexical_recall_counts.append(len(lexical_docs))
            weights.append(0.15 * factor)

        return {
            "retrieval_results": retrieval_results,
            "retrieval_weights": tuple(weights),
            "retrieval_queries": queries,
            "lexical_recall_count": sum(lexical_recall_counts),
            "lexical_recall_queries": (queries[0],) if canonical_chunks else (),
        }

    def fuse_retrieved_documents_node(self, state: State) -> State:
        lists = tuple(
            state.get("retrieval_results", ())
        )
        weights = tuple(
            state.get("retrieval_weights", ())
        )
        queries = tuple(
            state.get("retrieval_queries", ())
        )
        primary = _clean(
            state.get("resolved_question")
            or state.get("question")
        )

        fused = self.deps.fuse_ranked_lists(
            lists,
            weights=weights if len(weights) == len(lists) else None,
            primary_query=primary,
            retrieval_queries=queries,
        )

        # The RRF adapter returns canonical candidates and therefore retains
        # provenance. Nothing here turns provenance into answer text.
        return {
            "fused_candidates": tuple(fused),
            "fused_docs": tuple(
                getattr(item, "document", item)
                for item in fused
            ),
        }

    def verify_and_rank_node(self, state: State) -> State:
        query = state.get("query")
        frame = state.get("query_frame")

        if query is None or frame is None:
            raise ValueError(
                "verify_and_rank_node requires both 'query' and 'query_frame'."
            )

        candidates = list(state.get("fused_candidates", ()))

        if not candidates:
            return {
                "verified_candidates": (),
                "uncertain_candidates": (),
                "rejected_candidates": (),
                "ranked_candidates": (),
                "lexical_fallback_used": False,
                "lexical_fallback_candidates": 0,
                "lexical_fallback_trace": (),
                "retrieval_trace": (),
                "verification_trace": (),
            }

        # First pass: use the full semantic/alignment-aware verifier on normal
        # dense/BM25/RRF candidates.
        enriched: list[Any] = []
        for candidate in candidates:
            document = getattr(candidate, "document", candidate)
            try:
                _, meaning, alignment = self.deps.align_query_to_document(
                    query.original_query,
                    document,
                    _load_semantic_registry(document, self.deps),
                )
                candidate = _with_alignment(
                    candidate,
                    alignment,
                    meaning,
                )
            except Exception:
                # Retrieval must continue when the optional registry adapter
                # cannot enrich a candidate. Verification remains conservative.
                pass
            enriched.append(candidate)

        batch = self.deps.verify_candidates(
            frame,
            enriched,
        )

        verified = list(getattr(batch, "verified", ()))
        uncertain = list(getattr(batch, "uncertain", ()))
        rejected = list(getattr(batch, "rejected", ()))
        verification_trace: list[dict[str, Any]] = []
        for decision in getattr(batch, "decisions", ()) or ():
            if hasattr(decision, "to_dict"):
                verification_trace.append(decision.to_dict())

        # Recall rescue: if strict semantic qualification produced no usable
        # candidates (or just one weak candidate), perform a cheap direct
        # lexical scan over the canonical chunk corpus. This preserves the old
        # word-to-word retrieval safety net without weakening explicit conflict
        # handling. The lexical candidates still pass through the same verifier.
        canonical_chunks = tuple(self.deps.canonical_chunks_provider(state) or ())
        lexical_fallback_used = False
        lexical_fallback_trace: tuple[dict[str, Any], ...] = ()

        if canonical_chunks and len(verified) < 2:
            fallback_state = dict(state)
            fallback_state["canonical_chunks"] = canonical_chunks
            fallback_candidates, lexical_fallback_trace = _lexical_recall_candidates(
                fallback_state,
                query.original_query,
                frame,
                enriched,
                limit=40,
            )
            if fallback_candidates:
                lexical_fallback_used = True
                fallback_batch = self.deps.verify_candidates(
                    frame,
                    fallback_candidates,
                )
                verified.extend(getattr(fallback_batch, "verified", ()) or ())
                uncertain.extend(getattr(fallback_batch, "uncertain", ()) or ())
                rejected.extend(getattr(fallback_batch, "rejected", ()) or ())
                for decision in getattr(fallback_batch, "decisions", ()) or ():
                    if hasattr(decision, "to_dict"):
                        verification_trace.append(decision.to_dict())

        # Deduplicate by stable candidate identity while preserving the order in
        # which verified evidence was established.
        def _dedupe_candidates(items: Sequence[Any]) -> tuple[Any, ...]:
            output: list[Any] = []
            seen: set[str] = set()
            for item in items:
                key = _clean(getattr(item, "document_id", ""))
                if not key:
                    key = f"{_document_source(getattr(item, 'document', item))}|{_document_text(getattr(item, 'document', item))}"
                if key in seen:
                    continue
                seen.add(key)
                output.append(item)
            return tuple(output)

        verified_tuple = _dedupe_candidates(verified)
        uncertain_tuple = _dedupe_candidates(uncertain)
        rejected_tuple = _dedupe_candidates(rejected)

        ranked = tuple(
            self.deps.rank_candidates(
                query.original_query,
                verified_tuple,
                top_k=10,
                query_frame=frame,
            )
        )

        retrieval_trace = tuple(
            _candidate_trace(candidate, index)
            for index, candidate in enumerate(ranked[:20], start=1)
        )

        return {
            "verified_candidates": verified_tuple,
            "uncertain_candidates": uncertain_tuple,
            "rejected_candidates": rejected_tuple,
            "ranked_candidates": ranked,
            "lexical_fallback_used": lexical_fallback_used,
            "lexical_fallback_candidates": len(lexical_fallback_trace),
            "lexical_fallback_trace": lexical_fallback_trace,
            "retrieval_trace": retrieval_trace,
            "verification_trace": tuple(verification_trace),
        }

    # ------------------------------------------------------------------
    # Local context / evidence preparation
    # ------------------------------------------------------------------

    def evidence_context_node(self, state: State) -> State:
        """Build final evidence context without bypassing verification.

        Context expansion is recall-only. It may discover adjacent or related
        chunks, but no newly introduced document can enter the final evidence
        set unless it satisfies the same canonical verification contract used
        for the primary retrieval candidates.

        The final boundary therefore enforces: 

        retrieval -> verification -> context expansion -> re-verification

        This is intentionally fail-closed when the semantic frame is missing.
        """
        query = state.get("query")
        frame = state.get("query_frame")

        if query is None:
            raise ValueError("evidence_context_node requires 'query'.")

        ranked_candidates = tuple(state.get("ranked_candidates", ()) or ())
        ranked_docs = tuple(
            getattr(candidate, "document", candidate)
            for candidate in ranked_candidates
        )

        # Without the canonical semantic frame we cannot safely promote newly
        # expanded context. Existing ranked/verified candidates remain usable.
        if frame is None:
            return {
                "evidence_groups": (),
                "evidence_documents": ranked_docs,
                "evidence_candidates": ranked_candidates,
                "rejected_context_documents": (),
            }

        canonical_chunks = tuple(
            self.deps.canonical_chunks_provider(state) or ()
        )

        groups = list(
            self.deps.build_evidence_groups(
                ranked_docs
            )
        )

        if canonical_chunks:
            groups = list(
                self.deps.expand_group_context(
                    groups,
                    canonical_chunks,
                )
            )

        grouped_docs = tuple(
            self.deps.flatten_evidence_groups(
                groups,
                max_documents=40,
            )
        )

        # Keep the existing generic scope filter as an inexpensive first pass.
        # It is deliberately not the authoritative admission decision; every
        # document is checked again below using the canonical verifier.
        scope_filtered = tuple(
            self.deps.filter_scope_conflicts(
                query,
                grouped_docs,
                query_semantics=query,
            )
        )

        scope_policy = None
        scope_policy_provider = getattr(
            self.deps,
            "scope_policy_provider",
            None,
        )
        if callable(scope_policy_provider):
            try:
                scope_policy = scope_policy_provider(state)
            except Exception:
                scope_policy = None

        # Use canonical source+content identity, never arbitrary chunk IDs.
        # This prevents independently produced retrieval/context objects from
        # being conflated simply because they share a metadata identifier.
        ranked_by_identity: dict[tuple[str, str], Any] = {}
        for candidate in ranked_candidates:
            document = getattr(
                candidate,
                "document",
                candidate,
            )
            ranked_by_identity[_document_identity_key(document)] = candidate

        context_candidates: list[Any] = []
        rejected_context: list[Any] = []

        for document in scope_filtered:
            identity = _document_identity_key(document)
            candidate = ranked_by_identity.get(identity)

            if candidate is None:
                # Expanded context is a new retrieval candidate. Construct the
                # canonical contract, then apply the institution scope policy
                # before invoking the same verifier used upstream.
                try:
                    candidate = _to_candidate(
                        document,
                        self.deps,
                        query.original_query,
                    )
                except Exception:
                    rejected_context.append(document)
                    continue

            candidate = _apply_scope_conflicts(
                candidate,
                scope_policy,
                query.original_query,
            )

            try:
                decision_batch = self.deps.verify_candidates(
                    frame,
                    (candidate,),
                )
                verified = tuple(
                    getattr(
                        decision_batch,
                        "verified",
                        (),
                    )
                    or ()
                )
            except Exception:
                verified = ()

            if not verified:
                rejected_context.append(document)
                continue

            # Keep the verifier-returned candidate rather than reconstructing
            # the object after verification, preserving its canonical meaning,
            # alignment, provenance, and verification-derived state.
            context_candidates.append(verified[0])

        accepted_documents = tuple(
            getattr(candidate, "document", candidate)
            for candidate in context_candidates
        )

        return {
            "evidence_groups": tuple(groups),
            "evidence_documents": accepted_documents,
            "evidence_candidates": tuple(context_candidates),
            "rejected_context_documents": tuple(rejected_context),
        }

    # ------------------------------------------------------------------
    # Evidence qualification
    # ------------------------------------------------------------------

    def assess_evidence_node(self, state: State) -> State:
        query = state.get("query")
        candidates = tuple(
            state.get("ranked_candidates", ())
        )

        result = self.deps.assess_evidence(
            candidates,
            query=query,
            candidates_are_verified=True,
        )

        return {
            "evidence_assessment": result,
            "evidence_status": _status(result),
            "evidence_score": float(
                getattr(result, "score", 0.0) or 0.0
            ),
            "relevant_evidence_documents": int(
                getattr(result, "relevant_documents", 0) or 0
            ),
        }

    def assess_coverage_node(self, state: State) -> State:
        query = state.get("query")
        candidates = tuple(
            state.get("ranked_candidates", ())
        )

        result = self.deps.assess_coverage(
            query,
            candidates,
        )

        return {
            "coverage_assessment": result,
            "evidence_coverage_status": _status(result),
            "evidence_question_type": _clean(
                getattr(result, "question_type", "")
            ) or _clean(
                getattr(query, "request_type", "")
            ) or "descriptive",
        }

    def audit_claims_node(self, state: State) -> State:
        query = state.get("query")
        candidates = tuple(
            state.get("evidence_candidates")
            or state.get("ranked_candidates", ())
        )

        claims = tuple(
            self.deps.extract_claims(query)
        )
        units = tuple(
            self.deps.extract_evidence_units(candidates)
        )
        audit = self.deps.audit_claims(
            claims,
            units,
        )

        return {
            "claims": claims,
            "evidence_units": units,
            "claim_audit": audit,
            "claim_audit_status": _status(audit),
        }

    def package_evidence_node(self, state: State) -> State:
        units = tuple(
            state.get("evidence_units", ())
        )
        audit = state.get("claim_audit")

        package = self.deps.build_evidence_package(
            units,
            claim_audit=audit,
            include_partial=True,
            verified_evidence=True,
        )

        return {
            "evidence_package": package,
            "final_evidence_context": _clean(
                getattr(package, "context", "")
            ),
            "evidence_package_status": _status(
                package,
                default="empty",
            ),
            "ready_for_generation": bool(
                getattr(package, "ready_for_generation", False)
            ),
        }

    # ------------------------------------------------------------------
    # One-answer generation + E7.2 + E8
    # ------------------------------------------------------------------

    def answer_node(self, state: State) -> State:
        institution = self.deps.institution_provider(state)
        fallback = self.deps.fallback_provider(
            state,
            institution,
        )

        package = state.get("evidence_package")
        evidence_status = _status(
            state.get("evidence_assessment"),
            default=_clean(state.get("evidence_status")) or "insufficient",
        )
        coverage_status = _clean(
            state.get("evidence_coverage_status")
        ).casefold() or "insufficient"

        # Only missing/conflicted evidence stops generation. Coverage is a
        # completeness signal, not a blanket generation veto: a grounded model
        # can answer the supported portion and clearly state what the evidence
        # does not establish.
        if (
            package is None
            or not bool(getattr(package, "ready_for_generation", False))
            or evidence_status in {"insufficient", "conflicted"}
        ):
            guarded = self.deps.guard_answer(
                "",
                fallback=fallback,
            )
            return {
                "answer": _answer_text(guarded),
                "answer_generation": None,
                "answer_grounding": None,
                "answer_guard": guarded,
                "answer_grounding_status": "not_run",
                "answer_guard_status": _clean(
                    getattr(guarded, "status", "")
                    or (
                        guarded.get("status")
                        if isinstance(guarded, Mapping)
                        else ""
                    )
                ),
                "answer_guard_fallback_used": True,
                "answer_guard_reason": "insufficient_evidence",
                "context": "",
                "model_calls": 0,
                "answer_mode": "fallback",
            }

        question = _clean(
            state.get("resolved_question")
            or state.get("question")
        )

        if state.get("is_multi_intent"):
            intent_questions = tuple(
                _sequence(state.get("intent_questions"))
            )
            question = (
                "Answer each user request separately. "
                "Do not merge evidence or requirements between requests. "
                "Use only the verified evidence supplied for these requests. "
                "When one request lacks sufficient support, say so for that "
                "request while still answering any other supported request.\n\n"
                + "\n".join(
                    f"{index}. {_clean(item)}"
                    for index, item in enumerate(
                        intent_questions,
                        start=1,
                    )
                    if _clean(item)
                )
            )

        model = self.deps.answer_model_provider(state)

        request = self.deps.answer_request_type(
            question=question,
            evidence=package,
            institution=institution,
            chat_history=tuple(
                _sequence(state.get("chat_history"))
            ),
            question_type=_clean(
                state.get("evidence_question_type")
            ) or None,
            evidence_coverage=coverage_status,
            required_entities=tuple((_clean(getattr(e,"name","")),_clean(getattr(e,"entity_type",""))) for e in (getattr(state.get("query"),"entities",()) or ()) if _clean(getattr(e,"name",""))),
            known_program_names=tuple(state.get("known_program_aliases",()) or ()),
        )

        generation = self.deps.answer_generator(
            request,
            model,
        )
        generated_answer = _answer_text(generation)

        if not generated_answer:
            guarded = self.deps.guard_answer(
                "",
                fallback=fallback,
            )
            return {
                "answer": _answer_text(guarded),
                "answer_generation": generation,
                "answer_grounding": None,
                "answer_guard": guarded,
                "answer_grounding_status": "not_run",
                "answer_guard_status": _clean(
                    getattr(guarded, "status", "")
                    or (
                        guarded.get("status")
                        if isinstance(guarded, Mapping)
                        else ""
                    )
                ),
                "answer_guard_fallback_used": True,
                "answer_guard_reason": "empty_generation",
                "context": _clean(getattr(package, "context", "")),
                "model_calls": int(
                    getattr(generation, "model_calls", 0) or 0
                ),
                "answer_mode": _clean(getattr(generation, "answer_mode", "empty")) or "empty",
            }

        grounding = self.deps.assess_answer_grounding(
            answer=generated_answer,
            evidence=_clean(getattr(package, "context", "")),
        )

        grounding_status = _status(
            grounding,
            default="grounded",
        )

        if grounding_status in {"review", "conflict", "unsafe"}:
            guarded = self.deps.guard_answer(
                "",
                fallback=fallback,
            )
            final_answer = _answer_text(guarded)
            guard_reason = "answer_grounding_review"
        else:
            guarded = self.deps.guard_answer(
                generated_answer,
                fallback=fallback,
            )
            final_answer = _answer_text(guarded)
            guard_reason = _clean(
                getattr(guarded, "reason", "")
                or (
                    guarded.get("reason")
                    if isinstance(guarded, Mapping)
                    else ""
                )
            )

        return {
            "answer": final_answer,
            "context": _clean(
                getattr(package, "context", "")
            ),
            "answer_generation": generation,
            "answer_grounding": grounding,
            "answer_guard": guarded,
            "answer_grounding_status": grounding_status,
            "answer_guard_status": _clean(
                getattr(guarded, "status", "")
                or (
                    guarded.get("status")
                    if isinstance(guarded, Mapping)
                    else ""
                )
            ),
            "answer_guard_fallback_used": bool(
                getattr(guarded, "fallback_used", False)
                if not isinstance(guarded, Mapping)
                else guarded.get("fallback_used", False)
            ),
            "answer_guard_reason": (
                guard_reason or "guard_completed"
            ),
            "answer_mode": _clean(getattr(generation, "answer_mode", "model")) or "model",
            "model_calls": int(
                getattr(generation, "model_calls", 0) or 0
            ),
        }

    # ------------------------------------------------------------------
    # Composite graph-ready pipelines
    # ------------------------------------------------------------------

    def retrieve_and_qualify_node(self, state: State) -> State:
        """Run the deterministic retrieval/evidence path for one query."""
        result: State = {}
        result.update(self.understand_query_node(state))
        merged = dict(state)
        merged.update(result)

        result.update(self.hybrid_retrieve_node(merged))
        merged.update(result)

        result.update(self.fuse_retrieved_documents_node(merged))
        merged.update(result)

        result.update(self.verify_and_rank_node(merged))
        merged.update(result)

        # Qualification can fail closed before expensive context expansion.
        result.update(self.assess_evidence_node(merged))
        merged.update(result)

        result.update(self.assess_coverage_node(merged))
        merged.update(result)

        result.update(self.evidence_context_node(merged))
        merged.update(result)

        result.update(self.audit_claims_node(merged))
        merged.update(result)

        result.update(self.package_evidence_node(merged))
        return result

    def process_multi_intent_node(self, state: State) -> State:
        intent_units = tuple(
            _sequence(state.get("intent_units"))
        )

        intent_results: list[dict[str, Any]] = []
        all_units: list[Any] = []
        all_claims: list[Any] = []
        all_matches: list[Any] = []
        all_supported: list[str] = []
        all_partial: list[str] = []
        all_unsupported: list[str] = []
        all_conflicts: list[str] = []

        for index, unit in enumerate(intent_units, start=1):
            question = _clean(
                unit.get("question")
                if isinstance(unit, Mapping)
                else getattr(unit, "question", "")
            )
            if not question:
                continue

            child = dict(state)
            child["question"] = question
            child["resolved_question"] = question
            child["chat_history"] = ()

            child_result = self.retrieve_and_qualify_node(
                child
            )

            audit = child_result.get("claim_audit")
            if audit is not None:
                all_claims.extend(
                    _sequence(getattr(audit, "claims", ()))
                )
                all_matches.extend(
                    _sequence(getattr(audit, "matches", ()))
                )
                all_supported.extend(
                    _sequence(getattr(audit, "supported_claim_ids", ()))
                )
                all_partial.extend(
                    _sequence(getattr(audit, "partial_claim_ids", ()))
                )
                all_unsupported.extend(
                    _sequence(getattr(audit, "unsupported_claim_ids", ()))
                )
                all_conflicts.extend(
                    _sequence(getattr(audit, "conflicting_claim_ids", ()))
                )

            all_units.extend(
                _sequence(child_result.get("evidence_units"))
            )

            intent_results.append(
                {
                    "intent_index": index,
                    "question": question,
                    "evidence_status": child_result.get(
                        "evidence_status",
                        "insufficient",
                    ),
                    "evidence_coverage_status": child_result.get(
                        "evidence_coverage_status",
                        "insufficient",
                    ),
                    "evidence_package": child_result.get(
                        "evidence_package"
                    ),
                    "evidence_units": child_result.get(
                        "evidence_units",
                        (),
                    ),
                }
            )

        # A single composite audit gives E6 one deterministic package while
        # preserving claim IDs from each independent intent audit.
        composite_audit = None
        if all_claims:
            from backend.core.evidence.claims import ClaimAudit

            composite_audit = ClaimAudit(
                claims=tuple(all_claims),
                matches=tuple(all_matches),
                supported_claim_ids=tuple(
                    dict.fromkeys(
                        str(value) for value in all_supported
                    )
                ),
                partial_claim_ids=tuple(
                    dict.fromkeys(
                        str(value) for value in all_partial
                    )
                ),
                unsupported_claim_ids=tuple(
                    dict.fromkeys(
                        str(value) for value in all_unsupported
                    )
                ),
                conflicting_claim_ids=tuple(
                    dict.fromkeys(
                        str(value) for value in all_conflicts
                    )
                ),
            )

        package = self.deps.build_evidence_package(
            tuple(all_units),
            claim_audit=composite_audit,
            include_partial=True,
            verified_evidence=True,
        )

        statuses = [
            _clean(item.get("evidence_status")).casefold()
            for item in intent_results
        ]

        overall_status = (
            "insufficient"
            if not statuses or all(item == "insufficient" for item in statuses)
            else "supported"
        )

        overall_coverage = (
            "insufficient"
            if not statuses or all(
                _clean(item.get("evidence_coverage_status")).casefold()
                == "insufficient"
                for item in intent_results
            )
            else "supported"
        )

        return {
            "intent_results": tuple(intent_results),
            "evidence_package": package,
            "evidence_units": tuple(all_units),
            "claim_audit": composite_audit,
            "evidence_status": overall_status,
            "evidence_coverage_status": overall_coverage,
            "evidence_question_type": "multi_intent",
            "ready_for_generation": bool(
                getattr(package, "ready_for_generation", False)
            ),
        }


# ---------------------------------------------------------------------------
# Lazy module-level wrappers for LangGraph
# ---------------------------------------------------------------------------

_DEFAULT_NODES: CoreNodes | None = None


def _nodes() -> CoreNodes:
    global _DEFAULT_NODES
    if _DEFAULT_NODES is None:
        _DEFAULT_NODES = CoreNodes()
    return _DEFAULT_NODES


def resolve_conversation_node(state: State) -> State:
    return _nodes().resolve_conversation_node(state)


def understand_query_node(state: State) -> State:
    return _nodes().understand_query_node(state)


def plan_multi_intent_node(state: State) -> State:
    return _nodes().plan_multi_intent_node(state)


def hybrid_retrieve_node(state: State) -> State:
    return _nodes().hybrid_retrieve_node(state)


def fuse_retrieved_documents_node(state: State) -> State:
    return _nodes().fuse_retrieved_documents_node(state)


def verify_and_rank_node(state: State) -> State:
    return _nodes().verify_and_rank_node(state)


def evidence_context_node(state: State) -> State:
    return _nodes().evidence_context_node(state)


def assess_evidence_node(state: State) -> State:
    return _nodes().assess_evidence_node(state)


def assess_coverage_node(state: State) -> State:
    return _nodes().assess_coverage_node(state)


def audit_claims_node(state: State) -> State:
    return _nodes().audit_claims_node(state)


def package_evidence_node(state: State) -> State:
    return _nodes().package_evidence_node(state)


def process_multi_intent_node(state: State) -> State:
    return _nodes().process_multi_intent_node(state)


def answer_node(state: State) -> State:
    return _nodes().answer_node(state)


__all__ = [
    "NodeDependencies",
    "CoreNodes",
    "default_dependencies",
    "resolve_conversation_node",
    "understand_query_node",
    "plan_multi_intent_node",
    "hybrid_retrieve_node",
    "fuse_retrieved_documents_node",
    "verify_and_rank_node",
    "evidence_context_node",
    "assess_evidence_node",
    "assess_coverage_node",
    "audit_claims_node",
    "package_evidence_node",
    "process_multi_intent_node",
    "answer_node",
]
