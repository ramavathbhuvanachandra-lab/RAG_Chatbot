"""Canonical Assistant Service.

Provides a decoupled, stable service boundary between client frontends (such as
the Streamlit UI, CLI, or future API gateways) and the canonical AI platform core.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ai_platform.institutions.loader import (
    load_institution_profile,
    get_institution_handler,
)
from ai_platform.runtime.engine import ask_with_diagnostics
from ai_platform.service.suggestions import generate_follow_up_suggestions

logger = logging.getLogger(__name__)

# =============================================================================
# Routing Keywords
# =============================================================================

NAVIGATION_KEYWORDS: tuple[str, ...] = (
    "where is",
    "where are",
    "location",
    "locate",
    "map",
    "maps",
    "navigate",
    "navigation",
    "direction",
    "directions",
    "how do i reach",
    "how to reach",
    "how do i get",
    "how to get",
    "take me to",
    "route to",
    "way to",
    "where can i buy",
    "where can i get",
    "where can i find",
    "where do i get",
    "where to find",
    "where should i go",
    "where do i go",
    "where to go",
    "where to buy",
    "where to eat",
    "where can i eat",
    "places to eat",
    "is there a shop",
    "shop nearby",
    "nearby shop",
    "nearby store",
    "food joints",
    "food spots",
    "food places",
    "kaha milega",
    "kahan milega",
    "kidhar milega",
    "kaha milegi",
    "kidhar milegi",
    "kaha hai",
    "kahan hai",
    "kidhar hai",
    "kaha pe",
    "kahan pe",
    "kaise jau",
    "kaise jaye",
    "kaise pahunche",
    "kaise pahuchen",
    "rasta",
)

INFORMATION_KEYWORDS: tuple[str, ...] = (
    "timing",
    "timings",
    "time",
    "hours",
    "working hours",
    "open",
    "close",
    "fee",
    "fees",
    "admission",
    "hostel",
    "facility",
    "facilities",
    "department",
    "course",
    "syllabus",
    "process",
    "procedure",
    "rules",
    "eligibility",
    "contact",
    "email",
    "what",
    "when",
    "why",
    "who",
    "which",
    "explain",
    "tell me",
    "provide",
    "information",
    "details",
)


# =============================================================================
# Service Request & Response Contracts
# =============================================================================

@dataclass(frozen=True)
class ChatTurnRequest:
    """Request payload representing one conversational turn."""
    question: str
    chat_history: Sequence[dict[str, Any]] | Sequence[Any] = field(default_factory=tuple)
    user_id: str | None = None
    session_id: str | None = None
    institution_id: str | None = None


@dataclass(frozen=True)
class ChatTurnResponse:
    """Structured response payload returned by the assistant service."""
    answer: str
    structured_content: list[dict[str, Any]] = field(default_factory=list)
    formatted_response: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    is_safe_refusal: bool = False
    evidence_status: str = "supported"
    diagnostics: dict[str, Any] | None = None
    suggested_questions: list[str] = field(default_factory=list)


# =============================================================================
# Structured Presentation Formatting
# =============================================================================

def render_structured_blocks(structured_content: list[dict[str, Any]]) -> str:
    """Format structured metadata blocks (navigation, emergency) to markdown."""
    rendered_blocks: list[str] = []

    for item in structured_content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        data = item.get("data")
        if not isinstance(data, dict):
            continue

        if item_type in {"navigation", "emergency"}:
            lines: list[str] = []
            name = data.get("name")
            if name:
                lines.append(f"📍 **{name}**\n")
            desc = data.get("description")
            if desc:
                lines.append(f"{desc}\n")
            timings = data.get("timings")
            if timings:
                lines.append(f"🕒 **Timings:** {timings}\n")
            maps = data.get("google_maps")
            if maps:
                lines.append(f"🗺️ **Google Maps:**\n{maps}\n")

            if lines:
                rendered_blocks.append("\n".join(lines).strip())

    return "\n\n".join(rendered_blocks).strip()


# =============================================================================
# Routing Logic
# =============================================================================

def is_navigation_intent(question: str) -> bool:
    """Detect whether query phrasing expresses a navigation or location intent."""
    normalized = question.casefold()
    return any(keyword in normalized for keyword in NAVIGATION_KEYWORDS)


def needs_rag_answering(question: str, has_structured_match: bool) -> bool:
    """Determine whether the canonical RAG engine must run.

    Contract:
    - If no structured handler matched, RAG is always invoked.
    - If a structured handler matched, RAG is only invoked if the query also
      expresses an informational ask (e.g., fee, timing, rules, explanation).
    """
    if not has_structured_match:
        return True

    normalized = question.casefold()
    return any(keyword in normalized for keyword in INFORMATION_KEYWORDS)


# =============================================================================
# Main Service Handler
# =============================================================================

def handle_chat_turn(
    request: ChatTurnRequest | str,
    *,
    chat_history: Sequence[Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    institution_id: str | None = None,
) -> ChatTurnResponse:
    """Execute a single conversational turn through the canonical platform.

    Parameters:
        request: A `ChatTurnRequest` instance, or a plain question string.
        chat_history: Optional history sequence if `request` is a plain string.
        user_id: Optional user identifier if `request` is a plain string.
        session_id: Optional session identifier if `request` is a plain string.
        institution_id: Optional institution identifier if `request` is a plain string.

    Returns:
        `ChatTurnResponse` containing the answer, structured blocks, sources, and latency.
    """
    import uuid
    turn_id = f"turn_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    start_time_ns = time.perf_counter_ns()
    start_time = time.perf_counter()

    if isinstance(request, str):
        req = ChatTurnRequest(
            question=request,
            chat_history=tuple(chat_history or ()),
            user_id=user_id,
            session_id=session_id,
            institution_id=institution_id,
        )
    elif isinstance(request, ChatTurnRequest):
        req = request
    else:
        raise TypeError(f"request must be ChatTurnRequest or str, got {type(request).__name__}")

    clean_question = " ".join(str(req.question or "").strip().split())
    if not clean_question:
        return ChatTurnResponse(
            answer="",
            formatted_response="",
            latency_seconds=0.0,
            evidence_status="empty_query",
        )

    # 1. Resolve institution profile
    inst_id = req.institution_id
    try:
        profile = load_institution_profile(inst_id)
        active_inst_id = profile.institution_id
    except Exception as exc:
        logger.warning("Could not load profile for institution %s: %s", inst_id, exc)
        profile = None
        active_inst_id = inst_id or "iitj"

    # Resolve conversational anaphora and referents if history is present
    resolved_question = clean_question
    if req.chat_history:
        try:
            from ai_platform.core.conversation.conversation import resolve_conversation
            conv_res = resolve_conversation(question=clean_question, chat_history=req.chat_history)
            if conv_res and conv_res.get("resolved_question"):
                resolved_question = conv_res["resolved_question"]
        except Exception as c_exc:
            logger.debug("Conversation resolution failed: %s", c_exc)

    # 2. Query structured institution handlers (navigation, emergency)
    structured_content: list[dict[str, Any]] = []

    # Check navigation handler if query indicates navigation intent or matches an entity
    nav_handler = get_institution_handler(active_inst_id, "navigation")
    if callable(nav_handler):
        try:
            location = nav_handler(resolved_question) or nav_handler(clean_question)
            if location and isinstance(location, dict):
                structured_content.append({"type": "navigation", "data": location})
        except Exception as exc:
            logger.exception("Structured navigation handler failed for %s: %s", active_inst_id, exc)

    # Check emergency handler
    emergency_handler = get_institution_handler(active_inst_id, "emergency")
    if callable(emergency_handler):
        try:
            emergency = emergency_handler(resolved_question) or emergency_handler(clean_question)
            if emergency and isinstance(emergency, dict):
                structured_content.append({"type": "emergency", "data": emergency})
        except Exception as exc:
            logger.exception("Structured emergency handler failed for %s: %s", active_inst_id, exc)

    structured_text = render_structured_blocks(structured_content)
    has_structured = bool(structured_content)

    # 3. Determine if RAG execution is required
    should_run_rag = needs_rag_answering(resolved_question, has_structured) or needs_rag_answering(clean_question, has_structured)

    answer = ""
    sources: list[dict[str, Any]] = []
    is_safe_refusal = False
    evidence_status = "structured_direct" if has_structured and not should_run_rag else "supported"
    diagnostics: dict[str, Any] | None = None

    if has_structured and not should_run_rag:
        # Provide authentic provenance for structured responses
        for item in structured_content:
            data = item.get("data", {})
            loc_id = data.get("id", "location")
            loc_name = data.get("name", "Campus Facility")
            sources.append({
                "evidence_id": f"campus_location:{loc_id}",
                "source": "Campus Directory / Navigation Map",
                "heading": loc_name,
                "text": f"{loc_name}: {data.get('description', '')} ({data.get('directions', '')})",
                "url": data.get("google_maps", ""),
            })
        answer = structured_text

    if should_run_rag:
        try:
            diag_state = ask_with_diagnostics(
                clean_question,
                chat_history=tuple(req.chat_history or ()),
                institution_profile=profile,
            )
            t_rag_end = time.perf_counter()
            diagnostics = diag_state
            answer = str(diag_state.get("answer") or "").strip()

            answer_mode = str(diag_state.get("answer_mode") or "")
            guard_fallback = bool(diag_state.get("answer_guard_fallback_used"))
            is_safe_refusal = bool(
                diag_state.get("pipeline_stop")
                or diag_state.get("will_abstain")
                or guard_fallback
                or answer_mode == "fallback"
            )
            ans_lower = answer.casefold()
            ans_word_count = len(ans_lower.split())
            refusal_regexes = (
                r"\bdoes not contain\b.*\binformation\b",
                r"\bdoes not provide\b.*\binformation\b",
                r"\bno (?:specific )?information\b.*\b(?:available|provided|found|given)\b",
                r"\b(?:is|are)?\s*not\s+(?:provided|mentioned|available|specified|found)\s+in\s+the\b",
                r"\bthe provided evidence does not contain\b",
                r"\bbased on the provided evidence\b.*\bno\b",
                r"\b(i'm sorry|i am sorry),?\s*i (?:do not|don't) know\b",
                r"\bcould you please specify which programme\b",
                r"\bthe available official sources do not\b",
                r"\bcannot be (?:verified|confirmed)\s+from\s+the\s+available\b",
                r"\bout of scope\b",
            )
            is_explicit_refusal_text = any(re.search(pattern, ans_lower) for pattern in refusal_regexes)

            # A full refusal either has short length (<=55 words) with an abstention pattern,
            # or starts with an explicit refusal statement.
            is_full_refusal = False
            if is_explicit_refusal_text or ("no official" in ans_lower and "available" in ans_lower):
                if ans_word_count <= 55 or ans_lower.startswith((
                    "i'm sorry", "i cannot", "the available official", "could you please",
                    "the provided evidence does not contain", "based on the provided evidence, there is no",
                    "the provided documents do not contain", "information regarding"
                )):
                    is_full_refusal = True

            if is_full_refusal:
                is_safe_refusal = True

            evidence_status = str(diag_state.get("evidence_status") or ("refusal" if is_safe_refusal else "supported"))
            if is_safe_refusal:
                evidence_status = "refusal"

            # Extract source evidence units for transparent UI citations ONLY when answer is supported
            # Abstentions/refusals MUST NOT emit citations implying that unrelated documents support the answer.
            package = diag_state.get("evidence_package")
            if not is_safe_refusal and evidence_status == "supported" and package is not None and hasattr(package, "items"):
                source_candidates: dict[str, dict[str, Any]] = {}
                _GENERIC_CITATION_STOPWORDS = {
                    "iitj", "jodhpur", "indian", "institute", "technology", "department", "departments",
                    "engineering", "program", "programs", "programme", "programmes", "degree", "academic",
                    "course", "courses", "student", "students", "faculty", "center", "centre", "school",
                    "schools", "office", "section", "about", "their", "there", "these", "those", "which",
                    "where", "what", "when", "with", "from", "into", "also", "have", "been", "will", "would",
                    "could", "should", "more", "most", "some", "such", "than", "that", "this", "they", "them",
                    "then", "under", "over", "other", "only", "well", "both", "each", "every", "many", "much",
                }
                ans_words = {
                    w for w in re.findall(r"[a-z0-9]+", ans_lower)
                    if (len(w) >= 3 or (len(w) >= 2 and w.isdigit())) and w not in _GENERIC_CITATION_STOPWORDS
                }
                ans_numbers = {w for w in re.findall(r"\b\d+\b", ans_lower) if len(w) >= 2}

                for item_index, item in enumerate(package.items):
                    src_path = str(getattr(item, "source", "") or "").replace("\\", "/")
                    if not src_path:
                        continue
                    norm_key = src_path.casefold()
                    item_text = str(getattr(item, "text", "") or "").strip()
                    item_words = {
                        w for w in re.findall(r"[a-z0-9]+", item_text.casefold())
                        if (len(w) >= 3 or (len(w) >= 2 and w.isdigit())) and w not in _GENERIC_CITATION_STOPWORDS
                    }
                    overlap = len(ans_words & item_words)
                    num_overlap = len(ans_numbers & item_words)
                    rank_bonus = max(0.0, 6.0 - (0.3 * item_index))
                    score = overlap + (2.0 * num_overlap) + rank_bonus

                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", item_text) if s.strip()]
                    excerpt = " ".join(sentences[:2]) if sentences else item_text[:180]
                    if len(excerpt) > 220:
                        excerpt = excerpt[:217] + "..."

                    # If this source is not yet recorded or this chunk has higher overlap, update
                    if norm_key not in source_candidates or score > source_candidates[norm_key]["score"]:
                        source_candidates[norm_key] = {
                            "overlap": overlap,
                            "score": score,
                            "evidence_id": str(getattr(item, "evidence_id", "") or ""),
                            "source": src_path,
                            "heading": str(getattr(item, "heading", "") or ""),
                            "text": excerpt,
                        }

                # Sort distinct sources by their actual overlap/support with the answer text
                sorted_sources = sorted(
                    source_candidates.values(),
                    key=lambda s: s["score"],
                    reverse=True,
                )

                max_score = sorted_sources[0]["score"] if sorted_sources else 0
                threshold = max(3.0, max_score * 0.40) if max_score >= 4 else max(2.0, max_score)

                # Keep only distinct sources that actually substantiate substantive claims in the answer
                for s in sorted_sources:
                    if s["score"] >= threshold and s["overlap"] >= 2:
                        sources.append({
                            "evidence_id": s["evidence_id"],
                            "source": s["source"],
                            "heading": s["heading"],
                            "text": s["text"],
                        })
                    if len(sources) >= 4:
                        break

            # Build decoupled Top-15 retrieval diagnostics
            try:
                from ai_platform.service.retrieval_diagnostics import build_retrieval_diagnostics
                diagnostics = build_retrieval_diagnostics(
                    diag_state,
                    answer=answer,
                    sources=sources,
                    candidate_window=15,
                    tenant_id=active_inst_id,
                )
            except Exception as d_exc:
                logger.warning("Could not build detailed retrieval diagnostics: %s", d_exc)
                diagnostics = diag_state

        except Exception as exc:
            logger.exception("Canonical RAG engine execution error: %s", exc)
            answer = f"❌ Error:\n\n{exc}"
            is_safe_refusal = True
            evidence_status = "error"

    end_time_ns = time.perf_counter_ns()
    elapsed_time = (end_time_ns - start_time_ns) / 1_000_000_000.0
    post_proc_time = time.perf_counter() - t_rag_end if "t_rag_end" in locals() else 0.0

    # Ensure total pipeline latency and terminal status are explicitly recorded in diagnostics
    terminal_st = "error" if evidence_status == "error" else ("safe_refusal" if is_safe_refusal else "success")
    if diagnostics is None or not isinstance(diagnostics, dict):
        diagnostics = {
            "turn_id": turn_id,
            "resolved_question": resolved_question or clean_question,
            "route_selected": "direct_navigation" if has_structured and not should_run_rag else "direct_handler",
            "execution_route": "direct_navigation" if has_structured and not should_run_rag else "direct_handler",
            "latency_seconds": elapsed_time,
            "pipeline_start_ns": start_time_ns,
            "pipeline_end_ns": end_time_ns,
            "pipeline_start": start_time,
            "pipeline_end": time.perf_counter(),
            "terminal_status": terminal_st,
            "answer": structured_text or answer,
            "sources": sources,
            "funnel": {
                "retrieved_total": 0,
                "top_window_count": 0,
                "verified_count": 0,
                "uncertain_count": 0,
                "rejected_count": 0,
                "ranked_count": 0,
                "context_units_count": 0,
                "displayed_citations_count": len(sources),
                "route": "direct_navigation",
                "route_description": "Direct Campus Navigation (Fast Path)",
            },
            "section_a_candidates": [],
            "section_b_context": {
                "context_units_count": 0,
                "context_characters": 0,
                "estimated_tokens": 0,
                "token_budget": 4000,
                "token_budget_percentage": 0.0,
                "claim_audit_status": "bypassed_direct_handler",
                "supported_claims": [],
                "partial_claims": [],
                "omitted_claims": [],
                "packaged_units": [],
            },
            "section_c_citations": sources,
            "panel_query_understanding": {
                "user_input": {
                    "original_question": clean_question,
                    "resolved_question": resolved_question or clean_question,
                    "session_id": req.session_id or "default_session",
                    "chat_history": list(req.chat_history or []),
                },
                "interpretation": {
                    "resolved_question": resolved_question or clean_question,
                    "conversation_mode": "standalone",
                    "query_route": "direct_navigation",
                    "query_scope": "in_scope",
                    "query_scope_confidence": 1.0,
                    "detected_entities": [s.get("heading") for s in sources if s.get("heading")],
                    "detected_attributes": ["location", "directions", "campus_facility"],
                },
                "route_selection": {
                    "selected_route": "direct_navigation",
                    "route_reason": "Query matches registered campus facility location; routed to direct navigation fast-path handler.",
                    "qu_llm_calls": 0,
                    "total_qu_latency_seconds": elapsed_time,
                    "operation_timings": {"direct_navigation": elapsed_time},
                },
                "retrieval_handoff": {
                    "recipient_node": "campus_navigation",
                    "contract_validation_status": "SKIPPED_DIRECT_HANDLER",
                    "retrieval_queries": [],
                },
            },
            "panel_dense_retrieval": {
                "status": "intentionally_skipped",
                "reason": "Direct campus location match found in verified institutional directory. Full RAG dense vector retrieval bypassed.",
                "candidates": [],
                "candidate_count": 0,
            },
            "panel_bm25_retrieval": {
                "status": "intentionally_skipped",
                "reason": "Direct campus location match found in verified institutional directory. Full RAG BM25 lexical retrieval bypassed.",
                "candidates": [],
                "candidate_count": 0,
            },
            "panel_hybrid_fusion": {
                "status": "intentionally_skipped",
                "reason": "Direct structured response generated without vector/lexical candidate fusion.",
                "top_15_candidates": [],
                "total_fused_count": 0,
            },
            "panel_ranking_verification": {
                "status": "intentionally_skipped",
                "reason": "Structured institutional facility directory provides authenticated facts; candidate verification bypassed.",
                "verified_count": 0,
                "uncertain_count": 0,
                "rejected_count": 0,
                "ranked_count": 0,
                "decisions": [],
            },
            "panel_packaged_context": {
                "status": "intentionally_skipped",
                "context_units_count": 0,
                "estimated_tokens": 0,
                "packaged_units": [],
            },
            "panel_answer_generation": {
                "status": "direct_structured_render",
                "model_calls": 0,
                "generation_mode": "structured_card",
                "answer": structured_text or answer,
            },
            "panel_stage_timings": {
                "timings": {
                    "total_turn": elapsed_time,
                    "total_pipeline": elapsed_time,
                    "structured_handler": elapsed_time,
                },
                "inclusive_timings": {"total_pipeline": elapsed_time},
                "exclusive_timings": {"structured_handler": elapsed_time},
                "exclusive_sum": elapsed_time,
                "reconciliation_delta": 0.0,
                "warnings": [],
                "errors": [],
            },
        }
    else:
        diagnostics["turn_id"] = turn_id
        diagnostics["route_selected"] = diagnostics.get("route_selected", "rag_hybrid")
        diagnostics["execution_route"] = diagnostics.get("execution_route", "rag_hybrid")
        diagnostics["latency_seconds"] = elapsed_time
        diagnostics["pipeline_start_ns"] = start_time_ns
        diagnostics["pipeline_end_ns"] = end_time_ns
        diagnostics["pipeline_start"] = start_time
        diagnostics["pipeline_end"] = time.perf_counter()
        diagnostics["terminal_status"] = terminal_st
        if "panel_stage_timings" in diagnostics and isinstance(diagnostics["panel_stage_timings"], dict):
            timings_map = diagnostics["panel_stage_timings"].get("timings")
            if isinstance(timings_map, dict):
                timings_map["total_turn"] = elapsed_time
                timings_map["total_pipeline"] = elapsed_time
                timings_map["post_processing"] = post_proc_time
            excl_map = diagnostics["panel_stage_timings"].get("exclusive_timings")
            if isinstance(excl_map, dict):
                excl_map["post_processing"] = post_proc_time
                excl_sum = sum(v for k, v in excl_map.items() if isinstance(v, (int, float)))
                diagnostics["panel_stage_timings"]["exclusive_sum"] = round(excl_sum, 5)
                diagnostics["panel_stage_timings"]["reconciliation_delta"] = round(abs(elapsed_time - excl_sum), 5)

    # 4. Assemble presentation-ready formatted string
    if not answer and structured_text:
        answer = structured_text

    if structured_text and answer and answer != structured_text:
        formatted_response = f"{structured_text}\n\n---\n\n{answer}"
    elif structured_text:
        formatted_response = structured_text
    else:
        formatted_response = answer

    # 5. Generate lightweight, grounded follow-up suggestions (0 LLM invocations)
    suggested_questions = generate_follow_up_suggestions(
        clean_question,
        answer=answer,
        sources=sources,
        is_safe_refusal=is_safe_refusal,
        evidence_status=evidence_status,
    )

    return ChatTurnResponse(
        answer=answer,
        structured_content=structured_content,
        formatted_response=formatted_response.strip(),
        sources=sources,
        latency_seconds=elapsed_time,
        is_safe_refusal=is_safe_refusal,
        evidence_status=evidence_status,
        diagnostics=diagnostics,
        suggested_questions=suggested_questions,
    )



__all__ = [
    "ChatTurnRequest",
    "ChatTurnResponse",
    "render_structured_blocks",
    "is_navigation_intent",
    "needs_rag_answering",
    "handle_chat_turn",
    "generate_follow_up_suggestions",
]
