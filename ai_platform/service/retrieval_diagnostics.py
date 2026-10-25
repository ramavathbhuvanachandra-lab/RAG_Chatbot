"""Retrieval and Evidence Diagnostic Inspection Module.

Exposes structured Top-10 / Top-15 retrieval and evidence diagnostic tracking
across all stages:
  Retrieved (RRF) -> Verified -> Ranked -> Packaged -> Supports Claim -> Displayed Citation
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


DEFAULT_DIAGNOSTIC_WINDOW = 15
CONFIGURED_TOKEN_BUDGET = 4000


def _clean(val: Any) -> str:
    return " ".join(str(val or "").strip().split())


def _extract_path_metadata(source: str) -> dict[str, str]:
    norm = source.replace("\\", "/").casefold()
    department = ""
    programme = ""
    doc_type = "general"

    school_m = re.search(r"/schools/([a-z0-9_\-]+)", norm)
    if school_m:
        department = "School of " + school_m.group(1).replace("_", " ").title()

    dept_m = re.search(r"/departments/([a-z0-9_\-]+)", norm)
    if dept_m:
        department = dept_m.group(1).replace("_", " ").title()

    platform_m = re.search(r"/(?:research_platforms|centers|centres)/([a-z0-9_\-]+)", norm)
    if platform_m:
        department = platform_m.group(1).replace("_", " ").title()

    prog_m = re.search(r"/(?:programs|admissions)/([a-z0-9_\-]+)", norm)
    if prog_m:
        programme = prog_m.group(1).replace("_", " ").upper()

    if "fee" in norm or "finance" in norm:
        doc_type = "finance_and_fees"
    elif "admission" in norm:
        doc_type = "admissions"
    elif "research" in norm:
        doc_type = "research"
    elif "curriculum" in norm or "course" in norm:
        doc_type = "curriculum"
    elif "progression" in norm or "regulation" in norm or "rule" in norm:
        doc_type = "academic_rules"
    elif "hostel" in norm or "mess" in norm or "dining" in norm:
        doc_type = "campus_life"
    elif "overview" in norm:
        doc_type = "overview"

    return {
        "department": department or "Institute-Wide",
        "programme": programme or "General",
        "doc_type": doc_type,
    }


def build_retrieval_diagnostics(
    state: Mapping[str, Any],
    answer: str,
    sources: Sequence[Mapping[str, Any]],
    candidate_window: int = DEFAULT_DIAGNOSTIC_WINDOW,
    tenant_id: str = "iitj",
) -> dict[str, Any]:
    """Build a comprehensive, decoupled diagnostic payload for the developer score box.

    Parameters:
        state: The LangGraph execution state from ask_with_diagnostics.
        answer: The final generated answer string.
        sources: The list of displayed citation dicts.
        candidate_window: Number of top candidates to expose (default: 15).
        tenant_id: Active institutional tenant ID.

    Returns:
        Structured dictionary with Section A, Section B, Section C, and funnel counts.
    """
    fused_candidates = tuple(state.get("fused_candidates", ()) or ())
    verified_candidates = tuple(state.get("verified_candidates", ()) or ())
    uncertain_candidates = tuple(state.get("uncertain_candidates", ()) or ())
    rejected_candidates = tuple(state.get("rejected_candidates", ()) or ())
    ranked_candidates = tuple(state.get("ranked_candidates", ()) or ())
    verification_trace = tuple(state.get("verification_trace", ()) or ())
    package = state.get("evidence_package")
    final_context = _clean(state.get("final_evidence_context", ""))
    claim_audit = state.get("claim_audit")

    # Build verification lookup by document_id and clean source
    v_map: dict[str, dict[str, Any]] = {}
    for decision in verification_trace:
        if isinstance(decision, Mapping):
            cid = str(decision.get("candidate_id") or "")
            v_map[cid] = {
                "status": str(decision.get("status") or "unknown"),
                "accepted": bool(decision.get("accepted")),
                "reasons": list(decision.get("reasons") or []),
                "coverage": float(decision.get("coverage", 0.0) or 0.0),
                "score": float(decision.get("score", 0.0) or 0.0),
                "scope_compatible": bool(decision.get("scope_compatible", True)),
                "conflict_detected": bool(decision.get("conflict_detected", False)),
            }

    # Ranked order lookup
    ranked_order: dict[str, int] = {}
    for idx, rc in enumerate(ranked_candidates, start=1):
        rc_id = str(getattr(rc, "document_id", "") or "")
        if rc_id:
            ranked_order[rc_id] = idx

    # Context items lookup
    context_sources: set[str] = set()
    context_doc_ids: set[str] = set()
    packaged_items_list: list[dict[str, Any]] = []

    if package is not None and hasattr(package, "items"):
        for it in package.items:
            s_path = str(getattr(it, "source", "") or "").replace("\\", "/")
            context_sources.add(s_path.casefold())
            ev_id = str(getattr(it, "evidence_id", "") or "")
            context_doc_ids.add(ev_id)
            packaged_items_list.append({
                "evidence_id": ev_id,
                "source": s_path,
                "heading": str(getattr(it, "heading", "") or ""),
                "text": str(getattr(it, "text", "") or "").strip(),
            })

    # Displayed citation sources
    displayed_source_paths = {
        str(s.get("source", "")).replace("\\", "/").casefold()
        for s in sources
        if isinstance(s, Mapping)
    }

    # Words in final answer for claim support estimation
    ans_words = set(re.findall(r"[a-z0-9]{4,}", answer.casefold()))

    # Build Candidate Lookup Sets
    verified_ids = {str(getattr(vc, "document_id", "") or "") for vc in verified_candidates}
    uncertain_ids = {str(getattr(uc, "document_id", "") or "") for uc in uncertain_candidates}
    rejected_ids = {str(getattr(rc, "document_id", "") or "") for rc in rejected_candidates}

    # Build Section A: Top 10-15 candidates
    candidates_top15: list[dict[str, Any]] = []
    window_items = fused_candidates[:candidate_window]

    for idx, cand in enumerate(window_items, start=1):
        doc = getattr(cand, "document", cand)
        prov = getattr(cand, "provenance", None)
        cand_id = str(getattr(cand, "document_id", "") or "")
        source_path = str(
            getattr(cand, "source", "")
            or (getattr(doc, "metadata", {}).get("source", "") if hasattr(doc, "metadata") else "")
        ).replace("\\", "/")

        norm_src = source_path.casefold()
        path_meta = _extract_path_metadata(source_path)

        # Retrieve verification info
        v_info = v_map.get(cand_id, {})
        if cand_id in v_map:
            v_status = v_info.get("status") or ("verified" if cand_id in verified_ids else "uncertain")
            v_reasons = v_info.get("reasons", [])
            v_score = v_info.get("score", getattr(cand, "final_score", 0.0))
        elif cand_id in verified_ids:
            v_status = "verified"
            v_reasons = ["verified_candidate"]
            v_score = getattr(cand, "final_score", 0.0)
        elif cand_id in uncertain_ids:
            v_status = "uncertain"
            v_reasons = ["uncertain_evidence"]
            v_score = getattr(cand, "final_score", 0.0)
        elif cand_id in rejected_ids:
            v_status = "rejected"
            v_reasons = ["rejected_by_policy"]
            v_score = getattr(cand, "final_score", 0.0)
        else:
            v_status = "unverified"
            v_reasons = []
            v_score = getattr(cand, "final_score", 0.0)

        # Check if in final context
        in_ctx = (norm_src in context_sources) or (cand_id in context_doc_ids)

        # Check if text supports claim in answer
        doc_text = str(getattr(doc, "page_content", "") or getattr(cand, "text", "") or "")
        doc_words = set(re.findall(r"[a-z0-9]{4,}", doc_text.casefold()))
        overlap_count = len(ans_words & doc_words)
        supports_claim = overlap_count >= 3 if ans_words else False

        # Displayed citation check
        is_citation = norm_src in displayed_source_paths

        # Dense / BM25 / RRF details
        dense_rank = getattr(prov, "dense_rank", None) if prov else None
        dense_score = getattr(prov, "dense_score", None) if prov else None
        bm25_rank = getattr(prov, "bm25_rank", None) if prov else None
        bm25_score = getattr(prov, "bm25_score", None) if prov else None
        rrf_score = round(float(getattr(prov, "rrf_score", 0.0) or 0.0), 6) if prov else 0.0

        candidates_top15.append({
            "initial_rank": idx,
            "final_rank": ranked_order.get(cand_id),
            "document_id": cand_id,
            "source": source_path,
            "title": str(getattr(doc, "metadata", {}).get("heading", "") or getattr(cand, "heading", "") or source_path.split("/")[-1]),
            "institution_id": tenant_id,
            "department": path_meta["department"],
            "programme": path_meta["programme"],
            "doc_type": path_meta["doc_type"],
            "dense_rank": dense_rank,
            "dense_score": round(float(dense_score), 4) if dense_score is not None else None,
            "bm25_rank": bm25_rank,
            "bm25_score": round(float(bm25_score), 4) if bm25_score is not None else None,
            "fusion_score": rrf_score,
            "fusion_method": "Weighted RRF (k=60)",
            "verification_status": v_status,
            "verification_score": round(float(v_score or 0.0), 4),
            "verification_reasons": v_reasons,
            "scope_compatible": v_info.get("scope_compatible", True),
            "conflict_detected": v_info.get("conflict_detected", False),
            "in_final_context": in_ctx,
            "supports_claim": supports_claim,
            "displayed_as_citation": is_citation,
            "text_snippet": doc_text[:280].strip() if doc_text else "",
        })

    # Section B: Context token usage & claim coverage
    estimated_tokens = len(final_context) // 4 if final_context else 0
    token_budget = CONFIGURED_TOKEN_BUDGET
    token_pct = round((estimated_tokens / max(1, token_budget)) * 100, 1)

    claim_audit_status = str(getattr(claim_audit, "status", "") or "not_evaluated")
    supported_claim_ids = list(getattr(package, "supported_claim_ids", ()) or [])
    partial_claim_ids = list(getattr(package, "partial_claim_ids", ()) or [])
    omitted_claim_ids = list(getattr(package, "omitted_claim_ids", ()) or [])

    # Section C: Final citations
    supporting_citations = []
    for s in sources:
        if isinstance(s, Mapping):
            supporting_citations.append({
                "source": s.get("source", ""),
                "heading": s.get("heading", ""),
                "text": s.get("text", ""),
                "evidence_id": s.get("evidence_id", ""),
            })

    # Raw dense and BM25 candidate lists
    dense_records = list(state.get("dense_retrieval_records", ()) or ())
    bm25_records = list(state.get("bm25_retrieval_records", ()) or ())

    # Stage timings
    stage_timings = dict(state.get("stage_timings", {}) or {})

    # Extract entities, attributes, dates, numbers, qualifiers from plan
    plan_obj = state.get("query_plan")
    detected_entities: list[str] = []
    detected_attributes: list[str] = []
    detected_constraints: list[str] = []
    detected_dates: list[str] = []
    detected_numbers: list[str] = []
    detected_qualifiers: list[str] = []

    if plan_obj:
        for it in (getattr(plan_obj, "intents", ()) or ()):
            if getattr(it, "target", None):
                detected_entities.append(str(it.target))
            detected_attributes.extend(str(a) for a in (getattr(it, "requested_attributes", ()) or ()))
            detected_constraints.extend(str(c) for c in (getattr(it, "constraints", ()) or ()))
            detected_qualifiers.extend(str(q) for q in (getattr(it, "qualifiers", ()) or ()))
        for pt in (getattr(plan_obj, "preserved_terms", ()) or ()):
            pt_str = str(pt)
            if re.search(r"\b(?:202[0-9]|201[0-9]|june|july|august|september|october|november|december)\b", pt_str, re.I):
                detected_dates.append(pt_str)
            elif re.search(r"\b(?:\d+(?:\.\d+)?|\d+%\s*|\d+\s*(?:credits?|hours?|cgpa))\b", pt_str, re.I):
                detected_numbers.append(pt_str)

    sel_route = str(state.get("selected_route") or ("A" if "fast_path" in str(state.get("query_route")) else ("B" if "lightweight" in str(state.get("query_route")) else "C")))
    init_route = str(state.get("initial_route") or sel_route)
    qu_planning_time = float(stage_timings.get("query_planning_llm", 0.0))
    if "qu_llm_calls" in state:
        qu_llms = int(state.get("qu_llm_calls") or 0)
    else:
        qu_llms = 1 if (sel_route == "C" and qu_planning_time > 0.0) else 0
    route_reason = str(state.get("route_reason") or "deterministic_route_selection")
    route_escalated = bool(state.get("route_escalated", False))

    retrieval_queries_list = list(state.get("retrieval_queries") or [state.get("question")])
    intent_units_list = list(state.get("intent_units") or [])

    # Map search queries to appropriate intent units
    search_queries_with_intents = []
    for q_idx, rq in enumerate(retrieval_queries_list):
        assoc_intent = intent_units_list[q_idx].get("question") if q_idx < len(intent_units_list) else (intent_units_list[0].get("question") if intent_units_list else str(state.get("resolved_question") or state.get("question")))
        search_queries_with_intents.append({
            "query": rq,
            "associated_intent": assoc_intent,
            "weight": 1.0 if q_idx == 0 else max(0.5, round(0.85 ** q_idx, 2)),
        })

    # Exact structured payload passed to hybrid retrieval node
    exact_handoff_payload = {
        "recipient_node": "CoreNodes.hybrid_retrieve_node" if not state.get("is_multi_intent") else "CoreNodes.process_multi_intent_node",
        "primary_query": str(state.get("resolved_question") or state.get("question")),
        "retrieval_queries": retrieval_queries_list,
        "vector_collection": tenant_id,
        "is_multi_intent": bool(state.get("is_multi_intent")),
        "intent_units_count": len(intent_units_list),
        "institution_id": tenant_id,
        "top_k_per_branch": 25,
    }

    # 8-panel structured dictionary for Developer Diagnostics
    panel_1_query_understanding = {
        "user_input": {
            "original_question": str(state.get("question") or ""),
            "chat_history": list(state.get("chat_history") or ()),
            "session_id": str(state.get("session_id") or ""),
        },
        "route_selection": {
            "selected_route": sel_route,
            "initial_route": init_route,
            "route_reason": route_reason,
            "escalation_happened": route_escalated,
            "selector_latency_seconds": round(float(stage_timings.get("route_selection", 0.0005)), 5),
            "qu_llm_calls": qu_llms,
            "llm_used": sel_route == "C",
            "llm_model": "qwen2.5:3b" if sel_route == "C" else "none (Python-only)",
            "operation_timings": {
                "route_selection": round(float(stage_timings.get("route_selection", 0.0005)), 5),
                "route_a_fast_path": round(float(stage_timings.get("route_a_fast_path", 0.0)), 5),
                "route_b_lightweight": round(float(stage_timings.get("route_b_lightweight", 0.0)), 5),
                "query_planning_llm": round(float(stage_timings.get("query_planning_llm", 0.0)), 5),
                "guard_and_scope": round(float(stage_timings.get("guard_and_scope", 0.0)), 5),
                "total_understand_query": round(float(stage_timings.get("understand_query", 0.0)), 5),
            },
            "total_qu_latency_seconds": round(float(stage_timings.get("understand_query", 0.0)), 5),
        },
        "interpretation": {
            "original_question": str(state.get("question") or ""),
            "resolved_question": str(state.get("resolved_question") or state.get("question") or ""),
            "conversation_mode": str(state.get("conversation_mode") or "standalone"),
            "active_topic": str(state.get("active_topic") or ""),
            "active_entity": str(state.get("active_entity") or ""),
            "unanchored_reference": bool(state.get("unanchored_reference")),
            "is_multi_intent": bool(state.get("is_multi_intent")),
            "intent_questions": list(state.get("intent_questions") or []),
            "detected_entities": list(dict.fromkeys(detected_entities)),
            "detected_attributes": list(dict.fromkeys(detected_attributes)),
            "detected_dates": list(dict.fromkeys(detected_dates)),
            "detected_numbers": list(dict.fromkeys(detected_numbers)),
            "detected_qualifiers": list(dict.fromkeys(detected_qualifiers)),
            "query_route": str(state.get("query_route") or f"route_{sel_route.lower()}"),
            "query_scope": str(state.get("query_scope") or "in_scope"),
            "query_scope_confidence": float(state.get("query_scope_confidence") or 1.0),
            "pipeline_stop": bool(state.get("pipeline_stop")),
            "stop_reason": str(state.get("stop_reason") or ""),
        },
        "retrieval_handoff": {
            "recipient_node": exact_handoff_payload["recipient_node"],
            "retrieval_queries": retrieval_queries_list,
            "retrieval_weights": list(state.get("retrieval_weights") or [1.0]),
            "search_queries_with_intents": search_queries_with_intents,
            "channels_passed": list(state.get("retrieval_channels") or ["dense", "bm25", "lexical"]),
            "vector_collection": tenant_id,
            "top_k_per_branch": 25,
            "exact_payload": exact_handoff_payload,
            "contract_validation_status": "VALID" if retrieval_queries_list else "WARNING: Empty retrieval queries",
        },
    }

    panel_2_dense_retrieval = {
        "candidate_count": len(dense_records),
        "candidates": dense_records[:25],
    }

    panel_3_bm25_retrieval = {
        "candidate_count": len(bm25_records),
        "candidates": bm25_records[:25],
    }

    panel_4_hybrid_fusion = {
        "total_fused_count": len(fused_candidates),
        "top_15_candidates": candidates_top15,
        "is_multi_intent": bool(state.get("is_multi_intent")),
        "intent_results": list(state.get("intent_results") or []),
    }

    panel_5_ranking_verification = {
        "verified_count": len(verified_candidates),
        "uncertain_count": len(uncertain_candidates),
        "rejected_count": len(rejected_candidates),
        "ranked_count": len(ranked_candidates),
        "decisions": [
            {
                "candidate_id": c["document_id"],
                "source": c["source"],
                "initial_rank": c["initial_rank"],
                "final_rank": c["final_rank"],
                "verification_status": c["verification_status"],
                "reasons": c["verification_reasons"],
                "stage": "verification" if c["verification_status"] in ("rejected", "uncertain") else ("ranking" if c["final_rank"] else "unranked"),
            }
            for c in candidates_top15
        ],
    }

    panel_6_packaged_context = {
        "context_units_count": len(packaged_items_list),
        "estimated_tokens": estimated_tokens,
        "token_budget": token_budget,
        "token_budget_percentage": token_pct,
        "packaged_units": packaged_items_list,
        "raw_context_text": final_context,
        "prompt_messages": list(state.get("prompt_messages") or []),
    }

    panel_7_answer_generation = {
        "answer": answer,
        "supporting_citations": supporting_citations,
        "grounding_status": str(state.get("answer_grounding_status") or "not_run"),
        "guard_status": str(state.get("answer_guard_status") or "ok"),
        "guard_reason": str(state.get("answer_guard_reason") or ""),
        "answer_mode": str(state.get("answer_mode") or "model"),
        "model_calls": int(state.get("model_calls") or 0),
    }

    inclusive_timings = {
        "resolve_conversation": round(float(stage_timings.get("resolve_conversation", 0.0)), 5),
        "understand_query": round(float(stage_timings.get("understand_query", 0.0)), 5),
        "dense_retrieve": round(float(stage_timings.get("dense_retrieve", 0.0)), 5),
        "bm25_retrieve": round(float(stage_timings.get("bm25_retrieve", 0.0)), 5),
        "lexical_retrieve": round(float(stage_timings.get("lexical_retrieve", 0.0)), 5),
        "fusion": round(float(stage_timings.get("fusion", 0.0)), 5),
        "verify_and_rank": round(float(stage_timings.get("verify_and_rank", 0.0)), 5),
        "context_packaging": round(float(stage_timings.get("context_packaging", 0.0)), 5),
        "llm_generation": round(float(stage_timings.get("llm_generation", 0.0)), 5),
        "post_processing": round(float(stage_timings.get("post_processing", 0.0)), 5),
        "total_pipeline": round(float(stage_timings.get("total_pipeline", stage_timings.get("total_turn", 0.0))), 5),
    }
    exclusive_timings = {
        "resolve_conversation": round(float(stage_timings.get("resolve_conversation", 0.0)), 5),
        "understand_query": round(float(stage_timings.get("understand_query", 0.0)), 5),
        "query_embedding": round(float(stage_timings.get("query_embedding", 0.0)), 5),
        "chroma_vector_search": round(float(stage_timings.get("chroma_vector_search", 0.0)), 5),
        "bm25_retrieve": round(float(stage_timings.get("bm25_retrieve", 0.0)), 5),
        "lexical_retrieve": round(float(stage_timings.get("lexical_retrieve", 0.0)), 5),
        "fusion": round(float(stage_timings.get("fusion", 0.0)), 5),
        "verify_and_rank": round(float(stage_timings.get("verify_and_rank", 0.0)), 5),
        "context_packaging": round(float(stage_timings.get("context_packaging", 0.0)), 5),
        "llm_generation": round(float(stage_timings.get("llm_generation", 0.0)), 5),
        "post_processing": round(float(stage_timings.get("post_processing", 0.0)), 5),
    }

    panel_8_stage_timings = {
        "timings": stage_timings,
        "inclusive_timings": inclusive_timings,
        "exclusive_timings": exclusive_timings,
        "warnings": list(state.get("warnings") or []),
        "errors": list(state.get("errors") or []),
    }

    return {
        "resolved_question": str(state.get("resolved_question") or ""),
        "query_plan": state.get("query_plan"),
        "raw_state": dict(state),
        "funnel": {
            "retrieved_total": len(fused_candidates),
            "top_window_count": len(candidates_top15),
            "verified_count": len(verified_candidates),
            "uncertain_count": len(uncertain_candidates),
            "rejected_count": len(rejected_candidates),
            "ranked_count": len(ranked_candidates),
            "context_units_count": len(packaged_items_list),
            "displayed_citations_count": len(supporting_citations),
        },
        "section_a_candidates": candidates_top15,
        "section_b_context": {
            "context_units_count": len(packaged_items_list),
            "context_characters": len(final_context),
            "estimated_tokens": estimated_tokens,
            "token_budget": token_budget,
            "token_budget_percentage": token_pct,
            "claim_audit_status": claim_audit_status,
            "supported_claims": supported_claim_ids,
            "partial_claims": partial_claim_ids,
            "omitted_claims": omitted_claim_ids,
            "packaged_units": packaged_items_list,
        },
        "section_c_citations": supporting_citations,
        # Phase 1 Eight-Panel Payload
        "panel_query_understanding": panel_1_query_understanding,
        "panel_dense_retrieval": panel_2_dense_retrieval,
        "panel_bm25_retrieval": panel_3_bm25_retrieval,
        "panel_hybrid_fusion": panel_4_hybrid_fusion,
        "panel_ranking_verification": panel_5_ranking_verification,
        "panel_packaged_context": panel_6_packaged_context,
        "panel_answer_generation": panel_7_answer_generation,
        "panel_stage_timings": panel_8_stage_timings,
    }


__all__ = [
    "DEFAULT_DIAGNOSTIC_WINDOW",
    "CONFIGURED_TOKEN_BUDGET",
    "build_retrieval_diagnostics",
]
