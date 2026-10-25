"""End-to-end standalone execution entry point.

This is the runtime composition root: UI or external callers can invoke
``ask`` without importing any legacy application module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from ai_platform.core.graph.graph import build_graph


@lru_cache(maxsize=1)
def get_app_graph():
    """Create and cache the standalone production graph."""
    return build_graph()


def ask(
    question: str,
    *,
    chat_history: Sequence[Any] | None = None,
    institution_profile: Any | None = None,
) -> str:
    """Run one question through the complete reusable RAG pipeline."""
    clean = " ".join(str(question or "").strip().split())
    if not clean:
        raise ValueError("question cannot be empty")

    state: dict[str, Any] = {
        "question": clean,
        "chat_history": tuple(chat_history or ()),
    }
    if institution_profile is not None:
        from ai_platform.institutions.loader import register_institution_profile
        register_institution_profile(institution_profile)
        state["institution_profile"] = institution_profile
    result = get_app_graph().invoke(state)
    answer = result.get("answer")
    if answer is None:
        raise RuntimeError("RAG graph completed without an answer")
    return str(answer).strip()


def ask_with_diagnostics(
    question: str,
    *,
    chat_history: Sequence[Any] | None = None,
    institution_profile: Any | None = None,
) -> dict[str, Any]:
    """Run one question and return the complete graph state for diagnostics."""
    clean = " ".join(str(question or "").strip().split())
    if not clean:
        raise ValueError("question cannot be empty")
    state: dict[str, Any] = {
        "question": clean,
        "chat_history": tuple(chat_history or ()),
    }
    if institution_profile is not None:
        from ai_platform.institutions.loader import register_institution_profile
        register_institution_profile(institution_profile)
        state["institution_profile"] = institution_profile
    return dict(get_app_graph().invoke(state))


@lru_cache(maxsize=1)
def get_inspect_nodes():
    """Create and cache canonical nodes for context inspection."""
    from ai_platform.core.graph.nodes import CoreNodes, default_dependencies
    return CoreNodes(default_dependencies())


def inspect_context(
    question: str,
    *,
    chat_history: Sequence[Any] | None = None,
    institution_profile: Any | None = None,
    nodes: Any | None = None,
) -> dict[str, Any]:
    """Execute retrieval, verification, and context packaging WITHOUT LLM generation.

    Returns structured pre-generation context diagnostics:
    - context string sent to the answer model
    - evidence units, IDs, sources, headings
    - character count and estimated token count
    - fail-closed verification decisions and abstention status
    - exact prompt messages that would be passed to the LLM
    - evidence excluded or rejected at verification and packaging stages
    """
    import re
    from ai_platform.core.answering.generator import build_answer_prompt

    clean = " ".join(str(question or "").strip().split())
    if not clean:
        raise ValueError("question cannot be empty")

    active_nodes = nodes or get_inspect_nodes()
    state: dict[str, Any] = {
        "question": clean,
        "chat_history": tuple(chat_history or ()),
    }
    if institution_profile is not None:
        from ai_platform.institutions.loader import register_institution_profile
        register_institution_profile(institution_profile)
        state["institution_profile"] = institution_profile

    # Stage 1: Conversation resolution
    state.update(active_nodes.resolve_conversation_node(state))

    # Stage 2: Query understanding
    state.update(active_nodes.understand_query_node(state))

    # Check for early stop (out-of-scope / guard)
    if bool(state.get("pipeline_stop")):
        safe_resp = state.get("safe_response")
        stop_reason = state.get("stop_reason") or "out_of_scope"
        return {
            "question": clean,
            "resolved_question": clean,
            "pipeline_stop": True,
            "stop_reason": stop_reason,
            "is_multi_intent": False,
            "intent_questions": (),
            "evidence_status": "insufficient",
            "evidence_coverage_status": "insufficient",
            "ready_for_generation": False,
            "will_abstain": True,
            "abstention_reason": stop_reason,
            "fallback_response": safe_resp or "Sorry, I couldn't find reliable information about that in the available IIT Jodhpur documents.",
            "final_evidence_context": "",
            "context_character_count": 0,
            "estimated_token_count": 0,
            "evidence_units": [],
            "evidence_unit_count": 0,
            "prompt_messages": [],
            "candidate_count": 0,
            "verified_candidate_count": 0,
            "verification_trace": (),
            "unsupported_years": (),
            "rejected_candidate_count": 0,
            "rejected_candidates": (),
            "package_reasons": (),
            "omitted_units_count": 0,
            "omitted_claim_ids": (),
            "supported_claim_ids": (),
            "partial_claim_ids": (),
            "claim_audit_status": "",
            "raw_retrieval_count": 0,
        }

    # Stage 3: Multi-intent planning
    state.update(active_nodes.plan_multi_intent_node(state))
    is_multi = bool(state.get("is_multi_intent"))

    # Stage 4-7: Retrieval, verification, ranking, packaging
    if is_multi:
        state.update(active_nodes.process_multi_intent_node(state))
    else:
        state.update(active_nodes.retrieve_and_qualify_node(state))

    package = state.get("evidence_package")
    package_ready = bool(package is not None and getattr(package, "ready_for_generation", False))
    if not package_ready:
        evidence_status = "insufficient"
    else:
        evidence_status = str(state.get("evidence_status") or "").strip().casefold() or "supported"
    coverage_status = str(state.get("evidence_coverage_status") or "").strip() or (
        "available" if package is not None else "insufficient"
    )

    institution = active_nodes.deps.institution_provider(state)
    fallback = active_nodes.deps.fallback_provider(state, institution)

    # Fail-closed checks identical to answer_node
    raw_q = str(state.get("resolved_question") or state.get("question") or "").strip()
    requested_years = re.findall(r"\b(20\d\d)\b", raw_q)
    evidence_context = str(getattr(package, "context", "") or "").strip()
    unsupported_years = [y for y in requested_years if y not in evidence_context]

    fail_closed = bool(
        not state.get("evidence_candidates")
        or package is None
        or not bool(getattr(package, "ready_for_generation", False))
        or not evidence_context
        or bool(requested_years and unsupported_years)
    )

    abstention_reason = None
    fallback_response = None
    if fail_closed:
        if requested_years and unsupported_years:
            abstention_reason = "unsupported_temporal_year"
            fallback_response = f"The provided evidence does not contain information regarding the requested year ({', '.join(unsupported_years)})."
        else:
            abstention_reason = "insufficient_evidence"
            fallback_response = fallback

    # Formulate question string
    q_for_prompt = raw_q
    if is_multi:
        intent_questions = tuple(state.get("intent_questions") or ())
        q_parts = [
            "Answer each numbered user request explicitly in turn (1., 2., etc.). ",
            "Do not merge evidence between requests. ",
            "Use only the verified evidence supplied. ",
            "When a numbered request lacks sufficient support in the evidence, state explicitly for that request: ",
            "'Regarding [topic], the provided evidence does not contain information.' ",
            "Never omit or skip any numbered request.\n\n",
        ]
        q_parts.append("\n".join(
            f"{idx}. {item.strip()}"
            for idx, item in enumerate(intent_questions, start=1)
            if str(item).strip()
        ))
        q_for_prompt = "".join(q_parts)

    # Format evidence units
    unit_list: list[dict[str, Any]] = []
    if package is not None and hasattr(package, "items"):
        for item in package.items:
            unit_list.append({
                "evidence_id": getattr(item, "evidence_id", ""),
                "source": getattr(item, "source", ""),
                "heading": getattr(item, "heading", ""),
                "text": getattr(item, "text", ""),
            })

    # Exact prompt if ready for generation
    prompt_messages: list[dict[str, str]] = []
    if not fail_closed and package is not None:
        try:
            req = active_nodes.deps.answer_request_type(
                question=q_for_prompt,
                evidence=package,
                institution=institution,
                chat_history=tuple(state.get("chat_history") or ()),
                question_type=str(state.get("evidence_question_type") or "").strip() or None,
                evidence_coverage=coverage_status,
                required_entities=tuple(
                    (str(getattr(e, "name", "")).strip(), str(getattr(e, "entity_type", "")).strip())
                    for e in (getattr(state.get("query"), "entities", ()) or ())
                    if str(getattr(e, "name", "")).strip()
                ),
                known_program_names=tuple(state.get("known_program_aliases", ()) or ()),
            )
            prompt_messages = build_answer_prompt(req)
        except Exception:
            pass

    return {
        "question": clean,
        "resolved_question": raw_q,
        "pipeline_stop": False,
        "stop_reason": None,
        "is_multi_intent": is_multi,
        "intent_questions": tuple(state.get("intent_questions") or ()),
        "evidence_status": evidence_status,
        "evidence_coverage_status": coverage_status,
        "ready_for_generation": not fail_closed,
        "will_abstain": fail_closed,
        "abstention_reason": abstention_reason,
        "fallback_response": fallback_response,
        "final_evidence_context": evidence_context if not fail_closed else "",
        "context_character_count": len(evidence_context) if not fail_closed else 0,
        "estimated_token_count": len(evidence_context) // 4 if not fail_closed else 0,
        "evidence_units": unit_list,
        "evidence_unit_count": len(unit_list),
        "prompt_messages": prompt_messages,
        "candidate_count": len(tuple(state.get("ranked_candidates") or ())),
        "verified_candidate_count": len(tuple(state.get("verified_candidates") or ())),
        "verification_trace": tuple(state.get("verification_trace") or ()),
        "unsupported_years": tuple(unsupported_years),
        "rejected_candidate_count": len(tuple(state.get("rejected_candidates") or ())),
        "rejected_candidates": tuple(
            {
                "document_id": getattr(c, "document_id", ""),
                "source": getattr(c, "source", None) or getattr(getattr(c, "document", None), "source", ""),
                "score": float(getattr(c, "score", 0.0) or 0.0),
            }
            for c in (state.get("rejected_candidates") or ())
        ),
        "package_reasons": tuple(getattr(package, "reasons", ()) or ()),
        "omitted_units_count": int(getattr(package, "omitted_count", 0) or 0),
        "omitted_claim_ids": tuple(getattr(package, "omitted_claim_ids", ()) or ()),
        "supported_claim_ids": tuple(getattr(package, "supported_claim_ids", ()) or ()),
        "partial_claim_ids": tuple(getattr(package, "partial_claim_ids", ()) or ()),
        "claim_audit_status": str(getattr(state.get("claim_audit"), "status", "") or ""),
        "raw_retrieval_count": sum(len(r) for r in state.get("retrieval_results", ())),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the standalone AI platform RAG engine.")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--inspect", action="store_true", help="Inspect pre-generation context without LLM call")
    args = parser.parse_args()
    q = " ".join(args.question)
    if args.inspect:
        diag = inspect_context(q)
        print(json.dumps({k: v for k, v in diag.items() if k != "final_evidence_context"}, indent=2))
        print("\n--- PRE-GENERATION CONTEXT ---")
        print(diag["final_evidence_context"][:1000] + ("..." if len(diag["final_evidence_context"]) > 1000 else ""))
    else:
        print(ask(q))


__all__ = ["get_app_graph", "get_inspect_nodes", "ask", "ask_with_diagnostics", "inspect_context"]
