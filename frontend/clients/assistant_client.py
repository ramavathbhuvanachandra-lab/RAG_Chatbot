"""Assistant Service Client Adapter.

Provides a clean presentation-layer boundary between the Streamlit frontend
and the canonical AI platform assistant service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from ai_platform.service.assistant_service import (
    ChatTurnRequest,
    ChatTurnResponse,
    handle_chat_turn,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssistantResponse:
    """Client-facing representation of the assistant's turn response."""

    answer: str
    structured_blocks: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    is_safe_refusal: bool = False
    evidence_status: str = "supported"
    formatted_response: str = ""
    diagnostics: dict[str, Any] | None = None
    suggested_questions: list[str] = field(default_factory=list)


def send_chat_turn(
    question: str,
    *,
    history: Sequence[dict[str, Any]] | None = None,
    session_id: str | None = None,
    institution_id: str | None = None,
) -> AssistantResponse:
    """Send a user question to the canonical assistant service.

    Parameters:
        question: The user query string.
        history: Sequence of previous message dictionaries ({'role': ..., 'content': ...}).
        session_id: Active conversational session ID.
        institution_id: Target tenant institution ID.

    Returns:
        `AssistantResponse` with the generated answer, citations, and metadata.
    """
    clean_q = " ".join(str(question or "").strip().split())
    if not clean_q:
        return AssistantResponse(
            answer="Please enter a valid question.",
            formatted_response="Please enter a valid question.",
            is_safe_refusal=True,
            evidence_status="empty_query",
        )

    turn_req = ChatTurnRequest(
        question=clean_q,
        chat_history=tuple(history or ()),
        session_id=session_id,
        institution_id=institution_id,
    )

    try:
        service_resp: ChatTurnResponse = handle_chat_turn(turn_req)
        return AssistantResponse(
            answer=service_resp.answer,
            structured_blocks=service_resp.structured_content,
            sources=service_resp.sources,
            latency_seconds=service_resp.latency_seconds,
            is_safe_refusal=service_resp.is_safe_refusal,
            evidence_status=service_resp.evidence_status,
            formatted_response=service_resp.formatted_response,
            diagnostics=service_resp.diagnostics,
            suggested_questions=getattr(service_resp, "suggested_questions", []),
        )
    except Exception as exc:
        logger.exception("Error executing chat turn for institution %s: %s", institution_id, exc)
        safe_msg = (
            "⚠️ An error occurred while retrieving information for this question. "
            "Please try rephrasing your query or try again shortly."
        )
        err_diag = {
            "terminal_status": "error",
            "latency_seconds": 0.0,
            "resolved_question": clean_q,
            "errors": [str(exc)],
            "panel_stage_timings": {
                "timings": {"total_turn": 0.0, "total_pipeline": 0.0},
                "exclusive_timings": {},
                "errors": [str(exc)],
            },
        }
        return AssistantResponse(
            answer=safe_msg,
            formatted_response=safe_msg,
            latency_seconds=0.0,
            is_safe_refusal=True,
            evidence_status="service_error",
            diagnostics=err_diag,
        )


__all__ = [
    "AssistantResponse",
    "send_chat_turn",
]
