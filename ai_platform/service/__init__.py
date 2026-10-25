"""Service layer providing stable boundaries for clients."""

from .assistant_service import (
    ChatTurnRequest,
    ChatTurnResponse,
    handle_chat_turn,
    render_structured_blocks,
    is_navigation_intent,
    needs_rag_answering,
)

__all__ = [
    "ChatTurnRequest",
    "ChatTurnResponse",
    "handle_chat_turn",
    "render_structured_blocks",
    "is_navigation_intent",
    "needs_rag_answering",
]
