"""Message Rendering Component.

Renders conversation turns using Streamlit chat message containers, integrating
markdown answers, structured navigation/emergency blocks, citation drawers, and
response latency badges.
"""

from __future__ import annotations

from typing import Sequence
import streamlit as st

from frontend.components.diagnostics import render_retrieval_diagnostics_panel
from frontend.components.sources import render_sources_accordion
from frontend.components.structured_cards import render_structured_card
from frontend.core.models import ChatMessage


def render_chat_messages(messages: Sequence[ChatMessage]) -> str | None:
    """Render all messages in the active conversation thread and return any clicked suggestion."""
    clicked_suggestion: str | None = None
    num_messages = len(messages)

    for idx, message in enumerate(messages):
        is_last = idx == num_messages - 1
        role = message.role or "assistant"

        with st.chat_message(role):
            # 1. Render structured cards (e.g. navigation or emergency)
            if message.structured_blocks:
                for block in message.structured_blocks:
                    render_structured_card(block)

            # 2. Render primary markdown response text
            if message.content:
                st.markdown(message.content)

            # 3. Render verified sources/citations accordion if present
            if message.sources:
                render_sources_accordion(message.sources)

            # 4. Render context-aware suggested follow-up buttons on latest assistant turn
            if role == "assistant" and getattr(message, "suggested_questions", None) and is_last:
                st.markdown(
                    "<div style='margin-top: 10px; margin-bottom: 6px; font-size: 0.85rem; color: #4B5563; font-weight: 500;'>"
                    "💡 Suggested follow-ups:"
                    "</div>",
                    unsafe_allow_html=True,
                )
                def _handle_followup_click(prompt_text: str) -> None:
                    st.session_state["pending_prompt"] = prompt_text
                    st.session_state["_turn_in_progress"] = False

                # Consume any prompt queued via on_click callback
                queued_prompt = st.session_state.pop("pending_prompt", None)
                if queued_prompt:
                    clicked_suggestion = queued_prompt

                sugg_cols = st.columns(len(message.suggested_questions))
                for s_idx, sugg in enumerate(message.suggested_questions):
                    clicked_s = sugg_cols[s_idx].button(
                        f"💬 {sugg}",
                        key=f"sugg_btn_{idx}_{s_idx}",
                        use_container_width=True,
                        on_click=_handle_followup_click,
                        args=(sugg,),
                    )
                    if clicked_s and not clicked_suggestion:
                        clicked_suggestion = sugg

                # Unconditionally clear pending_prompt so no subsequent rerun can re-consume it
                st.session_state.pop("pending_prompt", None)


            # 5. Render latency caption & status metadata
            if role == "assistant":
                meta_parts = []
                if message.response_time is not None and message.response_time > 0:
                    meta_parts.append(f"⏱️ {message.response_time:.2f}s")

                if message.is_safe_refusal:
                    meta_parts.append("🛡️ Safe Abstention")

                if meta_parts:
                    st.caption(" • ".join(meta_parts))

    return clicked_suggestion


__all__ = ["render_chat_messages"]
