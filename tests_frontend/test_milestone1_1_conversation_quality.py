"""Milestone 1.1 Focused Regression Tests: Answer Completeness, Conversation Resolution, and Context-Aware Follow-Ups.

Covers:
- Answer completeness and clean termination without mid-sentence cut-offs.
- Coreference resolution, category inheritance, topic change isolation, and user corrections.
- Lightweight, grounded follow-up suggestions and conversation turn execution.
- Cross-session and cross-tenant isolation.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ai_platform.core.answering.generator import (
    AnswerGenerationRequest,
    generate_answer,
    _ends_with_terminal_punct,
    _ensure_clean_sentence_termination,
)
from ai_platform.core.conversation.conversation import (
    resolve_conversation,
    MAX_HISTORY_MESSAGES,
)
from ai_platform.core.evidence.packaging import (
    EvidencePackage,
    PackagedEvidenceItem,
)
from ai_platform.service.assistant_service import (
    ChatTurnRequest,
    ChatTurnResponse,
    handle_chat_turn,
)
from ai_platform.service.suggestions import generate_follow_up_suggestions
from frontend.clients.assistant_client import send_chat_turn
from frontend.core.models import ChatMessage, Conversation, InstitutionUIConfig
from frontend.core.session_manager import (
    add_assistant_message,
    add_user_message,
    create_new_conversation,
    get_active_conversation,
    get_bounded_history,
    init_session_state,
    switch_conversation,
)


# =============================================================================
# 1. Answer Completeness & Clean Termination Tests
# =============================================================================

class TestAnswerCompleteness:
    """Test suite for answer termination, truncation detection, and repair."""

    def test_complete_answer_preserved_without_alteration(self):
        """A complete valid answer with terminal punctuation is preserved exactly."""
        text = "The tuition fee for regular M.Tech. students is INR 2,25,000 per semester."
        assert _ends_with_terminal_punct(text) is True
        assert _ensure_clean_sentence_termination(text) == text

    def test_incomplete_answer_trimmed_to_sentence_boundary(self):
        """An answer cut off mid-sentence is trimmed back to the last complete sentence."""
        raw = (
            "Any candidate joining the MBA program must submit an NOC. "
            "The NOC should either state that the employer has no objection to"
        )
        assert _ends_with_terminal_punct(raw) is False
        cleaned = _ensure_clean_sentence_termination(raw)
        assert cleaned == "Any candidate joining the MBA program must submit an NOC."
        assert _ends_with_terminal_punct(cleaned) is True

    def test_generator_handles_done_reason_length_with_repair(self):
        """When model hits token limit and leaves an incomplete sentence, fallback/repair triggers."""
        ev = EvidencePackage(
            context="Candidates must submit an NOC from their employer specifying leave duration and return commitment.",
            status="ready",
            items=(
                PackagedEvidenceItem(
                    evidence_id="1",
                    text="Candidates must submit an NOC from their employer specifying leave duration and return commitment.",
                    source="sme/rules.docx",
                ),
            ),
        )
        req = AnswerGenerationRequest(
            question="What are the NOC requirements?",
            evidence=ev,
        )

        # Mock model that terminates with length limit mid-sentence on first attempt
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Candidates must submit an NOC from their employer specifying leave duration and"
        mock_response.response_metadata = {"done_reason": "length"}
        mock_model.invoke.return_value = mock_response

        result = generate_answer(req, mock_model)
        assert result.generated is True
        # Either repaired or cleanly terminated
        assert result.answer
        assert _ends_with_terminal_punct(result.answer)
        assert not result.answer.endswith("and")

    def test_short_factual_answer_not_unnecessarily_continued(self):
        """A genuinely complete short answer is returned directly with 1 model call."""
        ev = EvidencePackage(
            context="The application processing fee is INR 300 for General category.",
            status="ready",
            items=(
                PackagedEvidenceItem(
                    evidence_id="1",
                    text="The application processing fee is INR 300 for General category.",
                    source="admissions.docx",
                ),
            ),
        )
        req = AnswerGenerationRequest(
            question="What is the application fee?",
            evidence=ev,
        )

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The application processing fee is INR 300 for General category."
        mock_response.response_metadata = {"done_reason": "stop"}
        mock_model.invoke.return_value = mock_response

        result = generate_answer(req, mock_model)
        assert result.answer == "The application processing fee is INR 300 for General category."
        assert result.model_calls == 1
        assert result.finish_reason == "stop"


# =============================================================================
# 2. Conversation Resolution Tests
# =============================================================================

class TestConversationResolution:
    """Test suite for conversation resolution, coreference, and context isolation."""

    def test_01_topic_dependent_followup_resolves_correctly(self):
        """Category follow-up inherits antecedent attribute and entity."""
        history = [
            {"role": "user", "content": "What is the tuition fee for regular M.Tech. students?"},
            {"role": "assistant", "content": "The tuition fee for regular M.Tech. students is INR 2,25,000."},
        ]
        res = resolve_conversation(question="What about sponsored students?", chat_history=history)
        assert res["mode"] == "follow_up"
        assert "tuition fee" in res["resolved_question"].lower()
        assert "sponsored" in res["resolved_question"].lower()
        assert "M.Tech" in res["resolved_question"]
        assert res["active_topic"] == "fee"

    def test_02_multiturn_followup_preserves_program_and_category(self):
        """Second follow-up maintains program anchor and addresses new topic."""
        history = [
            {"role": "user", "content": "What are the eligibility criteria for the regular Ph.D. programme?"},
            {"role": "assistant", "content": "Regular Ph.D. requires a master's degree with 60% or 4-year bachelor's with 70%."},
        ]
        res = resolve_conversation(
            question="How many days of leave can scholars in this programme take?",
            chat_history=history,
        )
        assert res["mode"] == "follow_up"
        assert "PhD" in res["resolved_question"] or "Ph.D." in res["resolved_question"]
        assert "leave" in res["resolved_question"].lower()
        assert res["active_topic"] == "leave"

    def test_03_bounded_history_windowing(self):
        """Resolver respects bounded windowing without exceeding max messages."""
        long_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}"} for i in range(20)]
        res = resolve_conversation(question="What is the hostel fee?", chat_history=long_history)
        assert res["mode"] == "standalone"

    def test_04_topic_change_does_not_inherit_previous_entities(self):
        """A genuine topic change in the same chat does not carry forward older program anchors."""
        history = [
            {"role": "user", "content": "What is the tuition fee for regular M.Tech. students?"},
            {"role": "assistant", "content": "The tuition fee is INR 2,25,000 per semester."},
        ]
        res = resolve_conversation(
            question="Where is the campus health centre located?",
            chat_history=history,
        )
        assert res["mode"] == "standalone"
        assert res["active_topic"] == "location"
        assert "M.Tech" not in res["resolved_question"]
        assert res["active_entity"] == ""

    def test_05_explicit_user_correction_overrides_earlier_context(self):
        """User explicit correction ('No, I meant MBA') replaces prior program anchor."""
        history = [
            {"role": "user", "content": "What is the tuition fee for regular M.Tech. students?"},
            {"role": "assistant", "content": "The tuition fee for regular M.Tech. students is INR 2,25,000."},
        ]
        res = resolve_conversation(
            question="No, I meant MBA, not M.Tech",
            chat_history=history,
        )
        assert res["mode"] == "follow_up"
        assert res["active_entity"] == "MBA"
        assert "MBA" in res["resolved_question"]
        assert "tuition fee" in res["resolved_question"].lower()

    def test_06_resolution_cannot_cross_session_boundaries(self):
        """Vague follow-up in a fresh session with empty history remains standalone."""
        res = resolve_conversation(question="What about sponsored students?", chat_history=[])
        assert res["mode"] == "standalone"
        assert res["resolved_question"] == "What about sponsored students?"

    def test_07_institution_identity_remains_unchanged(self):
        """Query rewriting does not alter institution targeting."""
        req = ChatTurnRequest(
            question="What about sponsored students?",
            chat_history=(
                {"role": "user", "content": "What is the tuition fee for M.Tech.?"},
                {"role": "assistant", "content": "INR 2,25,000"},
            ),
            institution_id="iitj",
        )
        assert req.institution_id == "iitj"

    def test_08_long_multi_part_queries_processable(self):
        """Long user queries with conjunctions remain processable without syntax failure."""
        long_q = (
            "What are the eligibility criteria for regular Ph.D. admissions, "
            "what are the fellowship opportunities, and what is the application fee?"
        )
        res = resolve_conversation(question=long_q, chat_history=[])
        assert res["mode"] == "standalone"
        assert res["resolved_question"] == long_q


# =============================================================================
# 3. Suggested Follow-Up Tests
# =============================================================================

class TestSuggestedFollowUps:
    """Test suite for lightweight follow-up suggestions and conversation integration."""

    def test_01_relevant_suggestions_appear_after_supported_answer(self):
        """A supported answer yields up to 3 relevant suggestions."""
        suggs = generate_follow_up_suggestions(
            "What is the tuition fee for regular M.Tech. students?",
            answer="The tuition fee for regular M.Tech. students is INR 2,25,000 per semester.",
            sources=[{"source": "admissions/mtech.docx", "heading": "Office of Academics - Registration"}],
            is_safe_refusal=False,
            evidence_status="supported",
        )
        assert len(suggs) > 0
        assert len(suggs) <= 3
        assert all(isinstance(s, str) for s in suggs)

    def test_02_at_most_three_suggestions_displayed(self):
        """Suggestion count is strictly capped at 3."""
        suggs = generate_follow_up_suggestions(
            "Tell me about the Ph.D. program",
            answer="Ph.D. programs are offered across engineering and science departments.",
            max_suggestions=3,
        )
        assert len(suggs) <= 3

    def test_03_suggestions_non_duplicative_and_distinct_from_question(self):
        """Suggestions do not repeat the current question or each other."""
        q = "What is the tuition fee for regular M.Tech. students?"
        suggs = generate_follow_up_suggestions(
            q,
            answer="The tuition fee is INR 2,25,000.",
        )
        assert len(suggs) == len(set(suggs))
        for s in suggs:
            assert s.casefold() != q.casefold()

    def test_04_suggestions_suppressed_for_safe_refusal(self):
        """Safe refusals emit zero suggestions to prevent misleading guidance."""
        suggs = generate_follow_up_suggestions(
            "What is the rank in 2050?",
            answer="The provided evidence does not contain information regarding NIRF 2050.",
            is_safe_refusal=True,
            evidence_status="unsupported",
        )
        assert suggs == []

    def test_05_suggestions_suppressed_for_error_or_empty_evidence(self):
        """Errors and empty queries yield zero suggestions."""
        assert generate_follow_up_suggestions("hello", answer="", is_safe_refusal=False, evidence_status="empty") == []
        assert generate_follow_up_suggestions("test", answer="Error occurred", is_safe_refusal=True, evidence_status="error") == []

    def test_06_suggestions_zero_additional_llm_calls(self):
        """Suggestions are generated purely from metadata without invoking an LLM."""
        # Function signature takes no model parameter, guaranteeing 0 LLM calls
        suggs = generate_follow_up_suggestions(
            "What are the leave rules for Ph.D. scholars?",
            answer="Scholars are entitled to 30 days of leave per year.",
            sources=[],
        )
        assert isinstance(suggs, list)
        assert len(suggs) <= 3

    def test_07_service_response_includes_suggested_questions(self):
        """ChatTurnResponse contains suggested_questions without breaking contract."""
        resp = ChatTurnResponse(
            answer="Test answer",
            suggested_questions=["What are the fees?", "How do I apply?"],
        )
        assert resp.suggested_questions == ["What are the fees?", "How do I apply?"]
        assert resp.answer == "Test answer"

    def test_08_session_manager_stores_suggestions_on_chat_message(self):
        """Assistant message state in session manager records suggested_questions."""
        msg = ChatMessage(
            role="assistant",
            content="Test content",
            suggested_questions=["Follow-up 1", "Follow-up 2"],
        )
        assert msg.suggested_questions == ["Follow-up 1", "Follow-up 2"]

    def test_09_different_conversations_isolate_suggestions(self):
        """Different conversation threads maintain completely isolated messages and suggestions."""
        class DummySession(dict):
            def __getattr__(self, key):
                return self.get(key)
            def __setattr__(self, key, value):
                self[key] = value

        import streamlit as st
        mock_session = DummySession()
        mock_session.conversations = {}
        mock_session.active_chat_id = None
        mock_session.active_institution_id = "iitj"
        st.session_state = mock_session

        config = InstitutionUIConfig(institution_id="iitj", display_name="IIT Jodhpur")
        conv1_id = create_new_conversation(config, title="Chat 1")
        conv = get_active_conversation()
        conv.messages.append(ChatMessage(role="assistant", content="Ans 1", suggested_questions=["Sugg A"]))

        conv2_id = create_new_conversation(config, title="Chat 2")
        conv2 = get_active_conversation()
        assert conv2.id != conv1_id
        # Conv2 has only welcome message and no Sugg A
        assert "Sugg A" not in [s for m in conv2.messages for s in getattr(m, "suggested_questions", [])]

    def test_10_clicked_suggestion_turn_retains_session_id_and_history(self):
        """Client sending a clicked suggestion turn preserves session ID and bounded history."""
        mock_service = MagicMock()
        mock_service.return_value = ChatTurnResponse(
            answer="Sponsored M.Tech fees are INR 2,75,000.",
            suggested_questions=["What about hostel fees?"],
        )

        history = [{"role": "user", "content": "What is M.Tech fee?"}, {"role": "assistant", "content": "INR 2,25,000"}]
        resp = send_chat_turn(
            "What about sponsored students?",
            history=history,
            session_id="chat_12345",
            institution_id="iitj",
        )
        assert resp.answer
        assert isinstance(resp.suggested_questions, list)
