"""Focused tests for the Canonical Assistant Service and UI Integration.

Verifies:
1. Normal RAG question returns a valid ChatTurnResponse with answer and citations.
2. IIT Jodhpur navigation request resolves through the registered structured handler.
3. IIT Jodhpur emergency request resolves through the registered structured handler.
4. An institution without structured handlers falls back smoothly to normal RAG.
5. Changing the institution does not leak structured handlers across tenants.
6. A failing or exception-raising structured handler falls back safely to RAG.
7. An engine or generation error is captured safely without crashing.
8. Persistence failure behavior remains non-blocking for chat turns.
9. UI.chat.get_response remains fully compatible with existing UI entry points.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_platform.core.institution.profile import InstitutionProfile
from ai_platform.institutions.loader import (
    register_institution_profile,
    unregister_institution_profile,
    register_institution_handler,
    unregister_institution_handler,
    get_institution_handler,
)
from ai_platform.service.assistant_service import (
    ChatTurnRequest,
    ChatTurnResponse,
    handle_chat_turn,
    render_structured_blocks,
    is_navigation_intent,
    needs_rag_answering,
)


class TestAssistantService(unittest.TestCase):
    """Test suite validating canonical assistant service operations."""

    def test_01_normal_rag_question_contract(self):
        """Verify normal RAG question invokes canonical engine and returns ChatTurnResponse."""
        request = ChatTurnRequest(
            question="What is the passing grade point in the grading system at IIT Jodhpur?",
            chat_history=(),
            institution_id="iitj",
        )
        response = handle_chat_turn(request)

        self.assertIsInstance(response, ChatTurnResponse)
        self.assertTrue(len(response.answer) > 0, "Answer must not be empty")
        self.assertTrue(len(response.formatted_response) > 0, "Formatted response must not be empty")
        self.assertGreater(response.latency_seconds, 0.0)
        self.assertIn(response.evidence_status, {"supported", "partial"})
        self.assertFalse(response.is_safe_refusal)
        self.assertIsInstance(response.sources, list)
        self.assertGreater(len(response.sources), 0, "Must return evidence sources")

    def test_02_iitj_navigation_request_uses_registered_handler(self):
        """Verify IIT Jodhpur navigation request resolves through registered structured handler."""
        request = ChatTurnRequest(
            question="Where is the Lecture Hall Complex?",
            institution_id="iitj",
        )
        response = handle_chat_turn(request)

        self.assertIsInstance(response, ChatTurnResponse)
        self.assertGreaterEqual(len(response.structured_content), 1)
        self.assertEqual(response.structured_content[0]["type"], "navigation")

        nav_data = response.structured_content[0]["data"]
        self.assertIn("Lecture Hall", nav_data["name"])
        self.assertIn("📍", response.formatted_response)
        self.assertIn("🗺️ **Google Maps:**", response.formatted_response)


    def test_03_iitj_emergency_request_returns_structured_response(self):
        """Verify IIT Jodhpur emergency request returns structured emergency block."""
        request = ChatTurnRequest(
            question="emergency ambulance security contact",
            institution_id="iitj",
        )
        response = handle_chat_turn(request)

        self.assertIsInstance(response, ChatTurnResponse)
        self.assertGreaterEqual(len(response.structured_content), 1)
        self.assertTrue(
            any(item["type"] == "emergency" for item in response.structured_content),
            "Must contain structured emergency item",
        )
        self.assertIn("📍", response.formatted_response)

    def test_04_institution_without_structured_handler_uses_normal_rag(self):
        """Verify an institution without structured handlers continues smoothly to RAG."""
        synthetic_profile = InstitutionProfile(
            institution_id="synthetic_no_handler",
            display_name="Synthetic College",
            data_path=Path(__file__).resolve().parent,
            vectorstore_path=Path(__file__).resolve().parent,
        )
        register_institution_profile(synthetic_profile)

        try:
            handler = get_institution_handler("synthetic_no_handler", "navigation")
            self.assertIsNone(handler, "Synthetic institution must not have navigation handler")

            request = ChatTurnRequest(
                question="Where is the quantum computing lab?",
                institution_id="synthetic_no_handler",
            )
            # Mock ask_with_diagnostics to isolate RAG execution from missing synthetic documents
            with patch("ai_platform.service.assistant_service.ask_with_diagnostics") as mock_diag:
                mock_diag.return_value = {
                    "answer": "The quantum computing lab is in Euler Hall.",
                    "evidence_package": None,
                    "evidence_status": "supported",
                    "pipeline_stop": False,
                }
                response = handle_chat_turn(request)

                self.assertEqual(len(response.structured_content), 0)
                self.assertIn("Euler Hall", response.answer)
                self.assertFalse(response.is_safe_refusal)
        finally:
            unregister_institution_profile("synthetic_no_handler")

    def test_05_institution_isolation_no_handler_leakage(self):
        """Verify registered structured handlers are strictly tenant-isolated."""
        mock_handler_a = MagicMock(return_value={"name": "Tenant A Lab"})
        mock_handler_b = MagicMock(return_value={"name": "Tenant B Building"})

        register_institution_handler("tenant_alpha", "navigation", mock_handler_a)
        register_institution_handler("tenant_beta", "navigation", mock_handler_b)

        try:
            handler_alpha = get_institution_handler("tenant_alpha", "navigation")
            handler_beta = get_institution_handler("tenant_beta", "navigation")
            handler_gamma = get_institution_handler("tenant_gamma", "navigation")

            self.assertEqual(handler_alpha, mock_handler_a)
            self.assertEqual(handler_beta, mock_handler_b)
            self.assertIsNone(handler_gamma)

            # Handler invocation isolation
            res_alpha = handler_alpha("query")
            self.assertEqual(res_alpha["name"], "Tenant A Lab")
            mock_handler_b.assert_not_called()
        finally:
            unregister_institution_handler("tenant_alpha")
            unregister_institution_handler("tenant_beta")

    def test_06_failing_structured_handler_safe_fallback(self):
        """Verify an exception in a structured handler does not crash the service."""
        broken_handler = MagicMock(side_effect=RuntimeError("Data file corrupt"))
        register_institution_handler("tenant_broken", "navigation", broken_handler)

        try:
            request = ChatTurnRequest(
                question="Where is the administrative block?",
                institution_id="tenant_broken",
            )
            with patch("ai_platform.service.assistant_service.ask_with_diagnostics") as mock_diag:
                mock_diag.return_value = {
                    "answer": "Fallback: Information not available.",
                    "evidence_status": "insufficient",
                    "pipeline_stop": True,
                }
                # Should not raise RuntimeError
                response = handle_chat_turn(request)
                self.assertEqual(len(response.structured_content), 0)
                self.assertIn("Fallback", response.answer)
        finally:
            unregister_institution_handler("tenant_broken")

    def test_07_generation_engine_error_represented_safely(self):
        """Verify an engine execution error is represented safely in ChatTurnResponse."""
        request = ChatTurnRequest(
            question="What are the admission fees?",
            institution_id="iitj",
        )
        with patch("ai_platform.service.assistant_service.ask_with_diagnostics") as mock_diag:
            mock_diag.side_effect = RuntimeError("Ollama connection timed out")
            response = handle_chat_turn(request)

            self.assertIsInstance(response, ChatTurnResponse)
            self.assertIn("❌ Error:", response.formatted_response)
            self.assertIn("Ollama connection timed out", response.formatted_response)
            self.assertTrue(response.is_safe_refusal)
            self.assertEqual(response.evidence_status, "error")

    def test_08_application_persistence_failure_non_blocking(self):
        """Verify database persistence failure does not block response generation in UI."""
        import sys
        if "streamlit" not in sys.modules:
            mock_st = MagicMock()
            sys.modules["streamlit"] = mock_st

        from UI.chat import get_response
        import streamlit as st

        class SessionStateMock(dict):
            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError:
                    raise AttributeError(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_chat = {"title": "Test Chat", "session_id": "test_session_id", "messages": []}
        mock_state = SessionStateMock({
            "active_chat": "chat_001",
            "conversations": {"chat_001": mock_chat},
        })

        with patch.object(st, "session_state", mock_state):
            # Simulate persistence failure in session loading
            with patch("UI.chat.load_persistent_chat_history", side_effect=RuntimeError("Supabase DB down")):
                # Should not raise exception
                answer, resp_time = get_response(
                    question="Where is the Lecture Hall Complex?",
                    graph=None,
                )
                self.assertIsInstance(answer, str)
                self.assertIn("Lecture Hall", answer)
                self.assertGreater(resp_time, 0.0)

    def test_09_ui_chat_get_response_compatibility_with_service(self):
        """Verify UI.chat.get_response operates smoothly with handle_chat_turn."""
        import sys
        if "streamlit" not in sys.modules:
            mock_st = MagicMock()
            sys.modules["streamlit"] = mock_st

        from UI.chat import get_response
        import streamlit as st

        class SessionStateMock(dict):
            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError:
                    raise AttributeError(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_chat = {"title": "Test Chat", "session_id": "test_session_id", "messages": []}
        mock_state = SessionStateMock({
            "active_chat": "chat_001",
            "conversations": {"chat_001": mock_chat},
        })

        with patch.object(st, "session_state", mock_state):
            answer, resp_time = get_response(
                question="What is the grading system at IIT Jodhpur?",
                graph=None,
            )
            self.assertIsInstance(answer, str)
            self.assertTrue(len(answer) > 10)
            self.assertGreater(resp_time, 0.0)


if __name__ == "__main__":
    unittest.main()
