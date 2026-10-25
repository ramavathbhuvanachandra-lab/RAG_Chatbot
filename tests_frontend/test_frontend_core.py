"""Frontend Unit and Integration Test Suite.

Verifies:
1. Plugin and configuration schema validation.
2. IITJ plugin resolution and real data loading.
3. Handling of missing optional capabilities (e.g. no campus map).
4. Tenant and plugin identity mismatch protection.
5. Assistant client contract adaptation and safe refusal handling.
6. Session manager conversation lifecycle, unique session IDs, and chat switching.
7. Context isolation and bounded history windowing.
8. Architectural hygiene: zero legacy backend RAG imports and zero mandatory Supabase dependencies.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.clients.assistant_client import AssistantResponse, send_chat_turn
from frontend.core.config_loader import (
    get_active_tenant_id,
    load_institution_ui_config,
)
from frontend.core.models import (
    CampusLocationItem,
    ChatMessage,
    Conversation,
    EmergencyContactItem,
    InstitutionUIConfig,
    QuickActionItem,
)
from frontend.core.plugin_registry import (
    InstitutionUIPlugin,
    get_ui_plugin,
    list_registered_plugins,
    register_ui_plugin,
    unregister_ui_plugin,
)
from frontend.core.session_manager import (
    add_assistant_message,
    add_user_message,
    create_new_conversation,
    get_active_conversation,
    get_bounded_history,
    init_session_state,
    switch_conversation,
)


class MockSessionState(dict):
    """Dict-like mock of Streamlit session state supporting attribute access."""

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key: str, value):
        self[key] = value


class TestConfigAndPluginValidation(unittest.TestCase):
    """Tests for UI plugin registration, resolution, and configuration schema validation."""

    def test_iitj_plugin_resolution(self):
        """Verify IITJ plugin resolves with valid branding, locations, and quick actions."""
        plugin = get_ui_plugin("iitj")
        self.assertIsNotNone(plugin, "IITJ plugin must be registered.")
        config = plugin.get_config()
        config.validate()

        self.assertEqual(config.institution_id, "iitj")
        self.assertEqual(config.display_name, "IIT Jodhpur")
        self.assertTrue(config.has_campus_map)
        self.assertGreater(len(config.campus_map_locations), 0)
        self.assertTrue(config.has_emergency_cards)
        self.assertGreater(len(config.emergency_contacts), 0)
        self.assertGreater(len(config.quick_actions), 0)

    def test_template_plugin_omits_optional_capabilities(self):
        """Verify template plugin resolves cleanly while omitting optional features like campus map."""
        plugin = get_ui_plugin("template")
        self.assertIsNotNone(plugin, "Template plugin must be registered.")
        config = plugin.get_config()
        config.validate()

        self.assertEqual(config.institution_id, "template")
        self.assertFalse(config.has_campus_map)
        self.assertEqual(len(config.campus_map_locations), 0)
        self.assertFalse(config.has_emergency_cards)

    def test_config_validation_rejects_invalid_institution_id(self):
        """Verify invalid institution IDs fail schema validation."""
        bad_config = InstitutionUIConfig(
            institution_id="Invalid Space ID",
            display_name="Test College",
        )
        with self.assertRaises(ValueError):
            bad_config.validate()

    def test_config_validation_rejects_empty_display_name(self):
        """Verify empty display name fails schema validation."""
        bad_config = InstitutionUIConfig(
            institution_id="valid_id",
            display_name="   ",
        )
        with self.assertRaises(ValueError):
            bad_config.validate()

    def test_campus_map_validation_rejects_missing_locations(self):
        """Verify declaring has_campus_map=True with no locations raises ValueError."""
        bad_config = InstitutionUIConfig(
            institution_id="sample_college",
            display_name="Sample College",
            has_campus_map=True,
            campus_map_locations=(),
        )
        with self.assertRaises(ValueError):
            bad_config.validate()

    def test_unknown_institution_raises_key_error(self):
        """Verify requesting an unknown institution ID raises KeyError."""
        with self.assertRaises(KeyError):
            load_institution_ui_config("unknown_college_xyz")

    def test_tenant_identity_mismatch_detection(self):
        """Verify mismatched configuration identity is caught by loader."""
        class MaliciousPlugin(InstitutionUIPlugin):
            @property
            def institution_id(self) -> str:
                return "tenant_a"
            def get_config(self) -> InstitutionUIConfig:
                return InstitutionUIConfig(
                    institution_id="tenant_b",  # Mismatched!
                    display_name="Deceptive College",
                )

        plugin = MaliciousPlugin()
        register_ui_plugin(plugin)
        try:
            with self.assertRaises(ValueError):
                load_institution_ui_config("tenant_a", verify_ai_profile=False)
        finally:
            unregister_ui_plugin("tenant_a")


class TestSessionManager(unittest.TestCase):
    """Tests for temporary Streamlit conversation lifecycles and context isolation."""

    def setUp(self):
        self.mock_state = MockSessionState()
        self.patcher = patch("frontend.core.session_manager.st.session_state", self.mock_state)
        self.patcher.start()
        self.config = get_ui_plugin("iitj").get_config()
        init_session_state(self.config)

    def tearDown(self):
        self.patcher.stop()

    def test_init_creates_fresh_conversation_with_unique_id(self):
        """Verify session initialization establishes active conversation with unique ID."""
        conv = get_active_conversation()
        self.assertIsNotNone(conv)
        self.assertTrue(conv.id.startswith("chat_"))
        self.assertEqual(len(conv.messages), 1)
        self.assertEqual(conv.messages[0].role, "assistant")

    def test_create_new_conversation_uniqueness(self):
        """Verify multiple created conversations possess distinct IDs."""
        id1 = create_new_conversation(self.config, title="Chat 1")
        id2 = create_new_conversation(self.config, title="Chat 2")
        self.assertNotEqual(id1, id2)
        self.assertEqual(self.mock_state.active_chat_id, id2)
        self.assertEqual(len(self.mock_state.conversations), 3)  # initial + 2

    def test_conversation_switching_and_isolation(self):
        """Verify switching conversations does not leak messages across threads."""
        id1 = self.mock_state.active_chat_id
        add_user_message("Question for Chat 1")

        id2 = create_new_conversation(self.config, title="Chat 2")
        add_user_message("Question for Chat 2")

        # Verify chat 2 has only its question
        conv2 = get_active_conversation()
        self.assertEqual(conv2.id, id2)
        user_msgs_2 = [m.content for m in conv2.messages if m.role == "user"]
        self.assertEqual(user_msgs_2, ["Question for Chat 2"])

        # Switch back to chat 1
        switch_conversation(id1)
        conv1 = get_active_conversation()
        self.assertEqual(conv1.id, id1)
        user_msgs_1 = [m.content for m in conv1.messages if m.role == "user"]
        self.assertEqual(user_msgs_1, ["Question for Chat 1"])

    def test_bounded_history_truncation(self):
        """Verify get_bounded_history enforces the specified maximum turns."""
        for i in range(15):
            add_user_message(f"User turn {i}")
            add_assistant_message(
                AssistantResponse(
                    answer=f"Assistant turn {i}",
                    formatted_response=f"Assistant turn {i}",
                )
            )

        history = get_bounded_history(max_turns=3)
        # 3 turns = 6 messages max
        self.assertLessEqual(len(history), 6)
        self.assertEqual(history[-1]["content"], "Assistant turn 14")
        self.assertEqual(history[-2]["content"], "User turn 14")

    def test_tenant_change_resets_conversation_state(self):
        """Verify switching institution tenants clears conversations to prevent cross-tenant leaks."""
        add_user_message("Confidential inquiry")
        self.assertIn("chat_", self.mock_state.active_chat_id)

        template_config = get_ui_plugin("template").get_config()
        init_session_state(template_config)

        self.assertEqual(self.mock_state.active_institution_id, "template")
        conv = get_active_conversation()
        # Should be a fresh conversation without the previous user message
        user_msgs = [m for m in conv.messages if m.role == "user"]
        self.assertEqual(len(user_msgs), 0)


class TestAssistantClient(unittest.TestCase):
    """Tests for the presentation-layer assistant client adapter."""

    @patch("frontend.clients.assistant_client.handle_chat_turn")
    def test_client_adapts_service_response(self, mock_handle):
        """Verify client correctly maps canonical ChatTurnResponse to AssistantResponse."""
        from ai_platform.service.assistant_service import ChatTurnResponse

        mock_handle.return_value = ChatTurnResponse(
            answer="B.Tech requires 8 semesters.",
            structured_content=[{"type": "navigation", "data": {"name": "LHC"}}],
            sources=[{"source": "btech.docx", "heading": "Ordinances", "text": "Curriculum"}],
            latency_seconds=1.45,
            is_safe_refusal=False,
            evidence_status="supported",
            formatted_response="📍 LHC\n\n---\n\nB.Tech requires 8 semesters.",
        )

        resp = send_chat_turn(
            question="What is the B.Tech duration?",
            history=[{"role": "user", "content": "hi"}],
            session_id="session_test",
            institution_id="iitj",
        )

        self.assertEqual(resp.answer, "B.Tech requires 8 semesters.")
        self.assertEqual(len(resp.structured_blocks), 1)
        self.assertEqual(len(resp.sources), 1)
        self.assertEqual(resp.latency_seconds, 1.45)
        self.assertFalse(resp.is_safe_refusal)
        self.assertEqual(resp.evidence_status, "supported")

    def test_empty_query_safe_handling(self):
        """Verify empty query returns immediate safe response without invoking service."""
        resp = send_chat_turn("   ")
        self.assertTrue(resp.is_safe_refusal)
        self.assertEqual(resp.evidence_status, "empty_query")

    @patch("frontend.clients.assistant_client.handle_chat_turn")
    def test_service_exception_graceful_abstention(self, mock_handle):
        """Verify unexpected service exceptions are handled safely without exposing internal details."""
        mock_handle.side_effect = RuntimeError("Internal Chroma connection timeout at 127.0.0.1")

        resp = send_chat_turn(
            question="What is the fee?",
            session_id="session_test",
            institution_id="iitj",
        )

        self.assertTrue(resp.is_safe_refusal)
        self.assertEqual(resp.evidence_status, "service_error")
        # Ensure no sensitive stack trace or internal network paths are exposed
        self.assertNotIn("Chroma", resp.answer)
        self.assertNotIn("127.0.0.1", resp.answer)
        self.assertIn("error occurred while retrieving information", resp.answer)


class TestArchitectureHygiene(unittest.TestCase):
    """Tests guaranteeing zero legacy backend RAG imports and zero Supabase startup dependencies."""

    def test_no_legacy_backend_imports_in_frontend(self):
        """Verify frontend source files have zero direct imports of legacy backend modules."""
        frontend_dir = REPO_ROOT / "frontend"
        failures: list[str] = []

        for py_path in frontend_dir.rglob("*.py"):
            with open(py_path, "r", encoding="utf-8") as fp:
                tree = ast.parse(fp.read(), filename=str(py_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("backend"):
                            failures.append(f"{py_path.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("backend"):
                        failures.append(f"{py_path.name}: from {node.module} import ...")

        self.assertEqual(
            failures,
            [],
            f"Frontend must not import from legacy backend modules: {failures}",
        )

    def test_frontend_loads_cleanly_without_supabase(self):
        """Verify frontend packages import without requiring the supabase library."""
        # Check that importing frontend components succeeds even if supabase is None
        with patch.dict(sys.modules, {"supabase": None}):
            import frontend.app
            import frontend.core.config_loader
            import frontend.core.session_manager
            import frontend.clients.assistant_client

            self.assertIsNotNone(frontend.app)
            self.assertIsNotNone(frontend.clients.assistant_client)


class TestDeveloperDiagnosticsInspector(unittest.TestCase):
    """Verifies per-turn developer diagnostics storage, bounding, and inspection."""

    def setUp(self):
        self.mock_state = MockSessionState({
            "active_institution_id": "iitj",
            "conversations": {},
            "active_chat_id": None,
            "show_campus_map": False,
            "show_retrieval_diagnostics": True,
            "execution_traces": [],
            "selected_trace_id": None,
        })
        self.patcher = patch("frontend.core.session_manager.st.session_state", self.mock_state)
        self.patcher.start()

        self.config = InstitutionUIConfig(
            institution_id="iitj",
            display_name="IIT Jodhpur",
            welcome_message="Welcome!",
        )
        init_session_state(self.config)

    def tearDown(self):
        self.patcher.stop()

    def test_trace_capture_and_bounded_retention(self):
        """Verify traces are captured per turn and bounded to MAX_EXECUTION_TRACES (15)."""
        from frontend.core.session_manager import MAX_EXECUTION_TRACES

        # Simulate 20 conversational turns
        for i in range(1, 21):
            add_user_message(f"Query {i}")
            fake_resp = AssistantResponse(
                answer=f"Answer {i}",
                formatted_response=f"Answer {i}",
                latency_seconds=0.15,
                diagnostics={
                    "funnel": {"retrieved_total": 10, "verified_count": 5},
                    "panel_query_understanding": {"user_input": {"original_question": f"Query {i}"}},
                    "panel_dense_retrieval": {"candidates": []},
                    "panel_bm25_retrieval": {"candidates": []},
                    "panel_hybrid_fusion": {"top_15_candidates": []},
                    "panel_ranking_verification": {"decisions": []},
                    "panel_packaged_context": {"packaged_units": []},
                    "panel_answer_generation": {"answer": f"Answer {i}"},
                    "panel_stage_timings": {"timings": {"resolve_conversation": 0.01}},
                },
            )
            add_assistant_message(fake_resp)

        traces = self.mock_state["execution_traces"]
        self.assertEqual(len(traces), MAX_EXECUTION_TRACES)
        # The latest trace should be query 20
        self.assertEqual(traces[-1]["query"], "Query 20")
        self.assertEqual(self.mock_state["selected_trace_id"], traces[-1]["turn_id"])

    def test_render_developer_trace_inspector_runs_cleanly(self):
        """Verify render_developer_trace_inspector renders without errors when traces are present."""
        from frontend.components.diagnostics import render_developer_trace_inspector

        add_user_message("What is the M.Tech tuition fee?")
        fake_resp = AssistantResponse(
            answer="The tuition fee is 15000.",
            formatted_response="The tuition fee is 15000.",
            latency_seconds=0.45,
            diagnostics={
                "funnel": {"retrieved_total": 15, "verified_count": 8, "rejected_count": 2},
                "panel_query_understanding": {
                    "user_input": {"original_question": "What is the M.Tech tuition fee?"},
                    "interpretation": {"resolved_question": "What is the M.Tech tuition fee?"},
                    "retrieval_handoff": {"recipient_node": "CoreNodes.hybrid_retrieve_node"},
                },
                "panel_dense_retrieval": {"candidates": []},
                "panel_bm25_retrieval": {"candidates": []},
                "panel_hybrid_fusion": {"top_15_candidates": []},
                "panel_ranking_verification": {"decisions": []},
                "panel_packaged_context": {"packaged_units": [], "estimated_tokens": 300},
                "panel_answer_generation": {"answer": "The tuition fee is 15000."},
                "panel_stage_timings": {"timings": {"dense_retrieve": 0.05}},
            },
        )
        add_assistant_message(fake_resp)

        # Patch st.expander, st.selectbox, etc. in diagnostics module to ensure execution does not crash
        with patch("frontend.components.diagnostics.st.session_state", self.mock_state):
            with patch("frontend.components.diagnostics.st.expander"):
                with patch("frontend.components.diagnostics.st.selectbox", return_value=self.mock_state["execution_traces"][0]["turn_id"]):
                    with patch("frontend.components.diagnostics.st.columns", side_effect=lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]):
                        render_developer_trace_inspector()


if __name__ == "__main__":
    unittest.main()
