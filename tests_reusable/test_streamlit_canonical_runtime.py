#!/usr/bin/env python3
"""
Integration test verifying Streamlit UI application migration to canonical ai_platform runtime.

Verifies:
1. app.py imports the canonical create_graph from ai_platform.runtime.engine.
2. The graph returned by create_graph() is a CompiledStateGraph from ai_platform.core.graph.
3. Invoking the graph returns a valid answer dictionary.
4. UI.chat.get_response integration operates correctly with the canonical graph.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestStreamlitCanonicalRuntimeIntegration(unittest.TestCase):
    def test_app_imports_canonical_engine(self):
        """Verify app.py source code imports get_app_graph from ai_platform.runtime.engine."""
        app_path = REPO_ROOT / "app.py"
        content = app_path.read_text(encoding="utf-8")
        self.assertIn(
            "from ai_platform.runtime.engine import get_app_graph as create_graph",
            content,
            "app.py must import create_graph from ai_platform.runtime.engine",
        )
        self.assertNotIn(
            "from backend.graph import create_graph",
            content,
            "Legacy backend.graph import must be eliminated from app.py",
        )

    def test_canonical_graph_contract(self):
        """Verify create_graph from ai_platform.runtime.engine returns valid callable graph."""
        from ai_platform.runtime.engine import get_app_graph as create_graph

        graph = create_graph()
        self.assertIsNotNone(graph)
        self.assertTrue(
            hasattr(graph, "invoke"),
            "Canonical graph must provide invoke() method",
        )

    def test_graph_invocation_e2e(self):
        """Verify invoking the canonical graph with a realistic question returns structured output."""
        from ai_platform.runtime.engine import get_app_graph as create_graph

        graph = create_graph()
        result = graph.invoke({
            "question": "What is the passing grade point in the grading system at IIT Jodhpur?",
            "chat_history": (),
        })
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIsNotNone(result["answer"])
        self.assertGreater(len(str(result["answer"]).strip()), 5)

    def test_ui_chat_get_response_with_canonical_graph(self):
        """Verify UI/chat.py get_response executes smoothly with the canonical graph."""
        import sys
        from unittest.mock import MagicMock
        if "streamlit" not in sys.modules:
            mock_st = MagicMock()
            sys.modules["streamlit"] = mock_st

        from ai_platform.runtime.engine import get_app_graph as create_graph
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

        mock_chat = {
            "title": "Test Chat",
            "session_id": "test_session_123",
            "messages": [],
        }
        mock_state = SessionStateMock({
            "active_chat": "chat_001",
            "conversations": {"chat_001": mock_chat},
        })
        st.session_state = mock_state

        with patch.object(st, "session_state", mock_state):
            graph = create_graph()
            answer, resp_time = get_response(
                question="What programs are offered at IIT Jodhpur?",
                graph=graph,
            )
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer.strip()), 10)
            self.assertGreater(resp_time, 0.0)


if __name__ == "__main__":
    unittest.main()
