"""
IIT Jodhpur V1 — Phase 8D4
Authoritative retrieval query propagation.

Focused situation-aware requests must use the generated primary query for
all retrieval/evidence-selection stages after initial retrieval.

Broad institutional questions retain the resolved user question.
"""

from __future__ import annotations

from unittest.mock import patch

from backend.nodes import (
    _authoritative_retrieval_query,
)


def test_broadly_classified_specific_request_uses_generated_query():
    state = {
        "question": "What are the admission routes for M.Sc.?",
        "resolved_question": "What are the admission routes for M.Sc.?",
        "generated_queries": [
            "m.sc admission application requirements",
        ],
    }

    assert _authoritative_retrieval_query(
        state
    ) == "m.sc admission application requirements"


def test_focused_request_uses_first_generated_query_only():
    state = {
        "question": "What is the M.Tech eligibility?",
        "resolved_question": "What is the M.Tech eligibility?",
        "generated_queries": [
            "m.tech eligibility requirements",
            "m.tech admission criteria",
        ],
    }

    assert _authoritative_retrieval_query(
        state
    ) == "m.tech eligibility requirements"


def test_focused_request_falls_back_to_original_without_generated_query():
    state = {
        "question": "Where is the library?",
        "resolved_question": "Where is the library?",
        "generated_queries": [],
    }

    assert _authoritative_retrieval_query(
        state
    ) == "Where is the library?"


def test_broad_request_keeps_original_as_authoritative():
    state = {
        "question": "What academic programs are available at IIT Jodhpur?",
        "resolved_question": "What academic programs are available at IIT Jodhpur?",
        "generated_queries": [
            "academic programs admissions departments research",
        ],
    }

    assert _authoritative_retrieval_query(
        state
    ) == "What academic programs are available at IIT Jodhpur?"
