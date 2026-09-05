"""
IIT Jodhpur V1 — Phase 8D3
Situation-aware primary reranking integration tests.

The critical regression covered here is:
    broad classifier = True
    but generated query = concrete program/entity query

The concrete planner query must become the primary reranking query.
"""

from __future__ import annotations

from unittest.mock import patch

from backend.nodes import initial_rerank_documents


def _capture_rerank():
    captured = {}

    def fake_rerank_documents(
        *,
        query,
        documents,
        top_k,
        query_variants,
    ):
        captured["query"] = query
        captured["query_variants"] = query_variants
        return []

    return captured, fake_rerank_documents


def test_broad_classifier_does_not_override_specific_program_query():
    state = {
        "question": "What are the admission routes for M.Sc.?",
        "resolved_question": "What are the admission routes for M.Sc.?",
        "generated_queries": [
            "m.sc admission application requirements",
        ],
        "retrieval_queries": [
            "What are the admission routes for M.Sc.?",
        ],
        "fused_docs": [],
    }

    captured, fake = _capture_rerank()

    with patch(
        "backend.nodes.rerank_documents",
        side_effect=fake,
    ):
        initial_rerank_documents(state)

    assert captured["query"] == (
        "m.sc admission application requirements"
    )
    assert captured["query_variants"] == []


def test_broad_classifier_keeps_original_for_truly_broad_request():
    state = {
        "question": "What academic programs are available at IIT Jodhpur?",
        "resolved_question": "What academic programs are available at IIT Jodhpur?",
        "generated_queries": [
            "academic programs admissions departments research",
        ],
        "retrieval_queries": [
            "What academic programs are available at IIT Jodhpur?",
            "academic programs IIT Jodhpur",
            "available programs IIT Jodhpur",
        ],
        "fused_docs": [],
    }

    captured, fake = _capture_rerank()

    with patch(
        "backend.nodes.rerank_documents",
        side_effect=fake,
    ):
        initial_rerank_documents(state)

    assert captured["query"] == (
        "What academic programs are available at IIT Jodhpur?"
    )
    assert captured["query_variants"] == [
        "academic programs IIT Jodhpur",
        "available programs IIT Jodhpur",
    ]


def test_specific_generated_entity_becomes_primary_for_broad_question():
    state = {
        "question": "What admission opportunities are available for M.Tech?",
        "resolved_question": "What admission opportunities are available for M.Tech?",
        "generated_queries": [
            "m.tech admission application requirements",
            "m.tech eligibility criteria",
        ],
        "retrieval_queries": [
            "What admission opportunities are available for M.Tech?",
        ],
        "fused_docs": [],
    }

    captured, fake = _capture_rerank()

    with patch(
        "backend.nodes.rerank_documents",
        side_effect=fake,
    ):
        initial_rerank_documents(state)

    assert captured["query"] == (
        "m.tech admission application requirements"
    )
    assert captured["query_variants"] == [
        "m.tech eligibility criteria",
    ]