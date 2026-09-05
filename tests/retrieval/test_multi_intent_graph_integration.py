"""
Phase 4 — Multi-Intent Graph Integration Tests

These tests verify that multi-intent planning is connected to the graph
without changing the existing single-intent route.
"""

from langchain_core.documents import Document

from backend.graph import (
    route_after_intent_planning,
    create_graph,
)
import backend.nodes as nodes


# =========================================================
# Routing
# =========================================================

def test_multi_intent_state_routes_to_multi_path():

    assert (
        route_after_intent_planning(
            {
                "is_multi_intent": True,
            }
        )
        == "multi_intent"
    )


def test_single_intent_state_routes_to_existing_path():

    assert (
        route_after_intent_planning(
            {
                "is_multi_intent": False,
            }
        )
        == "single_intent"
    )


def test_missing_multi_intent_flag_defaults_to_single_path():

    assert (
        route_after_intent_planning(
            {}
        )
        == "single_intent"
    )


# =========================================================
# Graph compilation
# =========================================================

def test_production_graph_compiles():

    graph = create_graph()

    assert graph is not None


# =========================================================
# Multi-intent orchestration
# =========================================================

def test_multi_intent_process_keeps_intents_independent(monkeypatch):

    def fake_pipeline(question):

        if "hostel" in question.lower():
            return {
                "evidence_status": "supported",
                "evidence_score": 0.91,
                "evidence_coverage_status": "supported",
                "evidence_question_type": "quantitative",
                "relevant_evidence_documents": 2,
                "evidence_strong_documents": 2,
                "evidence_partial_documents": 2,
                "evidence_combined_characters": 1200,
                "compressed_docs": [
                    Document(
                        page_content=(
                            "Hostel room rent is ₹300 per day "
                            "for double occupancy."
                        )
                    )
                ],
                "evidence_groups": [],
            }

        return {
            "evidence_status": "insufficient",
            "evidence_score": 0.12,
            "evidence_coverage_status": "insufficient",
            "evidence_question_type": "quantitative",
            "relevant_evidence_documents": 0,
            "evidence_strong_documents": 0,
            "evidence_partial_documents": 0,
            "evidence_combined_characters": 0,
            "compressed_docs": [],
            "evidence_groups": [],
        }

    monkeypatch.setattr(
        nodes,
        "_run_intent_evidence_pipeline",
        fake_pipeline,
    )

    result = nodes.process_multi_intent_node(
        {
            "intent_units": [
                {
                    "question": "What are the hostel fees?",
                    "topics": ["fees"],
                    "entities": ["hostel"],
                },
                {
                    "question": "What is the professor salary?",
                    "topics": ["salary"],
                    "entities": ["professor"],
                },
            ],
            "is_multi_intent": True,
            "intent_count": 2,
        }
    )

    assert len(
        result["intent_results"]
    ) == 2

    assert (
        result["intent_results"][0]["evidence_status"]
        == "supported"
    )

    assert (
        result["intent_results"][1]["evidence_status"]
        == "insufficient"
    )

    assert (
        result["evidence_coverage_status"]
        == "partially_supported"
    )

    assert (
        len(result["compressed_docs"])
        == 1
    )


def test_all_unsupported_multi_intents_become_insufficient(
    monkeypatch,
):

    def fake_pipeline(question):
        return {
            "evidence_status": "insufficient",
            "evidence_score": 0.0,
            "evidence_coverage_status": "insufficient",
            "evidence_question_type": "descriptive",
            "relevant_evidence_documents": 0,
            "evidence_strong_documents": 0,
            "evidence_partial_documents": 0,
            "evidence_combined_characters": 0,
            "compressed_docs": [],
            "evidence_groups": [],
        }

    monkeypatch.setattr(
        nodes,
        "_run_intent_evidence_pipeline",
        fake_pipeline,
    )

    result = nodes.process_multi_intent_node(
        {
            "intent_units": [
                {
                    "question": "What is the professor salary?",
                    "topics": [],
                    "entities": [],
                },
                {
                    "question": "What is the latest 2030 hostel fee?",
                    "topics": [],
                    "entities": [],
                },
            ],
            "is_multi_intent": True,
            "intent_count": 2,
        }
    )

    assert result["evidence_status"] == "insufficient"
    assert result["evidence_coverage_status"] == "insufficient"
    assert result["compressed_docs"] == []