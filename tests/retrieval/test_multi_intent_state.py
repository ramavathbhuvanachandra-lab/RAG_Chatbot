"""
Phase 4 — Multi-Intent State Tests
"""

from backend.state import (
    GraphState,
    IntentState,
)


def test_graph_state_accepts_multi_intent_fields():

    state: GraphState = {
        "question": (
            "What are the Ph.D. requirements "
            "and what are the hostel fees?"
        ),
        "is_multi_intent": True,
        "intent_count": 2,
        "intent_questions": [
            "What are the Ph.D. requirements?",
            "What are the hostel fees?",
        ],
    }

    assert state["is_multi_intent"] is True
    assert state["intent_count"] == 2
    assert len(
        state["intent_questions"]
    ) == 2


def test_intent_state_holds_independent_request():

    intent: IntentState = {
        "question": (
            "What are the hostel fees?"
        ),
        "topics": [
            "fees",
        ],
        "entities": [
            "hostel",
        ],
        "evidence_status": "supported",
        "evidence_coverage_status": "supported",
        "answer": (
            "The hostel charges are..."
        ),
    }

    assert intent["question"]
    assert intent["evidence_status"] == "supported"


def test_single_intent_can_use_empty_intent_results():

    state: GraphState = {
        "question": (
            "What are the hostel fees?"
        ),
        "is_multi_intent": False,
        "intent_count": 1,
        "intent_units": [],
        "intent_results": [],
    }

    assert state["is_multi_intent"] is False
    assert state["intent_count"] == 1


def test_intent_results_can_store_mixed_evidence_status():

    state: GraphState = {
        "question": (
            "What are the hostel fees and "
            "what is the professor salary?"
        ),
        "is_multi_intent": True,
        "intent_count": 2,
        "intent_results": [
            {
                "question": "What are the hostel fees?",
                "evidence_status": "supported",
            },
            {
                "question": "What is the professor salary?",
                "evidence_status": "insufficient",
            },
        ],
    }

    assert (
        state["intent_results"][0][
            "evidence_status"
        ]
        == "supported"
    )

    assert (
        state["intent_results"][1][
            "evidence_status"
        ]
        == "insufficient"
    )
