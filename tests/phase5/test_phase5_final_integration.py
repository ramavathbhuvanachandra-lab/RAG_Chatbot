"""
Phase 5.8 — Final Phase-5 Integration Tests

The suite validates the integration boundary between:
    conversation → situation → intent planning → independent evidence
    → situational answer context → one final answer call.

External model execution is mocked at the answer-chain boundary for
deterministic tests. Retrieval is mocked where the test is validating
orchestration rather than retrieval quality.
"""

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from backend.graph import create_graph
from backend.nodes import (
    generate_answer,
    plan_multi_intent_node,
    process_multi_intent_node,
)


# =========================================================
# Shared fixtures
# =========================================================

def _phd_situation():
    return SimpleNamespace(
        intent="phd_eligibility",
        goal="determine_eligibility",
        entities=("phd", "b.tech"),
        user_facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (74.0,),
        },
        constraints={},
        preferences={},
    )


def _phd_decision():
    return SimpleNamespace(
        target="phd",
        goal="determine_eligibility",
        facts={
            "degree": "bachelors_degree",
            "degree_duration_years": 4,
            "percentages": (74.0,),
        },
        constraints={},
        preferences={},
        missing_information=(),
    )


def _supported_pipeline(question):
    return {
        "evidence_status": "supported",
        "evidence_score": 0.92,
        "evidence_coverage_status": "supported",
        "evidence_question_type": "descriptive",
        "relevant_evidence_documents": 1,
        "evidence_strong_documents": 1,
        "evidence_partial_documents": 0,
        "evidence_combined_characters": 500,
        "compressed_docs": [
            Document(
                page_content=(
                    "The institute provides the requested information."
                )
            )
        ],
        "evidence_groups": [],
        "contradiction_status": "none",
        "contradiction_count": 0,
        "contradictions": [],
    }


# =========================================================
# Graph / planning
# =========================================================

def test_production_graph_compiles():
    assert create_graph() is not None


def test_two_independent_requests_are_planned():
    result = plan_multi_intent_node(
        {
            "question": (
                "What are the hostel fees and where is the library?"
            ),
        }
    )

    assert result["is_multi_intent"] is True
    assert result["intent_count"] == 2
    assert len(result["intent_units"]) == 2


def test_three_independent_requests_are_planned():
    result = plan_multi_intent_node(
        {
            "question": (
                "What are the Ph.D. requirements, where is the library, "
                "and what are the hostel fees?"
            ),
        }
    )

    assert result["is_multi_intent"] is True
    assert result["intent_count"] == 3
    assert len(result["intent_units"]) == 3


def test_conversational_preamble_plus_question_is_split():
    result = plan_multi_intent_node(
        {
            "question": (
                "I have 74% in B.Tech. Can I apply for Ph.D. "
                "and where is the library?"
            ),
        }
    )

    assert result["is_multi_intent"] is True
    assert result["intent_count"] == 2

    assert (
        "ph.d"
        in result["intent_questions"][0].lower()
        or "phd"
        in result["intent_questions"][0].lower()
    )

    assert (
        "library"
        in result["intent_questions"][1].lower()
    )


def test_single_intent_remains_single():
    result = plan_multi_intent_node(
        {
            "question": (
                "What research areas are available in Electrical Engineering?"
            ),
        }
    )

    assert result["is_multi_intent"] is False
    assert result["intent_count"] == 1


# =========================================================
# Multi-intent orchestration
# =========================================================

def test_multi_intent_keeps_evidence_per_request(monkeypatch):
    monkeypatch.setattr(
        "backend.nodes._run_intent_evidence_pipeline",
        _supported_pipeline,
    )

    result = process_multi_intent_node(
        {
            "intent_units": [
                {
                    "question": "What are the hostel fees?",
                    "topics": ["fees"],
                    "entities": ["hostel"],
                },
                {
                    "question": "Where is the library?",
                    "topics": ["navigation"],
                    "entities": ["library"],
                },
            ],
            "is_multi_intent": True,
            "intent_count": 2,
        }
    )

    assert len(result["intent_results"]) == 2
    assert all(
        item["evidence_status"] == "supported"
        for item in result["intent_results"]
    )


def test_multi_intent_can_have_mixed_evidence_status(monkeypatch):
    def pipeline(question):
        if "hostel" in question.lower():
            return _supported_pipeline(
                question
            )

        return {
            "evidence_status": "insufficient",
            "evidence_score": 0.05,
            "evidence_coverage_status": "insufficient",
            "evidence_question_type": "descriptive",
            "relevant_evidence_documents": 0,
            "evidence_strong_documents": 0,
            "evidence_partial_documents": 0,
            "evidence_combined_characters": 0,
            "compressed_docs": [],
            "evidence_groups": [],
            "contradiction_status": "none",
            "contradiction_count": 0,
            "contradictions": [],
        }

    monkeypatch.setattr(
        "backend.nodes._run_intent_evidence_pipeline",
        pipeline,
    )

    result = process_multi_intent_node(
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


def test_all_unsupported_intents_are_insufficient(monkeypatch):
    def pipeline(question):
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
            "contradiction_status": "none",
            "contradiction_count": 0,
            "contradictions": [],
        }

    monkeypatch.setattr(
        "backend.nodes._run_intent_evidence_pipeline",
        pipeline,
    )

    result = process_multi_intent_node(
        {
            "intent_units": [
                {
                    "question": "What is the professor salary?",
                    "topics": [],
                    "entities": [],
                },
                {
                    "question": "What is the 2030 hostel fee?",
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


def test_phd_situation_attaches_only_to_phd_intent(monkeypatch):
    monkeypatch.setattr(
        "backend.nodes._run_intent_evidence_pipeline",
        _supported_pipeline,
    )

    result = process_multi_intent_node(
        {
            "intent_units": [
                {
                    "question": "Can I apply for Ph.D. admission?",
                    "topics": ["eligibility"],
                    "entities": ["phd"],
                },
                {
                    "question": "Where is the library?",
                    "topics": ["navigation"],
                    "entities": ["library"],
                },
            ],
            "is_multi_intent": True,
            "intent_count": 2,
            "student_situation": _phd_situation(),
            "decision_context": _phd_decision(),
        }
    )

    first = result["intent_results"][0]["situation_context"]
    second = result["intent_results"][1]["situation_context"]

    assert first["applies"] is True
    assert second["applies"] is False
    assert second["facts"] == {}
    assert second["constraints"] == {}


# =========================================================
# Final answer generation
# =========================================================

def _answer_state(question, *, multi=False):
    return {
        "question": question,
        "resolved_question": question,
        "chat_history": [],
        "evidence_status": "supported",
        "evidence_coverage_status": "supported",
        "evidence_question_type": (
            "multi_intent"
            if multi
            else "descriptive"
        ),
        "contradiction_status": "none",
        "is_multi_intent": multi,
        "compressed_docs": [
            Document(
                page_content=(
                    "Applicants with a four-year bachelor's degree "
                    "require the documented minimum marks."
                )
            )
        ],
    }


def test_generate_answer_uses_one_answer_chain_call():
    state = _answer_state(
        "What are the Ph.D. requirements?"
    )

    with patch(
        "backend.nodes.answer_chain"
    ) as chain:

        chain.invoke.return_value = SimpleNamespace(
            content=(
                "The documented Ph.D. requirements include the "
                "relevant academic qualification and marks."
            )
        )

        result = generate_answer(
            state
        )

        assert chain.invoke.call_count == 1
        assert result["answer"]


def test_generate_answer_can_include_situation_context():
    state = _answer_state(
        "I have 74% in B.Tech. Can I apply for Ph.D.?"
    )

    state["student_situation"] = _phd_situation()
    state["decision_context"] = _phd_decision()

    with patch(
        "backend.nodes.answer_chain"
    ) as chain:

        chain.invoke.return_value = SimpleNamespace(
            content=(
                "The documented eligibility route requires a "
                "four-year bachelor's degree and the required marks."
            )
        )

        generate_answer(
            state
        )

        payload = chain.invoke.call_args.args[0]

        assert "SITUATION CONTEXT:" in payload["question"]
        assert "74.0" in payload["question"]


def test_multi_intent_answer_uses_one_answer_chain_call():
    state = _answer_state(
        "What are the hostel fees and where is the library?",
        multi=True,
    )

    state["intent_results"] = [
        {
            "question": "What are the hostel fees?",
            "evidence_status": "supported",
            "evidence_coverage_status": "supported",
            "evidence_question_type": "quantitative",
            "compressed_docs": [
                Document(
                    page_content=(
                        "Double occupancy costs ₹300 per day."
                    )
                )
            ],
            "situation_context": {
                "applies": False,
                "facts": {},
                "preferences": {},
                "constraints": {},
            },
        },
        {
            "question": "Where is the library?",
            "evidence_status": "supported",
            "evidence_coverage_status": "supported",
            "evidence_question_type": "descriptive",
            "compressed_docs": [
                Document(
                    page_content=(
                        "The library is located on campus."
                    )
                )
            ],
            "situation_context": {
                "applies": False,
                "facts": {},
                "preferences": {},
                "constraints": {},
            },
        },
    ]

    from backend.multi_intent_answer import (
        build_multi_intent_answer_package,
    )

    state["multi_intent_answer_package"] = (
        build_multi_intent_answer_package(
            state["intent_results"]
        )
    )

    with patch(
        "backend.nodes.answer_chain"
    ) as chain:

        chain.invoke.return_value = SimpleNamespace(
            content=(
                "1. The hostel charge is ₹300 per day for the "
                "supported category.\n"
                "2. The library is located on campus."
            )
        )

        result = generate_answer(
            state
        )

        assert chain.invoke.call_count == 1
        assert result["answer"]


def test_grounded_single_intent_has_final_context():
    state = _answer_state(
        "What research is available?"
    )

    with patch(
        "backend.nodes.answer_chain"
    ) as chain:

        chain.invoke.return_value = SimpleNamespace(
            content=(
                "The available evidence describes research "
                "opportunities."
            )
        )

        result = generate_answer(
            state
        )

    assert result["context"]
    assert result["answer"]


def test_insufficient_evidence_never_calls_answer_chain():
    state = _answer_state(
        "What is something unknown?"
    )

    state["evidence_status"] = "insufficient"

    with patch(
        "backend.nodes.answer_chain"
    ) as chain:

        result = generate_answer(
            state
        )

        chain.invoke.assert_not_called()

    assert (
        "don't know"
        in result["answer"].lower()
    )


def test_phase5_state_contains_situation_fields_when_planned():
    result = plan_multi_intent_node(
        {
            "question": (
                "I have 74% in B.Tech. Can I apply for Ph.D.?"
            ),
        }
    )

    assert "student_situation" in result
    assert "decision_context" in result
    assert result["student_situation"] is not None
    assert result["decision_context"] is not None


def test_phase5_preserves_one_answer_call_for_three_requests():
    state = _answer_state(
        (
            "What are the Ph.D. requirements, where is the library, "
            "and what are the hostel fees?"
        ),
        multi=True,
    )

    state["intent_results"] = [
        {
            "question": "What are the Ph.D. requirements?",
            "evidence_status": "supported",
            "evidence_coverage_status": "supported",
            "evidence_question_type": "descriptive",
            "compressed_docs": [
                Document(
                    page_content="Ph.D. requirements are documented."
                )
            ],
            "situation_context": {
                "applies": False,
                "facts": {},
                "preferences": {},
                "constraints": {},
            },
        },
        {
            "question": "Where is the library?",
            "evidence_status": "supported",
            "evidence_coverage_status": "supported",
            "evidence_question_type": "descriptive",
            "compressed_docs": [
                Document(
                    page_content="The library is on campus."
                )
            ],
            "situation_context": {
                "applies": False,
                "facts": {},
                "preferences": {},
                "constraints": {},
            },
        },
        {
            "question": "What are the hostel fees?",
            "evidence_status": "supported",
            "evidence_coverage_status": "supported",
            "evidence_question_type": "quantitative",
            "compressed_docs": [
                Document(
                    page_content="Hostel fees depend on the applicable rate."
                )
            ],
            "situation_context": {
                "applies": False,
                "facts": {},
                "preferences": {},
                "constraints": {},
            },
        },
    ]

    from backend.multi_intent_answer import (
        build_multi_intent_answer_package,
    )

    state["multi_intent_answer_package"] = (
        build_multi_intent_answer_package(
            state["intent_results"]
        )
    )

    with patch(
        "backend.nodes.answer_chain"
    ) as chain:

        chain.invoke.return_value = SimpleNamespace(
            content="Three independent answers.",
        )

        result = generate_answer(
            state
        )

        assert chain.invoke.call_count == 1
        assert result["answer"]


def test_multi_intent_result_retains_request_identity():
    monkeypatch_pipeline = _supported_pipeline

    with patch(
        "backend.nodes._run_intent_evidence_pipeline",
        side_effect=monkeypatch_pipeline,
    ):
        result = process_multi_intent_node(
            {
                "intent_units": [
                    {
                        "question": "What are hostel fees?",
                        "topics": ["fees"],
                        "entities": ["hostel"],
                    },
                    {
                        "question": "Where is the library?",
                        "topics": ["navigation"],
                        "entities": ["library"],
                    },
                ],
                "is_multi_intent": True,
                "intent_count": 2,
            }
        )

    assert [
        item["question"]
        for item in result["intent_results"]
    ] == [
        "What are hostel fees?",
        "Where is the library?",
    ]
