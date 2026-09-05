"""
Phase 4 — Multi-Intent Answer Composition Tests
"""

from langchain_core.documents import Document

from backend.multi_intent_answer import (
    build_multi_intent_answer_package,
    build_multi_intent_context,
    build_multi_intent_instruction,
    classify_intents,
)


def doc(
    text: str,
):
    return Document(
        page_content=text
    )


def test_supported_intent_keeps_its_evidence():

    result = classify_intents(
        [
            {
                "question": "What are the hostel fees?",
                "evidence_status": "supported",
                "evidence_coverage_status": "supported",
                "compressed_docs": [
                    doc(
                        "Hostel room rent is ₹300 per day."
                    )
                ],
            }
        ]
    )

    assert result[0]["status"] == "supported"

    assert len(
        result[0]["documents"]
    ) == 1


def test_insufficient_intent_has_no_factual_evidence():

    result = build_multi_intent_context(
        [
            {
                "question": "What is the professor salary?",
                "evidence_status": "insufficient",
                "evidence_coverage_status": "insufficient",
                "compressed_docs": [
                    doc(
                        "This document mentions professors."
                    )
                ],
            }
        ]
    )

    assert (
        "Evidence: NONE"
        in result
    )

    assert (
        "professors"
        not in result.lower()
    )


def test_supported_intent_is_preserved_when_other_is_insufficient():

    result = build_multi_intent_answer_package(
        [
            {
                "question": "What are the hostel fees?",
                "evidence_status": "supported",
                "evidence_coverage_status": "supported",
                "compressed_docs": [
                    doc(
                        "Hostel room rent is ₹300 per day."
                    )
                ],
            },
            {
                "question": "What is the professor salary?",
                "evidence_status": "insufficient",
                "evidence_coverage_status": "insufficient",
                "compressed_docs": [],
            },
        ]
    )

    assert (
        result["overall_status"]
        == "partially_supported"
    )

    assert (
        result["supported_intents"]
        == 1
    )

    assert (
        result["insufficient_intents"]
        == 1
    )

    context = result["context"]

    assert "₹300" in context
    assert "Evidence: NONE" in context


def test_insufficient_intent_does_not_leak_unrelated_documents():

    result = build_multi_intent_context(
        [
            {
                "question": "What is the professor salary?",
                "evidence_status": "insufficient",
                "evidence_coverage_status": "insufficient",
                "compressed_docs": [
                    doc(
                        "Hostel room rent is ₹300 per day."
                    )
                ],
            }
        ]
    )

    assert "₹300" not in result


def test_partial_intent_explicitly_marks_partial_scope():

    result = build_multi_intent_context(
        [
            {
                "question": "What programs are offered?",
                "evidence_status": "supported",
                "evidence_coverage_status": "partially_supported",
                "compressed_docs": [
                    doc(
                        "B.Tech. programs include Electrical Engineering."
                    )
                ],
            }
        ]
    )

    assert "PARTIAL" in result


def test_instruction_forbids_cross_intent_merging():

    result = build_multi_intent_instruction(
        [
            {
                "question": "What are the Ph.D. requirements?",
                "evidence_status": "supported",
                "evidence_coverage_status": "supported",
                "compressed_docs": [
                    doc(
                        "Minimum qualification is a master's degree."
                    )
                ],
            },
            {
                "question": "What are the hostel fees?",
                "evidence_status": "insufficient",
                "evidence_coverage_status": "insufficient",
                "compressed_docs": [],
            },
        ]
    )

    assert "independently" in result.lower()
    assert "never merge" in result.lower()
    assert "insufficient" in result.lower()


def test_all_insufficient_package_is_insufficient():

    result = build_multi_intent_answer_package(
        [
            {
                "question": "What is salary?",
                "evidence_status": "insufficient",
                "evidence_coverage_status": "insufficient",
                "compressed_docs": [],
            },
            {
                "question": "What is the fee for 2035?",
                "evidence_status": "insufficient",
                "evidence_coverage_status": "insufficient",
                "compressed_docs": [],
            },
        ]
    )

    assert (
        result["overall_status"]
        == "insufficient"
    )

    assert (
        result["insufficient_intents"]
        == 2
    )


def test_all_supported_package_is_supported():

    result = build_multi_intent_answer_package(
        [
            {
                "question": "What are the hostel fees?",
                "evidence_status": "supported",
                "evidence_coverage_status": "supported",
                "compressed_docs": [
                    doc(
                        "Hostel room rent is ₹300 per day."
                    )
                ],
            },
            {
                "question": "What are the Ph.D. requirements?",
                "evidence_status": "supported",
                "evidence_coverage_status": "supported",
                "compressed_docs": [
                    doc(
                        "A master's degree is required."
                    )
                ],
            },
        ]
    )

    assert (
        result["overall_status"]
        == "supported"
    )

    assert (
        result["supported_intents"]
        == 2
    )


def test_empty_package_is_insufficient():

    result = build_multi_intent_answer_package(
        []
    )

    assert (
        result["overall_status"]
        == "insufficient"
    )

    assert (
        result["intent_count"]
        == 0
    )
