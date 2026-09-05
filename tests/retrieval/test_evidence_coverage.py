from langchain_core.documents import Document

from backend.evidence_coverage import (
    assess_evidence_coverage,
)


# =========================================================
# Descriptive
# =========================================================

def test_descriptive_question_with_strong_evidence_is_supported():

    result = assess_evidence_coverage(
        query=(
            "What is the M.Tech. Drone and Anti-Drone "
            "Technologies program about?"
        ),
        documents=[
            Document(
                page_content=(
                    "The M.Tech. program in Drone and "
                    "Anti-Drone Technologies aims to equip "
                    "students with skills and knowledge "
                    "for detection, tracking, identification, "
                    "and neutralization systems."
                    + " " * 450
                ),
                metadata={
                    "source": "aerospace_general.docx"
                },
            )
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# List
# =========================================================

def test_list_question_can_be_partially_supported():

    result = assess_evidence_coverage(
        query=(
            "What research areas are available in "
            "Electrical Engineering?"
        ),
        documents=[
            Document(
                page_content=(
                    "Research areas include VLSI, "
                    "signal integrity, and neuromorphic "
                    "computing."
                ),
                metadata={
                    "source": "electrical_research.docx"
                },
            )
        ],
    )

    assert result["status"] == "partially_supported"


# =========================================================
# Requirements
# =========================================================

def test_requirement_question_with_strong_evidence_is_supported():

    result = assess_evidence_coverage(
        query=(
            "What are the eligibility requirements "
            "for regular Ph.D. admission?"
        ),
        documents=[
            Document(
                page_content=(
                    "Candidates must have a master's degree "
                    "in engineering, pharmacy, agricultural "
                    "science, science, humanities, social "
                    "sciences, or management with at least "
                    "60% marks or 6.0 CGPA. Candidates may "
                    "also qualify through the four-year "
                    "bachelor's degree route."
                ),
                metadata={
                    "source": "phd_admissions.docx"
                },
            )
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# Empty
# =========================================================

def test_empty_evidence_is_insufficient():

    result = assess_evidence_coverage(
        query="What are the hostel fees?",
        documents=[],
    )

    assert result["status"] == "insufficient"


# =========================================================
# Question type
# =========================================================

def test_question_type_detection_is_recorded():

    result = assess_evidence_coverage(
        query="What are the hostel fees?",
        documents=[],
    )

    assert result["question_type"] == "quantitative"