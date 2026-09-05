"""
Phase 4 — Adversarial Evidence Regression

Purpose
-------
Attack the evidence layer the way a real student would, including
questions that deliberately create semantic conflicts.

The goal is not to make every test pass immediately.

The goal is to expose:
- claim-scope confusion
- topic contamination
- route/mode mixing
- false quantitative support
- false requirement support
- incomplete multi-document support
- unsupported claims that should become "insufficient"
"""

from langchain_core.documents import Document

from backend.evidence import (
    assess_evidence_sufficiency,
)


def doc(text: str) -> Document:
    return Document(
        page_content=text
    )


# =========================================================
# 1. Admission vs financial assistance
# =========================================================

def test_financial_assistance_must_not_support_admission_claim():

    result = assess_evidence_sufficiency(
        query=(
            "Can someone with a four-year bachelor's degree "
            "apply for regular Ph.D. admission?"
        ),
        documents=[
            doc(
                "A candidate having a B.Tech / B.S. "
                "(four-year program) with GATE/NET(JRF)/NBHM "
                "can have financial assistance as per MHRD norms."
            )
        ],
    )

    assert result["status"] == "insufficient"


def test_direct_admission_evidence_supports_admission_claim():

    result = assess_evidence_sufficiency(
        query=(
            "Can someone with a four-year bachelor's degree "
            "apply for regular Ph.D. admission?"
        ),
        documents=[
            doc(
                "The applicant must have a bachelor's degree "
                "of minimum four-year duration in engineering "
                "or science or medicine or pharmacy or agricultural "
                "science or veterinary science or equivalent with "
                "at least 70% marks or at least 7.0/10 CPI or CGPA "
                "for GEN/OBC."
            )
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# 2. School-specific vs institute-wide
# =========================================================

def test_school_specific_eligibility_must_not_be_treated_as_institute_wide():

    result = assess_evidence_sufficiency(
        query=(
            "Can someone with a four-year bachelor's degree "
            "apply for regular Ph.D. admission?"
        ),
        documents=[
            doc(
                "Applicants to the School of AI and Data Science "
                "must have a master's degree in engineering, "
                "science, humanities, social sciences, or equivalent "
                "with at least 60% marks or 6.0/10 CGPA."
            )
        ],
    )

    assert result["status"] == "insufficient"


# =========================================================
# 3. Part-time vs regular admission
# =========================================================

def test_part_time_experience_rule_must_not_support_regular_admission():

    result = assess_evidence_sufficiency(
        query=(
            "What are the eligibility requirements "
            "for regular Ph.D. admission?"
        ),
        documents=[
            doc(
                "For sponsored, external, or part-time Ph.D. "
                "admission, the applicant must have a minimum "
                "of two years of work experience after the "
                "qualifying degree."
            )
        ],
    )

    assert result["status"] == "insufficient"


# =========================================================
# 4. Financial assistance question
# =========================================================

def test_financial_assistance_claim_accepts_financial_evidence():

    result = assess_evidence_sufficiency(
        query=(
            "Can a four-year B.Tech graduate receive "
            "Ph.D. financial assistance?"
        ),
        documents=[
            doc(
                "A candidate having a B.Tech / B.S. "
                "(four-year program) with GATE/NET(JRF)/NBHM "
                "(or equivalent) can have financial assistance "
                "as per applicable norms."
            )
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# 5. Quantitative false positive
# =========================================================

def test_numeric_text_without_fee_claim_does_not_support_fee_question():

    result = assess_evidence_sufficiency(
        query=(
            "What is the hostel fee?"
        ),
        documents=[
            doc(
                "The hostel has 80 square feet rooms and "
                "supports students throughout the academic year."
            )
        ],
    )

    assert result["status"] == "insufficient"


def test_hostel_fee_requires_explicit_charge_evidence():

    result = assess_evidence_sufficiency(
        query=(
            "What is the hostel fee?"
        ),
        documents=[
            doc(
                "Hostel room rent is ₹500 per day "
                "excluding GST."
            )
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# 6. Units / temporal quantity conflict
# =========================================================

def test_monthly_fee_evidence_supports_monthly_question():

    result = assess_evidence_sufficiency(
        query=(
            "What is the monthly hostel accommodation charge?"
        ),
        documents=[
            doc(
                "Long-term hostel accommodation costs "
                "₹2,175 per month excluding GST."
            )
        ],
    )

    assert result["status"] == "supported"


def test_daily_fee_question_is_not_supported_by_monthly_only_evidence():

    result = assess_evidence_sufficiency(
        query=(
            "What is the daily hostel accommodation charge?"
        ),
        documents=[
            doc(
                "Long-term hostel accommodation costs "
                "₹2,175 per month excluding GST."
            )
        ],
    )

    assert result["status"] == "insufficient"


# =========================================================
# 7. Temporal / latest questions
# =========================================================

def test_latest_future_fee_should_not_be_supported_by_old_dated_data():

    result = assess_evidence_sufficiency(
        query=(
            "What is the latest hostel fee for 2027?"
        ),
        documents=[
            doc(
                "AY 2026-2027 hostel charges are "
                "₹2,175 per month excluding GST."
            )
        ],
    )

    # Current evidence does not establish that this is the latest
    # available fee for 2027.
    assert result["status"] == "insufficient"


# =========================================================
# 8. Unsupported ranking / opinion question
# =========================================================

def test_best_researcher_question_requires_direct_ranking_evidence():

    result = assess_evidence_sufficiency(
        query=(
            "Which professor is the best researcher "
            "in Electrical Engineering?"
        ),
        documents=[
            doc(
                "Faculty members conduct research in power systems, "
                "VLSI, signal processing, robotics, and control systems."
            )
        ],
    )

    assert result["status"] == "insufficient"


# =========================================================
# 9. Hostel guarantee
# =========================================================

def test_hostel_facilities_do_not_prove_guaranteed_allocation():

    result = assess_evidence_sufficiency(
        query=(
            "Can you guarantee that I will get hostel accommodation?"
        ),
        documents=[
            doc(
                "Hostels provide furnished rooms, Wi-Fi, study rooms, "
                "gyms, and dining facilities."
            )
        ],
    )

    assert result["status"] == "insufficient"


def test_hostel_allocation_policy_supports_allocation_question():

    result = assess_evidence_sufficiency(
        query=(
            "Is hostel accommodation guaranteed?"
        ),
        documents=[
            doc(
                "Hostel allocation depends on eligibility, "
                "academic category, and availability of rooms."
            )
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# 10. Broad list aggregation
# =========================================================

def test_broad_program_list_can_aggregate_multiple_program_documents():

    result = assess_evidence_sufficiency(
        query=(
            "What programs are offered at the institute?"
        ),
        documents=[
            doc(
                "The institute offers undergraduate "
                "B.Tech programs."
            ),
            doc(
                "The institute offers M.Tech programs "
                "in multiple specializations."
            ),
            doc(
                "The institute offers Ph.D. programs "
                "across multiple disciplines."
            ),
        ],
    )

    assert result["status"] == "supported"


# =========================================================
# 11. Broad list must not be satisfied by unrelated evidence
# =========================================================

def test_broad_program_question_rejects_unrelated_institutional_documents():

    result = assess_evidence_sufficiency(
        query=(
            "What programs are offered at the institute?"
        ),
        documents=[
            doc(
                "The institute has hostels with Wi-Fi, gyms, "
                "study rooms, and dining facilities."
            ),
            doc(
                "The institute has a health centre and "
                "24-hour security."
            ),
        ],
    )

    assert result["status"] == "insufficient"


# =========================================================
# 12. Multi-topic contamination
# =========================================================

def test_phd_plus_hostel_question_requires_both_claim_domains():

    result = assess_evidence_sufficiency(
        query=(
            "What are the Ph.D. eligibility requirements "
            "and what are the hostel fees?"
        ),
        documents=[
            doc(
                "Regular Ph.D. applicants need a master's degree "
                "with the specified academic percentage."
            ),
        ],
    )

    assert result["status"] == "insufficient"


def test_phd_plus_hostel_question_supported_by_both_domains():

    result = assess_evidence_sufficiency(
        query=(
            "What are the Ph.D. eligibility requirements "
            "and what are the hostel fees?"
        ),
        documents=[
            doc(
                "Regular Ph.D. applicants need a master's degree "
                "with the specified academic percentage."
            ),
            doc(
                "Hostel room rent is ₹500 per day "
                "excluding GST."
            ),
        ],
    )

    assert result["status"] == "supported"
