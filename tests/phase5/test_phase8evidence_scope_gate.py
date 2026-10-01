"""
IIT Jodhpur V1 — Phase 8E
Final evidence-scope regression tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from backend.evidence_scope_gate import (
    select_answer_evidence,
)


def _doc(source: str, text: str):
    return SimpleNamespace(
        page_content=text,
        metadata={"source": source},
    )


def _programs(text: str):
    value = str(text or "").casefold()
    result = []

    if "m.sc" in value or "msc" in value:
        result.append("msc")

    if "m.tech" in value or "mtech" in value:
        result.append("mtech")

    if "ph.d" in value or "phd" in value:
        result.append("phd")

    return result


def _entities(text: str):
    return []


def _patches():
    return (
        patch(
            "backend.evidence_scope_gate.detect_programs",
            side_effect=_programs,
        ),
        patch(
            "backend.evidence_scope_gate.detect_entities",
            side_effect=_entities,
        ),
    )


def test_robotics_broad_query_recovers_lower_ranked_robotics_evidence():
    docs = [
        _doc(
            "/research/research.docx",
            "IIT Jodhpur current research areas.",
        ),
        _doc(
            "/departments/chemistry/research.docx",
            "Research areas include inorganic, organic and physical chemistry.",
        ),
        _doc(
            "/departments/electrical_engineering/research.docx",
            "Electrical Engineering research themes include robotics and adaptive control.",
        ),
        _doc(
            "/departments/electrical_engineering/overview.docx",
            "The department's research activities span embedded systems, robotics and signal processing.",
        ),
        _doc(
            "/departments/economics/research.docx",
            "Research areas in economics include econometrics and human capital.",
        ),
        _doc(
            "/schools/artificial_intelligence_and_data_science/research.docx",
            "Research project positions and AI research opportunities.",
        ),
    ]

    result = select_answer_evidence(
        query="What research areas are related to robotics at IIT Jodhpur?",
        documents=docs,
        max_documents=5,
    )

    sources = [
        doc.metadata["source"]
        for doc in result
    ]

    assert (
        "/departments/electrical_engineering/research.docx"
        in sources
    )
    assert (
        "/departments/electrical_engineering/overview.docx"
        in sources
    )


def test_program_specific_admission_source_beats_generic_source():
    generic = _doc(
        "/admissions/general_admissions.docx",
        "General M.Sc. admissions.",
    )
    scoped = _doc(
        "/admissions/msc_admissions.docx",
        "M.Sc. admission routes.",
    )

    p1, p2 = _patches()
    with p1, p2:
        result = select_answer_evidence(
            query="m.sc admission routes",
            documents=[generic, scoped],
        )

    assert result == [scoped]


def test_program_overview_is_not_used_as_admission_evidence():
    generic = _doc(
        "/admissions/general_admissions.docx",
        "General admission information.",
    )
    overview = _doc(
        "/programs/msc/programs.docx",
        "M.Sc. program overview.",
    )

    p1, p2 = _patches()
    with p1, p2:
        result = select_answer_evidence(
            query="m.sc admission routes",
            documents=[generic, overview],
        )

    assert result == [generic, overview]


def test_mtech_admission_prefers_mtech_admission_source():
    generic = _doc(
        "/admissions/general_admissions.docx",
        "General M.Tech admission.",
    )
    scoped = _doc(
        "/admissions/mtech_admissions.docx",
        "M.Tech admission eligibility.",
    )

    p1, p2 = _patches()
    with p1, p2:
        result = select_answer_evidence(
            query="m.tech admission requirements",
            documents=[generic, scoped],
        )

    assert result == [scoped]


def test_phd_admission_prefers_phd_admission_source():
    generic = _doc(
        "/admissions/general_admissions.docx",
        "General Ph.D. admission.",
    )
    scoped = _doc(
        "/admissions/phd_admissions.docx",
        "Ph.D. admission procedure.",
    )

    p1, p2 = _patches()
    with p1, p2:
        result = select_answer_evidence(
            query="phd admission requirements",
            documents=[generic, scoped],
        )

    assert result == [scoped]


def test_broad_admission_query_is_not_program_filtered():
    docs = [
        _doc(
            "/admissions/general_admissions.docx",
            "General admissions.",
        ),
        _doc(
            "/admissions/mtech_admissions.docx",
            "M.Tech admissions.",
        ),
        _doc(
            "/admissions/phd_admissions.docx",
            "Ph.D. admissions.",
        ),
    ]

    p1, p2 = _patches()
    with p1, p2:
        result = select_answer_evidence(
            query="What admission opportunities are available?",
            documents=docs,
        )

    assert result == docs


def test_broad_facilities_query_is_unchanged():
    docs = [
        _doc(
            "/facilities/facilities.docx",
            "Library and laboratories.",
        ),
        _doc(
            "/research/facilities.docx",
            "Research facilities.",
        ),
    ]

    p1, p2 = _patches()
    with p1, p2:
        result = select_answer_evidence(
            query="What facilities are available?",
            documents=docs,
        )

    assert result == docs


def test_no_matching_scope_falls_back_safely():
    docs = [
        _doc(
            "/admissions/general_admissions.docx",
            "General admission information.",
        ),
        _doc(
            "/programs/msc/programs.docx",
            "M.Sc. program overview.",
        ),
    ]

    p1, p2 = _patches()
    with p1, p2:
        result = select_answer_evidence(
            query="m.sc admission routes",
            documents=docs,
        )

    assert result == docs


def test_real_life_msc_questions_stay_on_admission_scope():
    questions = [
        "What are the admission routes for M.Sc.?",
        "How can I get into the MSc program?",
        "I want to apply for the M.Sc. — what are my options?",
        "How do I get admission into MSc?",
    ]

    generic = _doc(
        "/admissions/general_admissions.docx",
        "General M.Sc. admission information.",
    )

    scoped = _doc(
        "/admissions/msc_admissions.docx",
        "M.Sc. admission routes.",
    )

    for question in questions:
        p1, p2 = _patches()

        with p1, p2:
            result = select_answer_evidence(
                query=question,
                documents=[generic, scoped],
            )

        assert result == [scoped]


def test_real_life_mtech_questions_stay_on_admission_scope():
    questions = [
        "What is needed for M.Tech admission?",
        "Can I apply for MTech after my degree?",
        "How do I get into the M.Tech program?",
    ]

    generic = _doc(
        "/admissions/general_admissions.docx",
        "General M.Tech admission information.",
    )

    scoped = _doc(
        "/admissions/mtech_admissions.docx",
        "M.Tech admission eligibility.",
    )

    for question in questions:
        p1, p2 = _patches()

        with p1, p2:
            result = select_answer_evidence(
                query=question,
                documents=[generic, scoped],
            )

        assert result == [scoped]


def test_real_life_phd_questions_stay_on_admission_scope():
    questions = [
        "What are the Ph.D. admission requirements?",
        "I did B.Tech and want to apply for a PhD. What do I need?",
        "How do I get admission into the PhD program?",
    ]

    generic = _doc(
        "/admissions/general_admissions.docx",
        "General postgraduate admission information.",
    )

    scoped = _doc(
        "/admissions/phd_admissions.docx",
        "Ph.D. admission procedure.",
    )

    for question in questions:
        p1, p2 = _patches()

        with p1, p2:
            result = select_answer_evidence(
                query=question,
                documents=[generic, scoped],
            )

        assert result == [scoped]


def test_content_can_establish_scope_for_non_program_query():
    docs = [
        _doc(
            "/hostel/general.docx",
            "Hostel accommodation and rooms.",
        ),
        _doc(
            "/research/facilities.docx",
            "Hostel Wi-Fi and laundry facilities are available.",
        ),
    ]

    with patch(
        "backend.evidence_scope_gate.detect_entities",
        side_effect=lambda text: (
            {"hostel"}
            if "hostel" in text.lower()
            else set()
        ),
    ):
        result = select_answer_evidence(
            query="hostel Wi-Fi laundry",
            documents=docs,
            max_documents=5,
        )

    assert result[0].metadata["source"] == (
        "/research/facilities.docx"
    )


def test_max_document_limit_is_respected():
    docs = [
        _doc(
            "/admissions/msc_admissions_1.docx",
            "M.Sc. admission routes.",
        ),
        _doc(
            "/admissions/msc_admissions_2.docx",
            "M.Sc. admission requirements.",
        ),
        _doc(
            "/admissions/msc_admissions_3.docx",
            "M.Sc. application.",
        ),
    ]

    p1, p2 = _patches()

    with p1, p2:
        result = select_answer_evidence(
            query="m.sc admission",
            documents=docs,
            max_documents=2,
        )

    assert result == docs[:2]