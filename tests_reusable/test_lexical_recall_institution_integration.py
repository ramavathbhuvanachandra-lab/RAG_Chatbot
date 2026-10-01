"""Hard integration tests for generic lexical engine + institution adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.core.retrieval import lexical_recall


@dataclass(frozen=True)
class Doc:
    page_content: str
    metadata: dict


def test_active_institution_adapter_is_loaded() -> None:
    institution_id, adapter = lexical_recall._load_institution_adapter()
    assert institution_id == "iitj"
    assert adapter is not None
    assert callable(getattr(adapter, "score_pair"))
    assert callable(getattr(adapter, "expand_query"))


def test_iitj_alias_enrichment_is_visible() -> None:
    document = Doc(
        "Master of Technology eligibility requirements are listed here.",
        {"source": "admissions/mtech_admissions.docx"},
    )
    hit = lexical_recall.score_document(
        "What are the M.Tech eligibility requirements?",
        document,
    )

    assert hit.institution_id == "iitj"
    assert hit.institution_score > 0.0
    assert "mtech" in hit.institution_terms
    assert hit.retrieval_queries[0].startswith("What are the M.Tech")
    assert hit.score > 0.0


def test_institution_phrases_are_a_bounded_boost_not_a_verifier() -> None:
    relevant = Doc(
        "M.Tech eligibility requirements are listed in the admission document.",
        {"source": "admissions/mtech_admissions.docx"},
    )
    distractor = Doc(
        "The admission processor receives requests and sends them for review.",
        {"source": "administration/other.docx"},
    )

    relevant_hit = lexical_recall.score_document(
        "M.Tech eligibility requirements",
        relevant,
    )
    distractor_hit = lexical_recall.score_document(
        "admission process",
        distractor,
    )

    assert relevant_hit.institution_phrases
    assert "admission process" not in distractor_hit.institution_phrases


def test_larger_words_do_not_get_program_alias_rescue() -> None:
    document = Doc(
        "M.Technique is not the name of a degree program here.",
        {"source": "noise.docx"},
    )
    hit = lexical_recall.score_document(
        "M.Tech eligibility",
        document,
    )

    assert hit.institution_score == 0.0
    assert hit.score < 0.30


def test_generic_engine_still_works_without_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lexical_recall, "_load_institution_adapter", lambda: ("", None))

    document = Doc(
        "The admission requirements and fee structure are published online.",
        {"source": "fees/general.docx"},
    )
    hit = lexical_recall.score_document(
        "admission requirements fee",
        document,
    )

    assert hit.institution_id == ""
    assert hit.institution_score == 0.0
    assert hit.score > 0.0


def test_duplicate_documents_are_removed_after_institution_scoring() -> None:
    documents = [
        Doc(
            "M.Tech eligibility requirements are listed here.",
            {"source": "admissions/mtech.docx"},
        ),
        Doc(
            "M.Tech eligibility requirements are listed here.",
            {"source": "admissions/mtech.docx"},
        ),
        Doc(
            "The hostel provides accommodation.",
            {"source": "hostel/general.docx"},
        ),
    ]

    hits = lexical_recall.retrieve_lexical(
        "M.Tech eligibility requirements",
        documents,
        limit=10,
        min_score=0.0,
    )

    identities = [
        (hit.document.metadata.get("source"), hit.document.page_content)
        for hit in hits
    ]
    assert len(identities) == len(set(identities))


def test_results_are_deterministic() -> None:
    documents = [
        Doc("M.Tech admission requirements", {"source": "a.docx"}),
        Doc("Hostel accommodation", {"source": "b.docx"}),
        Doc("M.Sc admission route", {"source": "c.docx"}),
    ]

    first = lexical_recall.retrieve_lexical(
        "M.Tech admission requirements",
        documents,
        limit=10,
        min_score=0.0,
    )
    second = lexical_recall.retrieve_lexical(
        "M.Tech admission requirements",
        documents,
        limit=10,
        min_score=0.0,
    )

    assert [h.document.metadata["source"] for h in first] == [
        h.document.metadata["source"] for h in second
    ]
    assert [h.score for h in first] == [h.score for h in second]


def test_scores_remain_bounded() -> None:
    documents = [
        Doc("M.Tech M.Tech M.Tech admission requirements", {"source": "a.docx"}),
        Doc("completely unrelated content", {"source": "b.docx"}),
    ]

    for hit in lexical_recall.retrieve_lexical(
        "M.Tech admission requirements",
        documents,
        limit=10,
        min_score=0.0,
    ):
        assert 0.0 <= hit.score <= 1.0
        assert 0.0 <= hit.institution_score <= 1.0
