from __future__ import annotations

from dataclasses import dataclass

from ai_platform.core.retrieval.final_context_builder import build_final_context


@dataclass
class Doc:
    page_content: str
    metadata: dict


@dataclass
class Candidate:
    document: Doc
    document_id: str
    source: str


def c(cid: str, text: str) -> Candidate:
    return Candidate(Doc(text, {"source": f"source/{cid}.docx"}), cid, f"source/{cid}.docx")


def test_only_ranked_candidates_can_enter_context():
    result = build_final_context(
        "M.Tech eligibility requirements",
        (
            c("a", "M.Tech applicants must satisfy the eligibility requirements."),
            c("b", "Hostel accommodation is available on campus."),
        ),
    )
    assert result.ready
    assert {item.candidate_id for item in result.items} <= {"a", "b"}
    assert "M.Tech applicants" in result.context


def test_duplicate_units_are_removed():
    result = build_final_context(
        "M.Tech eligibility",
        (
            c("a", "M.Tech eligibility requires a relevant degree."),
            c("b", "M.Tech eligibility requires a relevant degree."),
        ),
    )
    assert result.selected_items == 1


def test_irrelevant_sentence_is_removed_but_candidate_is_not_reverified():
    result = build_final_context(
        "M.Tech eligibility",
        (
            c(
                "a",
                "M.Tech eligibility requires a relevant degree. Hostel rooms are available. Faculty members conduct research.",
            ),
        ),
    )
    assert "M.Tech eligibility" in result.context
    assert "Hostel rooms" not in result.context


def test_context_budget_is_hard():
    result = build_final_context(
        "M.Tech eligibility",
        (c("a", "M.Tech eligibility requires a relevant degree. " * 100),),
        max_context_chars=200,
    )
    assert result.context_chars <= 200


def test_ranked_order_breaks_ties():
    result = build_final_context(
        "M.Tech eligibility",
        (
            c("first", "Eligibility information for the M.Tech programme."),
            c("second", "Eligibility information for the M.Tech programme."),
        ),
    )
    assert result.items[0].candidate_id == "first"
