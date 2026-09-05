"""
IITJ V1 — Phase 8B Graph Integration Tests
"""

from langchain_core.documents import Document

from backend import nodes


def make_doc(
    text: str,
    source: str,
) -> Document:

    return Document(
        page_content=text,
        metadata={
            "source": source,
        },
    )


def test_broad_question_invokes_8b_candidate_assembly(
    monkeypatch,
):

    calls = []

    broad_doc = make_doc(
        "Academic programs overview.",
        "/programs/programs.docx",
    )

    monkeypatch.setattr(
        nodes,
        "reciprocal_rank_fusion",
        lambda results: [broad_doc],
    )

    monkeypatch.setattr(
        nodes,
        "deduplicate_documents",
        lambda documents: list(documents),
    )

    monkeypatch.setattr(
        nodes,
        "is_broad_institutional_question",
        lambda question: True,
    )

    def fake_assemble(
        query,
        documents,
    ):
        calls.append(
            (
                query,
                documents,
            )
        )
        return documents

    monkeypatch.setattr(
        nodes,
        "assemble_broad_candidates",
        fake_assemble,
    )

    result = nodes.fuse_retrieved_documents(
        {
            "question": (
                "What academic programs are available?"
            ),
            "resolved_question": (
                "What academic programs are available?"
            ),
            "retrieval_results": [
                [],
                [],
            ],
        }
    )

    assert calls

    assert (
        calls[0][0]
        == "What academic programs are available?"
    )

    assert result["fused_docs"] == [
        broad_doc
    ]


def test_focused_question_bypasses_8b(
    monkeypatch,
):

    document = make_doc(
        "M.Tech admission eligibility.",
        "/admissions/mtech.docx",
    )

    calls = []

    monkeypatch.setattr(
        nodes,
        "reciprocal_rank_fusion",
        lambda results: [document],
    )

    monkeypatch.setattr(
        nodes,
        "deduplicate_documents",
        lambda documents: list(documents),
    )

    monkeypatch.setattr(
        nodes,
        "is_broad_institutional_question",
        lambda question: False,
    )

    def fail_assemble(
        query,
        documents,
    ):
        calls.append(
            (
                query,
                documents,
            )
        )
        raise AssertionError(
            "8B must not run for focused questions."
        )

    monkeypatch.setattr(
        nodes,
        "assemble_broad_candidates",
        fail_assemble,
    )

    result = nodes.fuse_retrieved_documents(
        {
            "question": (
                "What is the M.Tech eligibility?"
            ),
            "resolved_question": (
                "What is the M.Tech eligibility?"
            ),
            "retrieval_results": [
                [],
                [],
            ],
        }
    )

    assert not calls

    assert result["fused_docs"] == [
        document
    ]