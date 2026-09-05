"""
IITJ V1 — Phase 8 Final Retrieval Core Contract Tests

This file tests retrieval primitives and graph compatibility.
It intentionally avoids using LangChain Document objects as set members because
Document is unhashable.
"""

from langchain_core.documents import Document

from backend import retriever
from backend import nodes


def make_doc(text: str, source: str = "source.docx") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source},
    )


def test_rrf_supports_weighted_multi_query_lists():

    d1 = make_doc(
        "robotics control",
        "a.docx",
    )
    d2 = make_doc(
        "robotics research",
        "b.docx",
    )
    d3 = make_doc(
        "control systems",
        "c.docx",
    )

    result = retriever.reciprocal_rank_fusion(
        [
            [d1, d2],
            [d3, d2],
            [d2, d1],
        ],
        weights=[
            0.70,
            0.30,
            0.45,
        ],
    )

    assert result

    # Document is unhashable; compare its stable content instead.
    assert result[0].page_content in (
        d1.page_content,
        d2.page_content,
        d3.page_content,
    )


def test_rrf_rejects_mismatched_weight_count():

    document = make_doc("test")

    try:
        retriever.reciprocal_rank_fusion(
            [
                [document],
                [document],
            ],
            weights=[1.0],
        )
    except ValueError:
        return

    raise AssertionError(
        "RRF must reject mismatched list/weight counts."
    )


def test_variant_reranking_accepts_query_variants():

    strong = make_doc(
        "robotics adaptive control",
        "ee-research.docx",
    )

    weak = make_doc(
        "generic research overview",
        "research.docx",
    )

    result = retriever.rerank_documents(
        query="robotics adaptive control",
        query_variants=[
            "general research topics",
        ],
        documents=[
            weak,
            strong,
        ],
        top_k=2,
    )

    assert result

    assert result[0].page_content == (
        "robotics adaptive control"
    )


def test_primary_query_remains_authoritative_over_noisy_alternate():

    strong = make_doc(
        "robotics adaptive control",
        "ee-research.docx",
    )

    unrelated = make_doc(
        "hostel admission fees",
        "hostel.docx",
    )

    result = retriever.rerank_documents(
        query="robotics adaptive control",
        query_variants=[
            "hostel admission fees",
        ],
        documents=[
            unrelated,
            strong,
        ],
        top_k=2,
    )

    assert result

    assert result[0].page_content == (
        "robotics adaptive control"
    )


def test_context_sanitization_removes_known_ingestion_wrappers():

    text = (
        "Useful research content.\n\n"
        "Command 6 • Retrieval Representation •\n\n"
        "More useful content.\n\n"
        "Source Original source URLs preserved from the "
        "Command 5 knowledge source."
    )

    cleaned = retriever.sanitize_context_text(
        text
    )

    lowered = cleaned.lower()

    assert (
        "retrieval representation"
        not in lowered
    )

    assert (
        "command 6"
        not in lowered
    )

    assert (
        "source original source urls preserved"
        not in lowered
    )

    assert (
        "useful research content."
        in lowered
    )

    assert (
        "more useful content."
        in lowered
    )


def test_context_sanitization_preserves_real_facts():

    text = (
        "The research topic is robotics.\n"
        "Command 6 • Retrieval Representation •\n"
        "Adaptive control & robotics is a listed research area."
    )

    cleaned = retriever.sanitize_context_text(
        text
    )

    lowered = cleaned.lower()

    assert "robotics" in lowered

    assert (
        "adaptive control & robotics"
        in lowered
    )


def test_format_context_uses_neutral_evidence_labels():

    context = retriever.format_context(
        [
            make_doc("Robotics research."),
            make_doc("Control systems research."),
        ]
    )

    assert "Evidence 1" in context
    assert "Evidence 2" in context

    assert "Document 1" not in context
    assert "Document 2" not in context


def test_hybrid_retrieval_uses_bounded_queries(
    monkeypatch,
):

    calls = []

    def fake_dense(query):
        calls.append(("dense", query))
        return []

    def fake_bm25(query):
        calls.append(("bm25", query))
        return []

    monkeypatch.setattr(
        nodes,
        "dense_retrieve",
        fake_dense,
    )

    monkeypatch.setattr(
        nodes,
        "keyword_retrieve",
        fake_bm25,
    )

    state = {
        "question": "original question",
        "resolved_question": "resolved question",
        "generated_queries": [
            "resolved question",
            "alternate one",
            "alternate two",
            "alternate three should be dropped",
        ],
    }

    result = nodes.hybrid_retrieve(
        state
    )

    assert result["retrieval_queries"] == [
        "resolved question",
        "alternate one",
        "alternate two",
    ]

    assert len(calls) == 6

    assert len(
        result["retrieval_results"]
    ) == 6

    assert len(
        result["retrieval_weights"]
    ) == 6


def test_retrieval_query_bound_is_three():

    assert (
        retriever.MAX_RETRIEVAL_QUERIES
        == 3
    )
