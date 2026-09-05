"""
Evidence Group Tests

Validate that an initial anchor and its local context are treated
as one coherent evidence unit.
"""

from langchain_core.documents import Document

from backend.evidence_groups import (
    build_evidence_groups,
    attach_local_context,
    rank_evidence_groups,
    flatten_evidence_groups,
)


def test_each_anchor_creates_one_group():

    anchors = [
        Document(
            page_content="Anchor A",
            metadata={
                "source": "a.docx",
            },
        ),
        Document(
            page_content="Anchor B",
            metadata={
                "source": "b.docx",
            },
        ),
    ]

    groups = build_evidence_groups(
        anchors
    )

    assert len(groups) == 2

    assert groups[0].anchor.page_content == "Anchor A"
    assert groups[1].anchor.page_content == "Anchor B"


def test_local_context_stays_attached_to_anchor():

    anchors = [
        Document(
            page_content="Electrical engineering research",
            metadata={
                "source": "ee.docx",
            },
        )
    ]

    expanded = [
        anchors[0],
        Document(
            page_content="Power systems and smart grids",
            metadata={
                "source": "ee.docx",
            },
        ),
    ]

    groups = build_evidence_groups(
        anchors
    )

    groups = attach_local_context(
        groups,
        expanded,
    )

    assert len(groups) == 1

    assert len(
        groups[0].documents
    ) == 2


def test_different_source_is_not_attached():

    anchors = [
        Document(
            page_content="Electrical engineering research",
            metadata={
                "source": "ee.docx",
            },
        )
    ]

    expanded = [
        anchors[0],
        Document(
            page_content="Physics research",
            metadata={
                "source": "physics.docx",
            },
        ),
    ]

    groups = build_evidence_groups(
        anchors
    )

    groups = attach_local_context(
        groups,
        expanded,
    )

    assert len(
        groups[0].documents
    ) == 1


def test_flatten_preserves_anchor_before_context():

    anchor = Document(
        page_content="Anchor",
        metadata={
            "source": "a.docx",
        },
    )

    context = Document(
        page_content="Context",
        metadata={
            "source": "a.docx",
        },
    )

    groups = build_evidence_groups(
        [anchor]
    )

    groups[0].documents.append(
        context
    )

    flattened = flatten_evidence_groups(
        groups
    )

    assert (
        flattened[0].page_content
        == "Anchor"
    )

    assert (
        flattened[1].page_content
        == "Context"
    )


def test_group_ranking_returns_requested_number():

    anchors = [
        Document(
            page_content="Electrical Engineering research areas",
            metadata={
                "source": "ee.docx",
            },
        ),
        Document(
            page_content="Hostel accommodation facilities",
            metadata={
                "source": "hostel.docx",
            },
        ),
        Document(
            page_content="Chemistry research",
            metadata={
                "source": "chemistry.docx",
            },
        ),
    ]

    groups = build_evidence_groups(
        anchors
    )

    ranked = rank_evidence_groups(
        query=(
            "What research areas are available "
            "in Electrical Engineering?"
        ),
        groups=groups,
        top_k=2,
    )

    assert len(ranked) == 2

    assert (
        "Electrical"
        in ranked[0].anchor.page_content
    )