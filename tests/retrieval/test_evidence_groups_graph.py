"""
Evidence Group Graph Integration

Validate the complete grouped-evidence retrieval pipeline.

The important invariant is the number of final evidence groups,
not the flattened number of Documents inside those groups.
"""

from backend.graph import create_graph


QUESTIONS = [
    "What research areas are available in Electrical Engineering?",
    "What are the eligibility requirements for regular Ph.D. admission?",
    "What are the eligibility requirements for regular M.Tech. admission?",
    "What are the hostel fees?",
]


def test_graph_produces_final_evidence_after_grouping():

    graph = create_graph()

    for question in QUESTIONS:

        result = graph.invoke(
            {
                "question": question,
                "chat_history": [],
            }
        )

        assert result.get(
            "answer"
        )

        assert result.get(
            "reranked_docs"
        )

        assert result.get(
            "evidence_groups"
        )

        # -------------------------------------------------
        # The production limit is on evidence groups.
        # A group may legitimately contain multiple Documents
        # because local context remains attached to its anchor.
        # -------------------------------------------------

        assert len(
            result["evidence_groups"]
        ) <= 5

        # Flattened evidence can be larger than five because
        # each selected group may contain local context.
        assert len(
            result["reranked_docs"]
        ) >= len(
            result["evidence_groups"]
        )

        assert result.get(
            "compressed_docs"
        )

        # Final context should equal the selected grouped
        # evidence rather than arbitrarily slicing it back to five.
        assert len(
            result["compressed_docs"]
        ) == len(
            result["reranked_docs"]
        )