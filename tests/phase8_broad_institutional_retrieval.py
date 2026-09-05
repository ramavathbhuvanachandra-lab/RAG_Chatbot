"""
IITJ V1 — Phase 8B Broad Institutional Retrieval Diagnostic

Purpose
-------
Diagnose how broad institutional questions travel through the existing
graph without changing production behavior.

Focus:
    - broad programs
    - broad departments
    - broad research
    - broad facilities
    - broad admissions

This intentionally uses the existing graph so we inspect the real
production path rather than a synthetic retrieval function.
"""

from __future__ import annotations

from backend.graph import create_graph


QUESTIONS = [
    "What academic programs are available at IIT Jodhpur?",
    "What departments are there at IIT Jodhpur?",
    "What research opportunities are available at IIT Jodhpur?",
    "What facilities are available at IIT Jodhpur?",
    "What admission opportunities are available at IIT Jodhpur?",
]


def text(value) -> str:
    return str(value or "")


def show_value(
    label: str,
    value,
    limit: int = 5000,
):
    print("\n" + "-" * 100)
    print(label)
    print("-" * 100)

    rendered = text(value)

    if len(rendered) > limit:
        rendered = (
            rendered[:limit]
            + "\n...[TRUNCATED]..."
        )

    print(rendered)


def run_question(
    graph,
    question: str,
):
    print("\n" + "=" * 110)
    print("QUESTION")
    print("=" * 110)

    print(question)

    result = graph.invoke(
        {
            "question": question,
            "chat_history": [],
        }
    )

    print(
        "\nRESULT KEYS:"
    )

    for key in sorted(
        result.keys()
    ):
        print(
            f"  - {key}"
        )

    # ---------------------------------------------------------
    # Query / planning visibility
    # ---------------------------------------------------------

    for key in (
        "resolved_question",
        "retrieval_queries",
        "generated_queries",
        "intent_questions",
        "intent_units",
        "conversation_mode",
        "active_topic",
        "active_entity",
    ):

        if key in result:

            show_value(
                key.upper(),
                result[key],
                limit=6000,
            )

    # ---------------------------------------------------------
    # Retrieval / evidence visibility
    # ---------------------------------------------------------

    for key in (
        "retrieval_results",
        "fused_docs",
        "initial_reranked_docs",
        "expanded_docs",
        "reranked_docs",
        "compressed_docs",
        "evidence_status",
        "evidence_score",
        "evidence_coverage_status",
        "evidence_question_type",
    ):

        if key in result:

            value = result[key]

            if key.endswith("_docs"):

                print(
                    "\n"
                    + "-" * 100
                )

                print(
                    key.upper()
                )

                print(
                    "-" * 100
                )

                try:

                    print(
                        "COUNT:",
                        len(value),
                    )

                    for index, document in enumerate(
                        value,
                        start=1,
                    ):

                        metadata = getattr(
                            document,
                            "metadata",
                            {},
                        )

                        source = (
                            metadata.get(
                                "source",
                                "",
                            )
                            if isinstance(
                                metadata,
                                dict,
                            )
                            else ""
                        )

                        content = text(
                            getattr(
                                document,
                                "page_content",
                                document,
                            )
                        )

                        print(
                            f"\n[{index}] SOURCE:"
                            f" {source}"
                        )

                        print(
                            content[:1200]
                        )

                except Exception as exc:

                    print(
                        "Unable to inspect:",
                        exc,
                    )

            else:

                show_value(
                    key.upper(),
                    value,
                    limit=6000,
                )

    # ---------------------------------------------------------
    # Final answer
    # ---------------------------------------------------------

    answer = result.get(
        "answer",
        "",
    )

    context = result.get(
        "context",
        "",
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 110
    )

    print(
        answer
    )

    print(
        "\nFINAL CONTEXT CHARACTERS:",
        len(text(context)),
    )


def main():

    print(
        "=" * 110
    )

    print(
        "IITJ V1 — PHASE 8B "
        "BROAD INSTITUTIONAL RETRIEVAL DIAGNOSTIC"
    )

    print(
        "=" * 110
    )

    graph = create_graph()

    for question in QUESTIONS:

        run_question(
            graph,
            question,
        )


if __name__ == "__main__":
    main()