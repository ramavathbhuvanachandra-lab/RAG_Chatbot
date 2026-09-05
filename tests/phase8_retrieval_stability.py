"""
Phase 8 — Retrieval Stability Test

Runs the same retrieval questions repeatedly to detect intermittent
zero-context / early-exit behavior.
"""

from backend.graph import create_graph


QUESTIONS = [
    "What research areas are related to control systems in Electrical Engineering?",
    "What research areas are available in Electrical Engineering?",
    "What research areas are related to robotics at IIT Jodhpur?",
]


def main():

    graph = create_graph()

    for question in QUESTIONS:

        print("\n" + "=" * 100)
        print("QUESTION:")
        print(question)
        print("=" * 100)

        for run in range(1, 6):

            result = graph.invoke(
                {
                    "question": question,
                    "chat_history": [],
                }
            )

            answer = str(
                result.get("answer", "")
            )

            context = str(
                result.get("context", "")
            )

            print(
                f"\nRUN {run}"
            )

            print(
                f"answer_chars={len(answer)} "
                f"context_chars={len(context)}"
            )

            print(
                "answer:",
                answer[:300].replace("\n", " "),
            )


if __name__ == "__main__":
    main()