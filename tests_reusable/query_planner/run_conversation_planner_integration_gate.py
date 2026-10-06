"""Real conversation -> planner integration gate.

This intentionally stops before retrieval/answer generation. It proves the
boundary between conversation resolution and the frozen Query Planner V2.3.
"""
from __future__ import annotations

from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.planner import build_query_plan


HISTORY = [
    {"role": "user", "content": "What are the admission requirements for M.Tech?"},
    {"role": "assistant", "content": "[previous grounded answer omitted]"},
]

QUESTION = "What about the application fee for that program?"


def main() -> None:
    print("=" * 100)
    print("CONVERSATION -> QUERY PLANNER PRODUCTION INTEGRATION GATE")
    print("=" * 100)

    resolution = resolve_conversation(
        question=QUESTION,
        chat_history=HISTORY,
    )

    print("\nCONVERSATION RESOLUTION")
    print("original:", QUESTION)
    print("resolved:", resolution.get("resolved_question"))
    print("mode:", resolution.get("mode"))
    print("source:", resolution.get("resolution_source"))
    print("entity:", resolution.get("active_entity"))
    print("topic:", resolution.get("active_topic"))

    assert resolution["mode"] == "follow_up"
    assert resolution["resolved_question"] == "What about the application fee for M.Tech?"
    assert resolution["active_entity"] == "M.Tech"

    plan = build_query_plan(resolution["resolved_question"])

    print("\nQUERY PLAN")
    print("scope:", plan.domain_decision)
    print("scope_confidence:", plan.domain_confidence)
    print("multi:", plan.is_multi_intent)
    print("intent_count:", len(plan.intents))

    assert plan.domain_decision == "in_scope"
    assert plan.is_multi_intent is False
    assert len(plan.intents) == 1

    intent = plan.intents[0]
    print("type:", intent.request_type)
    print("target:", intent.target)
    print("attributes:", intent.requested_attributes)
    print("retrieval:", intent.retrieval_query)

    assert intent.target == "M.Tech"
    assert "application fee" in intent.requested_attributes
    assert intent.request_type == "fee"

    print("\nPASS: CONVERSATION -> PLANNER CONTRACT")


if __name__ == "__main__":
    main()
