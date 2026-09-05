from backend.nodes import (
    resolve_conversation_node,
    hybrid_retrieve,
)


def test_standalone_question_reaches_retriever_unchanged(
    monkeypatch,
):
    """
    A standalone question should reach retrieval unchanged.
    """

    captured = {}

    def fake_dense_retrieve(question):
        captured["question"] = question
        return []

    def fake_keyword_retrieve(question):
        return []

    monkeypatch.setattr(
        "backend.nodes.dense_retrieve",
        fake_dense_retrieve,
    )

    monkeypatch.setattr(
        "backend.nodes.keyword_retrieve",
        fake_keyword_retrieve,
    )

    state = {
        "question": "What is the B.Tech admission process?",
        "chat_history": [],
    }

    resolved_state = resolve_conversation_node(
        state
    )

    hybrid_retrieve(
        {
            **state,
            **resolved_state,
        }
    )

    assert captured["question"] == (
        "What is the B.Tech admission process?"
    )


def test_follow_up_question_reaches_retriever_resolved(
    monkeypatch,
):
    """
    A contextual follow-up should be resolved before retrieval.
    """

    captured = {}

    def fake_dense_retrieve(question):
        captured["question"] = question
        return []

    def fake_keyword_retrieve(question):
        return []

    monkeypatch.setattr(
        "backend.nodes.dense_retrieve",
        fake_dense_retrieve,
    )

    monkeypatch.setattr(
        "backend.nodes.keyword_retrieve",
        fake_keyword_retrieve,
    )

    history = [
        {
            "role": "user",
            "content": "What is the B.Tech admission process?",
        },
        {
            "role": "assistant",
            "content": (
                "The B.Tech admission process is handled "
                "through the relevant admission route."
            ),
        },
    ]

    def fake_resolver(
        question,
        chat_history,
    ):
        assert question == "What about fees?"
        assert chat_history == history

        return {
            "mode": "follow_up",
            "resolved_question": (
                "What are the B.Tech admission fees?"
            ),
            "active_topic": "admission",
            "active_entity": "B.Tech",
        }

    monkeypatch.setattr(
        "backend.nodes.resolve_conversation",
        fake_resolver,
    )

    state = {
        "question": "What about fees?",
        "chat_history": history,
    }

    resolved_state = resolve_conversation_node(
        state
    )

    hybrid_retrieve(
        {
            **state,
            **resolved_state,
        }
    )

    assert captured["question"] == (
        "What are the B.Tech admission fees?"
    )


def test_topic_switch_reaches_retriever_unchanged(
    monkeypatch,
):
    """
    An explicit topic switch should pass through unchanged.

    The resolver function may still be called because it performs
    the cheap local classification, but the query LLM must not be
    invoked for an explicit topic switch.
    """

    captured = {}
    resolver_llm_called = False

    def fake_dense_retrieve(question):
        captured["question"] = question
        return []

    def fake_keyword_retrieve(question):
        return []

    monkeypatch.setattr(
        "backend.nodes.dense_retrieve",
        fake_dense_retrieve,
    )

    monkeypatch.setattr(
        "backend.nodes.keyword_retrieve",
        fake_keyword_retrieve,
    )

    history = [
        {
            "role": "user",
            "content": "Tell me about hostel facilities.",
        },
        {
            "role": "assistant",
            "content": "IIT Jodhpur hostels provide ...",
        },
    ]

    def fail_if_query_llm_called(*args, **kwargs):
        nonlocal resolver_llm_called

        resolver_llm_called = True

        raise AssertionError(
            "Query LLM should not be called for an explicit "
            "topic switch."
        )

    monkeypatch.setattr(
        "backend.conversation_resolver.invoke_query_llm",
        fail_if_query_llm_called,
    )

    state = {
        "question": "Now tell me about M.Sc. admission.",
        "chat_history": history,
    }

    resolved_state = resolve_conversation_node(
        state
    )

    hybrid_retrieve(
        {
            **state,
            **resolved_state,
        }
    )

    assert resolved_state["conversation_mode"] == (
        "topic_switch"
    )

    assert captured["question"] == (
        "Now tell me about M.Sc. admission."
    )

    assert resolver_llm_called is False