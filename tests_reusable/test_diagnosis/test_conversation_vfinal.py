from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class FakeResponse:
    content: str


class FakeModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def _module(monkeypatch):
    mod = importlib.import_module("ai_platform.core.conversation.conversation")
    return mod


def test_that_program_resolves_deterministically_without_llm(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="What about the application fee for that program?",
        chat_history=[
            {"role": "user", "content": "What are the admission requirements for M.Tech?"},
            {"role": "assistant", "content": "The answer is based on the admission information."},
        ],
    )

    assert result["resolved_question"] == "What about the application fee for M.Tech?"
    assert result["mode"] == "follow_up"
    assert result["active_entity"] == "M.Tech"
    assert result["resolution_source"] == "deterministic"
    assert fake.calls == 0


def test_standalone_named_target_is_not_rewritten(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="What about the application fee for MBA?",
        chat_history=[{"role": "user", "content": "Please help with my previous question."}],
    )

    assert result["resolved_question"] == "What about the application fee for MBA?"
    assert result["mode"] == "standalone"
    assert fake.calls == 0


def test_multi_intent_first_reference(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="What about the application fee for the first program?",
        chat_history=[
            {"role": "user", "content": "Compare M.Tech and MBA admission processes."},
            {"role": "assistant", "content": "Both processes differ."},
        ],
    )

    assert result["resolved_question"] == "What about the application fee for M.Tech?"
    assert result["active_entity"] == "M.Tech"
    assert fake.calls == 0


def test_multi_intent_second_reference(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="What about the deadline for the second program?",
        chat_history=[
            {"role": "user", "content": "Compare M.Tech and MBA admission processes."},
        ],
    )

    assert result["resolved_question"] == "What about the deadline for MBA?"
    assert result["active_entity"] == "MBA"
    assert fake.calls == 0


def test_ambiguous_anaphor_is_not_guessed(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="What about the application fee for that program?",
        chat_history=[
            {"role": "user", "content": "Tell me about M.Tech and MBA admission."},
        ],
    )

    assert result["resolved_question"] == "What about the application fee for that program?"
    assert result["resolution_source"] == "ambiguous_preserved"
    assert result["active_entity"] == ""
    assert fake.calls == 0


def test_llm_resolves_contextual_followup_when_deterministic_data_is_insufficient(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(
        response=FakeResponse(
            '{"resolved_question":"What is the fee for the service desk?",'
            '"mode":"follow_up","active_topic":"fee",'
            '"active_entity":"service desk","confidence":0.91,"reason":"recent_context"}'
        )
    )
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="How much is it?",
        chat_history=[
            {"role": "user", "content": "Please refer to the service desk from the previous topic."},
            {"role": "assistant", "content": "Sure."},
        ],
    )

    assert fake.calls == 1
    # The test deliberately uses a history target not captured by the degree
    # extractor; the LLM result is accepted because it introduces no degree
    # contradiction. This verifies the bounded one-call fallback path.
    assert result["mode"] == "follow_up"
    assert result["resolved_question"] == "What is the fee for the service desk?"


def test_hallucinated_degree_rewrite_is_rejected(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(
        response=FakeResponse(
            '{"resolved_question":"What is the fee for PhD?",'
            '"mode":"follow_up","active_topic":"fee",'
            '"active_entity":"PhD","confidence":0.99}'
        )
    )
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="How much is it?",
        chat_history=[
            {"role": "user", "content": "I need help with the previous topic."},
        ],
    )

    assert fake.calls == 1
    assert result["resolved_question"] == "How much is it?"
    assert result["resolution_source"] == "llm_rejected"


def test_malformed_llm_output_fails_safe(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("not json"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="How much is it?",
        chat_history=[{"role": "user", "content": "Please help with my previous question."}],
    )

    assert fake.calls == 1
    assert result["resolved_question"] == "How much is it?"
    assert result["mode"] == "standalone"


def test_llm_failure_fails_safe(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(error=RuntimeError("boom"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="How much is it?",
        chat_history=[{"role": "user", "content": "Please help with my previous question."}],
    )

    assert fake.calls == 1
    assert result["resolved_question"] == "How much is it?"
    assert result["mode"] == "standalone"


def test_standalone_does_not_call_llm(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="What are the M.Tech eligibility requirements?",
        chat_history=[{"role": "user", "content": "What are the admission requirements?"}],
    )

    assert result["resolved_question"] == "What are the M.Tech eligibility requirements?"
    assert result["mode"] == "standalone"
    assert fake.calls == 0

def test_short_documents_question_resolves_to_single_history_anchor(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="What documents do I need?",
        chat_history=[
            {"role": "user", "content": "How do I apply for MBA?"},
            {"role": "assistant", "content": "You need to follow the MBA admission process."},
        ],
    )

    assert result["resolved_question"] == "What documents do I need for MBA?"
    assert result["mode"] == "follow_up"
    assert result["active_entity"] == "MBA"
    assert fake.calls == 0


def test_short_deadline_question_resolves_to_single_history_anchor(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="And what about the deadline?",
        chat_history=[
            {"role": "user", "content": "Tell me about M.Sc eligibility."},
            {"role": "assistant", "content": "Here are the eligibility requirements."},
        ],
    )

    assert result["resolved_question"] == "And what about the deadline for M.Sc?"
    assert result["mode"] == "follow_up"
    assert result["active_entity"] == "M.Sc"
    assert fake.calls == 0


def test_short_work_experience_question_resolves_to_single_history_anchor(monkeypatch):
    mod = _module(monkeypatch)
    fake = FakeModel(response=FakeResponse("{}"))
    monkeypatch.setattr(mod, "query_understanding_llm", fake)

    result = mod.resolve_conversation(
        question="Is work experience required?",
        chat_history=[
            {"role": "user", "content": "What are the requirements for M.Tech?"},
            {"role": "assistant", "content": "The program has specific eligibility criteria."},
        ],
    )

    assert result["resolved_question"] == "Is work experience required for M.Tech?"
    assert result["mode"] == "follow_up"
    assert result["active_entity"] == "M.Tech"
    assert fake.calls == 0
