import json

from ai_platform.core.query.planner import (
    build_query_plan,
    intent_units_from_plan,
    plan_to_frame,
    retrieval_queries_from_plan,
    should_stop_pipeline,
)


class FakeResponse:
    def __init__(self, payload):
        self.content = json.dumps(payload)


class FakeModel:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {}
        self.error = error
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


def make_plan(question, **payload):
    model = FakeModel(payload)
    plan = build_query_plan(question, model=model)
    assert model.calls == 1, "Planner must make exactly one LLM call"
    return plan


def attrs(plan):
    return [tuple(i.requested_attributes) for i in plan.intents]


def test_case_1_fee_duplicate_intents_collapse():
    plan = make_plan(
        "What is the application fee for the PhD program?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[
            {"request_type": "fee", "target": "PhD", "requested_attributes": ["application fee"]},
            {"request_type": "fee", "target": "PhD", "requested_attributes": ["fee"]},
        ],
    )
    assert plan.is_multi_intent is False
    assert len(plan.intents) == 1
    assert plan.intents[0].target == "PhD"
    assert plan.intents[0].requested_attributes == ["application fee"]
    assert plan.intents[0].request_type == "fee"


def test_case_2_apply_means_process_only_when_how_to_is_requested():
    plan = make_plan(
        "How do I apply for M.Tech admission?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "eligibility", "target": "M.Tech", "requested_attributes": []}],
    )
    assert plan.is_multi_intent is False
    assert attrs(plan) == [("process",)]
    assert plan.intents[0].request_type == "process"


def test_case_3_minimum_percentage_is_not_duplicated_with_percentage():
    plan = make_plan(
        "What is the minimum percentage required for M.Tech admission?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[
            {"request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["minimum percentage"]},
            {"target": "M.Tech", "requested_attributes": ["percentage"]},
        ],
    )
    assert plan.is_multi_intent is False
    assert attrs(plan) == [("minimum percentage",)]
    assert plan.intents[0].request_type == "eligibility"


def test_case_4_registration_documents():
    plan = make_plan(
        "What documents are required for registration?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "documents", "target": "registration", "requested_attributes": []}],
    )
    assert attrs(plan) == [("documents",)]
    assert plan.intents[0].target == "registration"
    assert plan.intents[0].request_type == "documents"


def test_case_5_eligibility_and_work_experience_not_process():
    question = (
        "I am currently in my final year of B.Tech and I want to apply for M.Tech after graduation. "
        "I have a CGPA of 7.1 and I have qualified GATE. Please tell me whether I am eligible "
        "and whether any work experience is required."
    )
    plan = make_plan(
        question,
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["process"]}],
    )
    assert plan.is_multi_intent is True
    assert set(attrs(plan)) == {("eligibility",), ("work experience",)}
    assert all(i.target == "M.Tech" for i in plan.intents)
    assert all("CGPA of 7.1" in i.constraints for i in plan.intents)
    assert all("qualified GATE" in i.constraints for i in plan.intents)


def test_case_6_three_distinct_objectives_and_no_fee_duplicate():
    plan = make_plan(
        "What is the M.Tech eligibility, what documents do I need for the application, and how much is the application fee?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "fee", "target": "M.Tech", "requested_attributes": ["application fee", "fee"]}],
    )
    assert plan.is_multi_intent is True
    assert set(attrs(plan)) == {("eligibility",), ("documents",), ("application fee",)}
    assert len(plan.intents) == 3
    assert all(i.target == "M.Tech" for i in plan.intents)


def test_case_7_three_distinct_mba_objectives():
    plan = make_plan(
        "I want to apply for MBA. Can you tell me the admission process, eligibility criteria, and important deadlines?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[
            {"request_type": "deadline", "target": "MBA", "requested_attributes": ["deadline"]},
            {"request_type": "eligibility", "target": "MBA", "requested_attributes": ["eligibility"]},
            {"request_type": "process", "target": "MBA", "requested_attributes": ["process"]},
        ],
    )
    assert plan.is_multi_intent is True
    assert set(attrs(plan)) == {("deadline",), ("eligibility",), ("process",)}
    assert len(plan.intents) == 3
    assert all(i.target == "MBA" for i in plan.intents)


def test_case_8_comparison_is_one_clean_intent():
    plan = make_plan(
        "What is the difference between the M.Tech and MBA admission processes?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[
            {
                "request_type": "comparison",
                "target": "M.Tech admission process and MBA admission process and M.Tech and MBA",
                "requested_attributes": ["admission process"],
                "comparison_targets": ["M.Tech", "MBA"],
            }
        ],
    )
    assert plan.is_multi_intent is False
    assert len(plan.intents) == 1
    intent = plan.intents[0]
    assert intent.target == "M.Tech and MBA"
    assert intent.request_type == "comparison"
    assert intent.requested_attributes == ["admission process"]
    assert intent.comparison_targets == ["M.Tech", "MBA"]
    assert intent.retrieval_query == "M.Tech and MBA admission process"


def test_case_9_can_i_apply_is_eligibility_not_process():
    question = (
        "I qualified GATE, but my undergraduate percentage is below the usual threshold. "
        "Can I still apply for M.Tech and are there any exceptions?"
    )
    plan = make_plan(
        question,
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["process"]}],
    )
    assert attrs(plan) == [("eligibility",)]
    assert plan.intents[0].request_type == "eligibility"
    assert "process" not in plan.intents[0].requested_attributes


def test_case_10_negation_preserved():
    plan = make_plan(
        "Is work experience not required for M.Tech admission?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["work experience"]}],
    )
    assert attrs(plan) == [("work experience",)]
    assert plan.intents[0].negated is True
    assert "not required" in plan.intents[0].constraints


def test_case_11_llm_target_contamination_is_removed():
    plan = make_plan(
        "What are the documents required for M.Tech admission?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[
            {
                "request_type": "eligibility",
                "target": "M.Tech admission documents",
                "requested_attributes": ["documents"],
            }
        ],
    )
    assert attrs(plan) == [("documents",)]
    assert plan.intents[0].target == "M.Tech"
    assert plan.intents[0].request_type == "documents"


def test_case_12_follow_up_text_is_not_split_into_synonym_duplicates():
    plan = make_plan(
        "What about the application fee for that program?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[
            {"request_type": "fee", "target": "program", "requested_attributes": ["application fee"]},
            {"request_type": "fee", "target": "program", "requested_attributes": ["fee"]},
        ],
    )
    assert plan.is_multi_intent is False
    assert attrs(plan) == [("application fee",)]
    assert plan.intents[0].target == "program"


def test_cases_13_to_15_are_stopped_out_of_scope():
    for q in (
        "Who won yesterday's cricket match?",
        "Explain binary search with an example.",
        "asdf qwerty 987 hello banana xyz",
    ):
        plan = make_plan(
            q,
            domain_decision="out_of_scope",
            domain_confidence=0.98,
            intents=[],
        )
        assert should_stop_pipeline(plan) is True
        assert plan.domain_decision == "out_of_scope"


def test_primary_query_remains_original_and_intent_queries_are_compact():
    plan = make_plan(
        "What is the minimum percentage required for M.Tech admission?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["minimum percentage"]}],
    )
    assert plan.resolved_query == plan.original_query
    assert retrieval_queries_from_plan(plan)[0] == plan.original_query
    assert plan.intents[0].retrieval_query == "M.Tech minimum percentage"


def test_plan_to_frame_preserves_attribute_and_constraints():
    question = "Is work experience not required for M.Tech admission?"
    plan = make_plan(
        question,
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["work experience"]}],
    )
    frame = plan_to_frame(plan)
    assert frame.target == "M.Tech"
    assert frame.request_type == "eligibility"
    assert "work experience" in {f.value for f in frame.facets}
    assert "not required" in frame.conditions


def test_intent_units_have_one_query_and_frame_per_intent():
    plan = make_plan(
        "What is the M.Tech eligibility, what documents do I need, and how much is the application fee?",
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "fee", "target": "M.Tech", "requested_attributes": ["application fee"]}],
    )
    units = intent_units_from_plan(plan)
    assert len(units) == 3
    assert all(unit["query"] is not None for unit in units)
    assert all(unit["query_frame"] is not None for unit in units)


def test_fallback_still_uses_exactly_one_failed_llm_call():
    model = FakeModel(error=RuntimeError("simulated model failure"))
    plan = build_query_plan("How do I apply for M.Tech admission?", model=model)
    assert model.calls == 1
    assert plan.domain_decision == "in_scope"
    assert len(plan.intents) == 1
    assert plan.intents[0].target == "M.Tech"
    assert plan.intents[0].requested_attributes == ["process"]
    assert plan.intents[0].request_type == "process"


def test_no_model_output_does_not_create_duplicate_intents():
    class Empty:
        def __init__(self): self.calls = 0
        def invoke(self, prompt):
            self.calls += 1
            return "not json"

    model = Empty()
    plan = build_query_plan("What is the application fee for the PhD program?", model=model)
    assert model.calls == 1
    assert len(plan.intents) == 1
    assert plan.intents[0].requested_attributes == ["application fee"]


def test_retrieval_query_never_becomes_a_huge_copy_of_the_question():
    question = (
        "I am currently in my final year of B.Tech and I want to apply for M.Tech after graduation. "
        "I have a CGPA of 7.1 and I have qualified GATE. Please tell me whether I am eligible "
        "and whether any work experience is required."
    )
    plan = make_plan(
        question,
        domain_decision="in_scope",
        domain_confidence=1.0,
        intents=[{"request_type": "eligibility", "target": "M.Tech", "requested_attributes": ["work experience"]}],
    )
    for intent in plan.intents:
        assert len(intent.retrieval_query) < len(question)
        assert "M.Tech" in intent.retrieval_query



def test_fallback_extracts_registration_target_without_llm_output():
    class Empty:
        def __init__(self): self.calls = 0
        def invoke(self, prompt):
            self.calls += 1
            return "not json"
    model = Empty()
    plan = build_query_plan("What documents are required for registration?", model=model)
    assert model.calls == 1
    assert len(plan.intents) == 1
    assert plan.intents[0].target == "registration"
    assert plan.intents[0].requested_attributes == ["documents"]


def test_constraints_preserve_degree_and_threshold_language():
    class Empty:
        def __init__(self): self.calls = 0
        def invoke(self, prompt):
            self.calls += 1
            return "not json"
    q = (
        "I qualified GATE, but my undergraduate percentage is below the usual threshold. "
        "Can I still apply for M.Tech and are there any exceptions?"
    )
    model = Empty()
    plan = build_query_plan(q, model=model)
    assert model.calls == 1
    assert plan.intents[0].target == "M.Tech"
    assert "qualified GATE" in plan.intents[0].constraints
    assert "below the usual threshold" in plan.intents[0].constraints
    assert plan.intents[0].requested_attributes == ["eligibility"]
