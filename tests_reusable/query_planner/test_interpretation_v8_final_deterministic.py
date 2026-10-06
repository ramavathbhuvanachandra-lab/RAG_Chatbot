from __future__ import annotations

import importlib.util
import importlib.machinery
import json
import sys
import types
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent


def install_stubs() -> None:
    models = types.ModuleType("ai_platform.core.query.models")
    for name in ("Ambiguity", "Constraint", "ListIntent", "Query", "QueryFacet", "Qualifier", "RetrievalRequirement", "SemanticQueryFrame", "Target"):
        setattr(models, name, type(name, (), {}))
    ai = types.ModuleType("ai_platform")
    core = types.ModuleType("ai_platform.core")
    query = types.ModuleType("ai_platform.core.query")
    ai.core = core
    core.query = query
    query.models = models
    sys.modules.update({"ai_platform": ai, "ai_platform.core": core, "ai_platform.core.query": query, "ai_platform.core.query.models": models})


def load_planner():
    install_stubs()
    source = ROOT / "planner.py.production_v6"
    loader = importlib.machinery.SourceFileLoader("planner_v6_test", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        return json.dumps(self.payload)


def intent(plan):
    assert plan.intents
    return plan.intents[0]


def run():
    p = load_planner()

    cases = []

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"Can I apply for MBA?","request_type":"eligibility","target":"MBA","requested_attributes":["eligibility"]},
    ]}
    cases.append(("apply_is_eligibility_without_steps", "Can I apply for MBA?", noisy, "eligibility", "MBA", {"eligibility"}))

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"What is the application fee for PhD?","request_type":"fee","target":"PhD","requested_attributes":["processing fee"]},
    ]}
    cases.append(("specific_fee_overrides_llm", "What is the application fee for PhD?", noisy, "fee", "PhD", {"application fee"}))

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"What is the admission process for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":[]},
    ]}
    cases.append(("process_overrides_llm_type", "What is the admission process for M.Tech?", noisy, "process", "M.Tech", {"admission process"}))

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"I am not asking about eligibility; I only need the application fee for M.Tech.","request_type":"fee","target":"M.Tech","requested_attributes":["application fee"]},
        {"question":"What are the eligibility requirements for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":["eligibility"]},
    ]}
    cases.append(("disclaimer_does_not_create_intent", "I am not asking about eligibility; I only need the application fee for M.Tech.", noisy, "fee", "M.Tech", {"application fee"}))

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"Is GATE required for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":["GATE"]},
        {"question":"What are the eligibility requirements for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":["eligibility"]},
    ]}
    cases.append(("eligibility_fragments_merge", "Is GATE mandatory for M.Tech, and can I still apply if my percentage is below the threshold?", noisy, "eligibility", "M.Tech", {"GATE","eligibility"}))

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"Is work experience required for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":["work experience"]},
        {"question":"What CGPA is required for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":["CGPA"]},
        {"question":"Is GATE required for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":["GATE"]},
        {"question":"What are the eligibility requirements for M.Tech?","request_type":"eligibility","target":"M.Tech","requested_attributes":["eligibility"]},
    ]}
    plan = p.build_query_plan("Please tell me whether I am eligible for M.Tech and whether any work experience is required.", model=FakeModel(noisy))
    assert len(plan.intents) == 2, plan
    assert any("work experience" in (i.requested_attributes or []) for i in plan.intents)
    assert any((i.requested_attributes or []) and "eligibility" in i.requested_attributes for i in plan.intents)
    print("PASS explicit whether objectives remain separate")

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"What is M.Sc process?","request_type":"process","target":"M.Sc","requested_attributes":["process"],"constraints":["Sc"]},
    ]}
    plan = p.build_query_plan("M.Sc ka application process batao.", model=FakeModel(noisy))
    i = intent(plan)
    assert "Sc" not in i.constraints, i.constraints
    assert i.target == "M.Sc", i.target
    print("PASS M.Sc does not leak SC constraint")

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"Is work experience required?","request_type":"eligibility","target":"M.Tech","requested_attributes":["work experience"],"constraints":["not compulsory"]},
    ]}
    plan = p.build_query_plan("Is work experience not compulsory for M.Tech?", model=FakeModel(noisy))
    i = intent(plan)
    assert i.request_type == "eligibility"
    assert "work experience" in i.requested_attributes
    assert any("not compulsory" in x.casefold() for x in i.constraints)
    assert any("not required" in x.casefold() for x in i.constraints)
    print("PASS negated work experience stays one eligibility objective")

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"What is the fee?","request_type":"fee","target":"M.Tech","requested_attributes":["fee"]},
        {"question":"What documents?","request_type":"documents","target":"M.Tech","requested_attributes":["documents"]},
    ]}
    plan = p.build_query_plan("What is the application fee for M.Tech and what documents are required?", model=FakeModel(noisy))
    assert len(plan.intents) == 2, plan
    print("PASS fee + documents remain two objectives")

    noisy = {"domain_decision":"in_scope","domain_confidence":0.95,"intents":[
        {"question":"generic model answer","request_type":"eligibility","target":"M.Tech","requested_attributes":["eligibility"]},
    ]}
    plan = p.build_query_plan("Walk me through the M.Tech admission process from application to selection.", model=FakeModel(noisy))
    i = intent(plan)
    assert i.request_type == "process" and "admission process" in i.requested_attributes, (i.request_type, i.requested_attributes)
    print("PASS admission process is not overridden by noisy eligibility")

    plan = p.build_query_plan("How much is the PhD application processing fee?", model=FakeModel(noisy))
    i = intent(plan)
    assert i.request_type == "fee" and "application fee" in i.requested_attributes, i
    print("PASS application processing fee maps to application fee")

    plan = p.build_query_plan("For M.Tech, tell me the application fee and also which supporting documents I need to upload at the time of submission.", model=FakeModel(noisy))
    assert len(plan.intents) == 2 and plan.is_multi_intent
    assert any(i.request_type == "fee" and "application fee" in i.requested_attributes for i in plan.intents)
    assert any(i.request_type == "documents" and "documents" in i.requested_attributes for i in plan.intents)
    print("PASS fee + documents split into two objectives")

    plan = p.build_query_plan("I know the MBA eligibility already. I only need the steps for submitting the application.", model=FakeModel(noisy))
    i = intent(plan)
    assert i.request_type == "process" and "application process" in i.requested_attributes, i
    print("PASS disclaimer does not steal application process")

    plan = p.build_query_plan("Can I apply for MBA this year?", model=FakeModel(noisy))
    i = intent(plan)
    assert i.request_type == "process" and i.target == "MBA", i
    assert i.requested_attributes == ["application process"], i.requested_attributes
    print("PASS application-availability is one process objective")

    plan = p.build_query_plan("I'm in the final semester of B.Tech and my final result is not out yet. Can I submit an M.Tech application now, or do I have to wait for graduation?", model=FakeModel(noisy))
    i = intent(plan)
    assert i.request_type == "eligibility" and any("final" in c.casefold() or "result" in c.casefold() or "graduation" in c.casefold() for c in i.constraints), i.constraints
    print("PASS final-semester constraints preserved")

    print("INTERPRETATION V6 DETERMINISTIC REGRESSION: PASS")


if __name__ == "__main__":
    run()
