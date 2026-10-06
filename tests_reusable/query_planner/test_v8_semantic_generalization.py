#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from importlib.machinery import SourceFileLoader
import pathlib, sys, types, json

ROOT=pathlib.Path(__file__).resolve().parents[2]
FIXTURE=ROOT/'tests_reusable/query_planner/planner.py.production_final_v8'

class Dummy:
    def __init__(self, **kw): self.__dict__.update(kw)
    def to_query(self): return self

models=types.ModuleType('ai_platform.core.query.models')
for n in ['Ambiguity','Constraint','ListIntent','Query','QueryFacet','Qualifier','RetrievalRequirement','SemanticQueryFrame','Target']:
    setattr(models,n,Dummy)
for pkg in ['ai_platform','ai_platform.core','ai_platform.core.query']:
    sys.modules.setdefault(pkg,types.ModuleType(pkg))
sys.modules['ai_platform.core.query.models']=models

loader=SourceFileLoader('v8planner', str(FIXTURE))
spec=importlib.util.spec_from_loader('v8planner', loader)
p=importlib.util.module_from_spec(spec)
loader.exec_module(p)
p.QueryPlan.model_rebuild(_types_namespace={'ScopeDecision': p.ScopeDecision, 'QueryIntentPlan': p.QueryIntentPlan})

class Noisy:
    def invoke(self,_):
        return json.dumps({
            'domain_decision':'in_scope','domain_confidence':0.2,
            'intents':[
                {'question':'eligibility + GATE','request_type':'fee','target':'GATE','requested_attributes':['GATE'],'constraints':[],'confidence':0.1},
                {'question':'eligibility','request_type':'eligibility','target':'M.Tech','requested_attributes':['eligibility','CGPA requirement for M.Tech admission'],'constraints':['59%','GATE qualified'],'confidence':0.2},
                {'question':'junk','request_type':'documents','target':'M.Tech','requested_attributes':['documents'],'constraints':[],'confidence':0.1},
            ],
            'confidence':0.1
        })

def plan(q):
    return p.build_query_plan(q, model=Noisy())

def assert_shape(q, expected_type, expected_target, attrs=(), max_intents=1):
    out=plan(q)
    intents=out.intents
    assert len(intents) <= max_intents, f'{q}: expected <= {max_intents}, got {len(intents)} {[(i.request_type,i.target,i.requested_attributes) for i in intents]}'
    ok=any(i.request_type==expected_type and str(i.target)==expected_target and all(a.casefold() in {x.casefold() for x in i.requested_attributes} for a in attrs) for i in intents)
    assert ok, f'{q}: wrong intents {[(i.request_type,i.target,i.requested_attributes,i.constraints) for i in intents]}'
    return out

checks=[
    ('I have 59% and GATE. Can I apply for M.Tech?', 'eligibility', 'M.Tech', ('eligibility','GATE'), 1),
    ('Is 60% enough for M.Tech admission?', 'eligibility', 'M.Tech', ('eligibility','percentage'), 1),
    ('I am in final semester. Can I submit my M.Tech application before my result?', 'eligibility', 'M.Tech', ('eligibility',), 1),
    ('Tell me whether I am eligible for M.Tech and separately whether work experience is compulsory.', 'eligibility', 'M.Tech', ('work experience',), 2),
]
for q,t,target,attrs,maxi in checks:
    o=assert_shape(q,t,target,attrs,maxi)
    print('PASS',q)
    print('   ',[(i.request_type,i.target,i.requested_attributes,i.constraints) for i in o.intents])
print('V8 SEMANTIC GENERALIZATION: PASS')
