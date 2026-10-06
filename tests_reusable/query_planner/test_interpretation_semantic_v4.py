#!/usr/bin/env python3
import json
from ai_platform.core.query import planner
class R:
    def __init__(self,p): self.content=json.dumps(p)
class M:
    def __init__(self,p): self.p=p
    def invoke(self,_): return R(self.p)
def sig(p): return sorted((i.request_type,i.target,tuple(sorted(i.requested_attributes,key=str.casefold))) for i in p.intents)
def chk(name,q,payload,expected):
    got=sig(planner.build_query_plan(question=q,model=M(payload))); assert got==sorted(expected),f'{name}\nEXPECTED={expected}\nOBSERVED={got}'; print('PASS',name)
chk('target punctuation','What is the deadline for M.Tech applications?',{'domain_decision':'in_scope','intents':[{'request_type':'deadline','target':'M','requested_attributes':['deadline']}]}, [('deadline','M.Tech',('deadline',))])
chk('hinglish deadline','M.Tech ka form kab bharna hai?',{'domain_decision':'in_scope','intents':[{'request_type':'fee','target':'M.Tech','requested_attributes':['fee']}]}, [('deadline','M.Tech',('deadline',))])
chk('gate target','M.Tech mein GATE zaroori hai kya?',{'domain_decision':'in_scope','intents':[{'request_type':'eligibility','target':'GATE','requested_attributes':['eligibility']}]}, [('eligibility','M.Tech',('GATE',))])
chk('disclaimer','I am not asking whether I am eligible for M.Tech. I only need the application fee.',{'domain_decision':'in_scope','intents':[{'request_type':'fee','target':'M.Tech','requested_attributes':['application fee']},{'request_type':'eligibility','target':'M.Tech','requested_attributes':['eligibility']}]}, [('fee','M.Tech',('application fee',))])
chk('oos','Can you recommend the best laptop under INR 80000?',{'domain_decision':'out_of_scope','intents':[{'request_type':'general'}]}, [])
print('SEMANTIC V4 FINAL REGRESSION: PASS')
