from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from tests_reusable.test_diagnosis.cases.benchmark import get_cases
from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules,graph_nodes,answer_node,make_initial_state,run_node,snapshot,best_relevant_rank,backend_imports,write_json,markdown,plain

def run_case(case,with_llm=False,save_state=False):
    modules=load_ai_modules(); gm=modules['nodes']; state=make_initial_state(case['question']); stages={}; first_failure=None; best_stage=None; best_rank=None
    for name,node in graph_nodes(gm):
        try:
            state=run_node(node,state); snap=snapshot(name,state,case); stages[name]=snap
            rr=snap.get('relevant_rank')
            if rr is not None and (best_rank is None or rr<best_rank): best_rank=rr; best_stage=name
            if first_failure is None and snap.get('document_objects_observed',0)>0 and rr is None: first_failure=name
        except Exception as e:
            stages[name]={"error":f"{type(e).__name__}: {e}","state_keys":sorted(state.keys()),"document_objects_observed":0,"documents":[],"wrong_evidence":[]}
            if first_failure is None: first_failure=name
            break
    if with_llm:
        node=answer_node(gm)
        if node is None:
            stages['answer_node']={"error":"answer_node unavailable"}
        else:
            try: state=run_node(node,state); stages['answer_node']=snapshot('answer_node',state,case)
            except Exception as e: stages['answer_node']={"error":f"{type(e).__name__}: {e}","state_keys":sorted(state.keys())}
    wrong=any(s.get('wrong_evidence') for s in stages.values())
    terminal_keys=('answer','response','final_answer','answer_text')
    has_answer=any(state.get(k) not in (None,'',[],{}) for k in terminal_keys)
    if with_llm and not has_answer: status='llm_stage_no_answer_observed'
    elif with_llm and 'answer_node' in stages and stages['answer_node'].get('error'): status='llm_stage_failure'
    elif best_stage: status='passed_to_final_context'
    else: status='no_relevant_evidence_observed'
    terminal={k:plain(state.get(k)) for k in ('answer','response','final_answer','answer_text','formatted_context','final_context','context','evidence_status','coverage_status','claim_status','package_status','verification_status','grounding_status','guard_status') if k in state}
    out={"case_id":case['id'],"question":case['question'],"first_failure_stage":first_failure,"best_evidence_stage":best_stage,"best_evidence_rank":best_rank,"wrong_evidence_survived":bool(wrong),"terminal_status":status,"stages":stages,"terminal":terminal}
    if save_state: out['terminal_state']=plain(state,items=60)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=6); ap.add_argument('--case',action='append',default=[]); ap.add_argument('--with-llm',action='store_true'); ap.add_argument('--save-state',action='store_true')
    a=ap.parse_args(); cases=get_cases();
    if a.case: cases=[c for c in cases if c['id'] in set(a.case)]
    cases=cases[:max(1,a.limit)]
    hits=backend_imports()
    print('='*88); print('AI PLATFORM V1 PRODUCTION DIAGNOSIS'); print('='*88); print(f'Cases: {len(cases)} | Answer LLM: {"ON" if a.with_llm else "OFF"} | Backend imports in core: {len(hits)}'); print('')
    if hits:
        print('BACKEND DEPENDENCY DETECTED:'); [print('  ',h) for h in hits]; print('')
    results=[]
    for i,c in enumerate(cases,1):
        print(f'[{i}/{len(cases)}] {c["id"]} :: {c["question"]}'); r=run_case(c,with_llm=a.with_llm,save_state=a.save_state); results.append(r); print(f'    first_failure={r["first_failure_stage"] or "none"} | best_evidence={r["best_evidence_stage"] or "none"} | wrong_survived={r["wrong_evidence_survived"]} | terminal={r["terminal_status"]}')
    d=Path(__file__).resolve().parent/'reports'; d.mkdir(parents=True,exist_ok=True); payload={"scope":"ai_platform/core","backend_imports":hits,"with_llm":a.with_llm,"results":results}; write_json(d/'retrieval_diagnosis.json',payload); (d/'retrieval_diagnosis.md').write_text(markdown(results,hits,a.with_llm),encoding='utf-8')
    print(''); print('Reports written:'); print('  ',d/'retrieval_diagnosis.md'); print('  ',d/'retrieval_diagnosis.json')

if __name__=='__main__': main()
