from __future__ import annotations
import dataclasses, importlib, inspect, json, re
from pathlib import Path
from typing import Any, Iterable

CORE_ROOT=Path(__file__).resolve().parents[3]/"ai_platform"/"core"

def load_ai_modules():
    names=("query.understanding","query.pipeline","retrieval.query","retrieval.hybrid","retrieval.rrf","retrieval.ranking","retrieval.semantic_alignment","retrieval.verification","evidence.evidence","evidence.grouping","evidence.coverage","evidence.claims","evidence.packaging","evidence.scope","graph.nodes","graph.graph","graph.state")
    out={}
    for n in names: out[n.rsplit('.',1)[-1]]=importlib.import_module("ai_platform.core."+n)
    return out

def safe_text(v): return "" if v is None else (v if isinstance(v,str) else str(v))
def norm(v): return re.sub(r"\s+"," ",safe_text(v)).strip().casefold()

def source_of(x):
    if isinstance(x,dict):
        md=x.get("metadata") or {}
        for k in ("source","file_path","path","url","document_source"):
            if x.get(k): return safe_text(x[k])
            if isinstance(md,dict) and md.get(k): return safe_text(md[k])
    md=getattr(x,"metadata",None)
    if isinstance(md,dict):
        for k in ("source","file_path","path","url","document_source"):
            if md.get(k): return safe_text(md[k])
    for k in ("source","file_path","path","url","document_source"):
        v=getattr(x,k,None)
        if v: return safe_text(v)
    return ""

def text_of(x):
    if isinstance(x,dict):
        for k in ("page_content","content","text","document_text","evidence_text","snippet"):
            if x.get(k): return safe_text(x[k])
    for k in ("page_content","content","text","document_text","evidence_text","snippet"):
        v=getattr(x,k,None)
        if v: return safe_text(v)
    return ""

def plain(v,depth=0,items=25):
    if depth>4: return repr(v)[:1200]
    if v is None or isinstance(v,(str,int,float,bool)): return v
    if dataclasses.is_dataclass(v):
        try: return plain(dataclasses.asdict(v),depth+1,items)
        except Exception: pass
    if hasattr(v,"model_dump"):
        try: return plain(v.model_dump(),depth+1,items)
        except Exception: pass
    if hasattr(v,"dict") and callable(v.dict):
        try: return plain(v.dict(),depth+1,items)
        except Exception: pass
    if isinstance(v,dict): return {str(k):plain(val,depth+1,items) for k,val in list(v.items())[:items]}
    if isinstance(v,(list,tuple,set,frozenset)): return [plain(i,depth+1,items) for i in list(v)[:items]]
    if hasattr(v,"__dict__"):
        d={}
        for k,val in list(vars(v).items())[:items]:
            if not k.startswith("_"): d[k]=plain(val,depth+1,items)
        return d or repr(v)[:1200]
    return repr(v)[:1200]

def walk(v,path="state",seen=None):
    if seen is None: seen=set()
    oid=id(v)
    if oid in seen: return
    seen.add(oid)
    yield path,v
    if isinstance(v,dict):
        for k,val in v.items(): yield from walk(val,f"{path}.{k}",seen)
    elif isinstance(v,(list,tuple)):
        for i,val in enumerate(v[:80]): yield from walk(val,f"{path}[{i}]",seen)
    elif hasattr(v,"__dict__"):
        for k,val in vars(v).items():
            if not k.startswith("_"): yield from walk(val,f"{path}.{k}",seen)

def collect_docs(state):
    out=[]; seen=set()
    for path,val in walk(state):
        t=text_of(val)
        if not t: continue
        s=source_of(val)
        if not (s or isinstance(val,dict) or hasattr(val,"metadata")): continue
        key=(norm(s),norm(t))
        if key in seen: continue
        seen.add(key); out.append({"path":path,"source":s,"text":t,"object":val})
    return out

def marker_hits(text,markers):
    n=norm(text); return [m for m in markers if norm(m) in n]

def snapshot(stage,state,case,limit=12):
    docs=collect_docs(state); rows=[]
    for i,d in enumerate(docs[:100],1):
        txt=d["text"]+" "+d["source"]
        pos=marker_hits(txt,case["positive_markers"]); neg=marker_hits(txt,case["negative_markers"])
        rows.append({"observed_rank":i,"path":d["path"],"source":d["source"],"text_preview":re.sub(r"\s+"," ",d["text"])[:600],"positive_hits":pos,"negative_hits":neg,"diagnostic_score":2*len(pos)-len(neg)})
    rows.sort(key=lambda x:x["diagnostic_score"],reverse=True)
    rel=next((i for i,x in enumerate(rows,1) if x["positive_hits"]),None)
    wrong=[x for x in rows if x["negative_hits"] and not x["positive_hits"]]
    return {"state_keys":sorted(state.keys()),"document_objects_observed":len(docs),"relevant_rank":rel,"wrong_evidence":wrong,"documents":rows[:limit],"state_preview":plain(state,items=30)}

def run_node(node,state):
    result=node(state)
    if inspect.isawaitable(result): raise RuntimeError(f"{getattr(node,'__name__',node)} returned awaitable; expected synchronous node")
    if result is None: return state
    if not isinstance(result,dict): raise TypeError(f"{getattr(node,'__name__',node)} returned {type(result).__name__}, expected dict state")
    merged=dict(state); merged.update(result); return merged

def graph_nodes(module):
    names=("resolve_conversation_node","understand_query_node","plan_multi_intent_node","hybrid_retrieve_node","fuse_retrieved_documents_node","verify_and_rank_node","evidence_context_node","assess_evidence_node","assess_coverage_node","audit_claims_node","package_evidence_node")
    return [(n,getattr(module,n)) for n in names if hasattr(module,n)]

def make_initial_state(q):
    return {"question":q,"query":q,"original_query":q,"resolved_question":q,"chat_history":[],"messages":[],"conversation":[]}

def answer_node(module): return getattr(module,"answer_node",None)


def best_relevant_rank(stage_snapshot):
    if not isinstance(stage_snapshot, dict):
        return None
    return stage_snapshot.get("relevant_rank")
def backend_imports():
    hits=[]
    if not CORE_ROOT.exists(): return hits
    for p in CORE_ROOT.rglob("*.py"):
        try: txt=p.read_text(encoding="utf-8")
        except Exception: continue
        if re.search(r"(^|\n)\s*(from|import)\s+backend(\.|\s|$)",txt): hits.append(str(p))
    return hits

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=plain),encoding="utf-8")

def inventory():
    mods=load_ai_modules(); out={}
    for key,m in mods.items():
        out[key]={}
        for name in sorted(dir(m)):
            if name.startswith("_"): continue
            obj=getattr(m,name)
            if inspect.isfunction(obj) or inspect.isclass(obj):
                try: sig=str(inspect.signature(obj))
                except Exception: sig="<signature unavailable>"
                out[key][name]=sig
    return out

def markdown(results,backend_hits,with_llm):
    L=["# AI Platform V1 Production Diagnosis","",f"Scope: `ai_platform/core` only. Backend imports in core: {len(backend_hits)}.",f"Answer LLM: {'ON' if with_llm else 'OFF'}","","## Summary","","| Case | First failure | Best evidence stage | Wrong evidence | Terminal |","|---|---|---|---|---|"]
    for r in results: L.append(f"| {r['case_id']} | {r.get('first_failure_stage') or '-'} | {r.get('best_evidence_stage') or '-'} | {'YES' if r.get('wrong_evidence_survived') else 'NO'} | {r.get('terminal_status','-')} |")
    for r in results:
        L += ["",f"## {r['case_id']}","",f"**Question:** {r['question']}","",f"**First failure:** `{r.get('first_failure_stage') or 'none observed'}`  ",f"**Best evidence:** `{r.get('best_evidence_stage') or 'none observed'}`  ",f"**Wrong evidence survived:** `{r.get('wrong_evidence_survived')}`","","### Stage trace","","| Stage | Docs | Relevant rank | Wrong-only docs | Error |","|---|---:|---:|---:|---|"]
        for s,d in r['stages'].items(): L.append(f"| {s} | {d.get('document_objects_observed',0)} | {d.get('relevant_rank','-') if d.get('relevant_rank') is not None else '-'} | {len(d.get('wrong_evidence',[]))} | {d.get('error','')} |")
        L += ["","### Top observed evidence"," "]
        for s,d in r['stages'].items():
            if d.get('documents'):
                L.append(f"#### {s}")
                for row in d['documents'][:5]: L.append(f"- `{row.get('source','')}` | +{row.get('positive_hits',[])} | -{row.get('negative_hits',[])} | {row.get('text_preview','')}")
        L += ["","### Terminal state","","```json",json.dumps(r.get('terminal',{}),indent=2,ensure_ascii=False,default=plain),"```"]
    return "\n".join(L)+"\n"
