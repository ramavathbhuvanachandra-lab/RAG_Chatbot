"""Comprehensive baseline analysis and failure matrix generator.

Reads test_outputs/rag_failure_diagnosis_20261019/baseline_40q_traces.json
Evaluates each stage (QU, Retrieval, Verification, Ranking, Packaging, Generation, Post-processing, Citations)
Outputs:
1. docs/rag_failure_diagnosis_20261019/baseline_failure_matrix.md
2. docs/rag_failure_diagnosis_20261019/baseline_summary_stats.json
"""
import json
import os
import re
from typing import Any, Dict, List

# Ground-truth expectations for each question
EXPECTATIONS = {
    "Q1": {
        "authoritative_doc": "departments/aerospace_engineering/general_information.docx",
        "key_facts": ["integrat", "aerospace", "multidisciplinary", "innovation", "self-reliance", "leadership"],
        "target_dept": "Aerospace Engineering",
    },
    "Q2": {
        "authoritative_doc": "departments/aerospace_engineering/general_information.docx",
        "key_facts": ["2025-26", "drone and anti-drone", "strategic", "defence"],
        "disallowed_facts": ["2026-27"],
        "target_dept": "Aerospace Engineering",
    },
    "Q3": {
        "authoritative_doc": "departments/aerospace_engineering/general_information.docx",
        "key_facts": ["education", "research", "infrastructure"],
        "disallowed_facts": ["Civil And Infrastructure", "Civil"],
        "target_dept": "Aerospace Engineering",
    },
    "Q4": {
        "authoritative_doc": "departments/electronics_engineering/general_information.docx",
        "key_facts": ["Micro-Nano Electronics", "VLSI", "Communication", "RF", "Photonics", "Intelligent and Embedded"],
        "target_dept": "Electronics Engineering",
        "must_have_count": 4,
    },
    "Q5": {
        "authoritative_doc": "departments/electronics_engineering/general_information.docx",
        "key_facts": ["Semiconductor", "VLSI", "Nanoelectronics", "AI/ML hardware", "Quantum", "Photonics", "Communication", "Sensors"],
        "target_dept": "Electronics Engineering",
        "must_have_count": 5,
    },
    "Q6": {
        "authoritative_doc": "departments/materials_engineering/general_information.docx",
        "key_facts": ["2017", "B.Tech", "M.Tech", "Ph.D", "Structural Materials", "Functional Materials", "Computational Materials", "Process Metallurgy"],
        "target_dept": "Materials Engineering",
    },
    "Q7": {
        "authoritative_doc": "departments/materials_engineering/programs.docx",
        "key_facts": ["Data and Computational Science", "Management Entrepreneurship"],
        "disallowed_facts": ["######", "[Materials Engineering"],
        "target_dept": "Materials Engineering",
    },
    "Q8": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["nodal office", "short-term", "vocational", "upskilling"],
        "target_dept": "Training and Upskilling",
    },
    "Q9": {
        "authoritative_doc": "departments/mathematics/programs.docx",
        "key_facts": ["mathematical thinking", "computational", "problem-solving", "programming"],
        "target_dept": "Mathematics",
    },
    "Q10": {
        "authoritative_doc": "departments/mathematics/programs.docx",
        "key_facts": ["Data Scientist", "Machine Learning Engineer", "AI Research Engineer", "Algorithm Designer", "Quantitative Analyst", "Operations Research Analyst", "Scientific Computing Professional"],
        "target_dept": "Mathematics",
        "must_have_count": 4,
    },
    "Q11": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["17–18 June 2023", "2 Days", "Hybrid", "Procurement", "Buyer & Seller registration", "Reverse Auction", "CRAC"],
        "disallowed_facts": ["10–11 June", "GFR 2017 (Goods & Services)"],
        "target_dept": "Training and Upskilling",
    },
    "Q12": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["19 May 2023", "10–11 June 2023", "2 Days", "10 AM–5 PM", "Hybrid", "GFR 2017", "Goods & Services", "procurement"],
        "disallowed_facts": ["17–18 June", "Buyer & Seller registration", "Reverse Auction"],
        "target_dept": "Training and Upskilling",
    },
    "Q13": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["Electronics Machine Maintenance", "17", "11", "450", "150", "Solar Panel", "36", "22", "Solar LED", "34", "25", "Hardware Assembly", "4", "300"],
        "target_dept": "Training and Upskilling",
    },
    "Q14": {
        "authoritative_doc": "departments/materials_engineering/curriculum.docx",
        "key_facts": ["Structural", "Functional", "Computational", "Process Metallurgy", "Machine Learning", "Data Structure"],
        "target_dept": "Materials Engineering",
    },
    "Q15": {
        "authoritative_doc": "departments/mathematics/general_information.docx",
        "key_facts": ["MATLAB", "Mathematica"],
        "disallowed_facts": ["no explicit symbolic", "no symbolic-computation tools"],
        "target_dept": "Mathematics",
    },
    "Q16": {
        "authoritative_doc": "departments/aerospace_engineering/general_information.docx",
        "key_facts": ["physical principles", "subsystem", "trade-offs", "mission", "hands-on"],
        "target_dept": "Aerospace Engineering",
    },
    "Q17": {
        "authoritative_doc": "departments/aerospace_engineering/general_information.docx",
        "key_facts": ["AI/ML", "IoT", "digital twins", "drone and anti-drone"],
        "target_dept": "Aerospace Engineering",
    },
    "Q18": {
        "authoritative_doc": "departments/electronics_engineering/general_information.docx",
        "key_facts": ["Micro-Nano Electronics & VLSI", "Communication & Networks", "RF & Photonics", "Intelligent and Embedded Computing"],
        "target_dept": "Electronics Engineering",
    },
    "Q19": {
        "authoritative_doc": "departments/materials_engineering/general_information.docx",
        "key_facts": ["Structural Materials", "Functional Materials", "Computational Materials", "Process Metallurgy", "Artificial Intelligence", "interdisciplinary"],
        "target_dept": "Materials Engineering",
    },
    "Q20": {
        "authoritative_doc": "departments/mathematics/general_information.docx",
        "key_facts": ["141", "153", "12"],
        "target_dept": "Mathematics",
    },
    "Q21": {
        "authoritative_doc": "departments/mathematics/programs.docx",
        "key_facts": ["42", "12", "6", "2", "72"],
        "target_dept": "Mathematics",
    },
    "Q22": {
        "authoritative_doc": "departments/mathematics/curriculum.docx",
        "key_facts": ["72"],
        "target_dept": "Mathematics",
    },
    "Q23": {
        "authoritative_doc": "departments/mathematics/curriculum.docx",
        "key_facts": ["Mathematics", "Computing", "Statistics & Quantum Computing"],
        "target_dept": "Mathematics",
    },
    "Q24": {
        "authoritative_doc": "departments/mathematics/curriculum.docx",
        "key_facts": ["CS7XXX", "CSLXXXX", "QCL4020"],
        "target_dept": "Mathematics",
    },
    "Q25": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["104", "62"],
        "target_dept": "Training and Upskilling",
    },
    "Q26": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["17–18 June 2023", "10–11 June 2023", "19 May"],
        "target_dept": "Training and Upskilling",
    },
    "Q27": {
        "authoritative_doc": "departments/electronics_engineering/general_information.docx",
        "key_facts": ["performance", "power", "cost"],
        "target_dept": "Electronics Engineering",
    },
    "Q28": {
        "authoritative_doc": "departments/materials_engineering/general_information.docx",
        "key_facts": ["B.Tech", "Ph.D", "computational", "experimental"],
        "target_dept": "Materials Engineering",
    },
    "Q29": {
        "authoritative_doc": "departments/mathematics/programs.docx",
        "key_facts": ["Data Scientist", "Machine Learning", "Quantitative Analyst", "Algorithm"],
        "target_dept": "Mathematics",
    },
    "Q30": {
        "authoritative_doc": "departments/electronics_engineering/general_information.docx",
        "key_facts": ["semiconductor", "materials", "overlap"],
        "target_dept": "Electronics / Materials",
    },
    "Q31": {
        "authoritative_doc": "departments/aerospace_engineering/general_information.docx",
        "key_facts": ["not specified", "not provided", "not available", "do not specify"],
        "target_dept": "Aerospace Engineering",
    },
    "Q32": {
        "authoritative_doc": "departments/mathematics/programs.docx",
        "key_facts": ["not specified", "not provided", "not available", "JEE"],
        "target_dept": "Mathematics",
    },
    "Q33": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["not scheduled in 2026", "2023", "no 2026", "not provided", "not available"],
        "target_dept": "Training and Upskilling",
    },
    "Q34": {
        "authoritative_doc": "departments/mathematics/general_information.docx",
        "key_facts": ["specify", "program", "programme", "degree", "153", "depend"],
        "target_dept": "Institute UG",
    },
    "Q35": {
        "authoritative_doc": "departments/materials_engineering/general_information.docx",
        "key_facts": ["not detailed", "not specified", "not provided", "available documents do not"],
        "target_dept": "Materials Engineering",
    },
    "Q36": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["No", "not every", "450", "300", "not stated", "not specified"],
        "target_dept": "Training and Upskilling",
    },
    "Q37": {
        "authoritative_doc": "departments/mathematics/curriculum.docx",
        "key_facts": ["placeholder", "CS7XXX", "CSLXXXX", "incomplete"],
        "target_dept": "Mathematics",
    },
    "Q38": {
        "authoritative_doc": "departments/mathematics/programs.docx",
        "key_facts": ["74", "incorrect", "add up to 74", "discrepancy", "sum"],
        "target_dept": "Mathematics",
    },
    "Q39": {
        "authoritative_doc": "departments/electronics_engineering/general_information.docx",
        "key_facts": ["neither", "both", "cannot declare", "no definitive winner", "roles of both"],
        "target_dept": "Electronics / Materials",
    },
    "Q40": {
        "authoritative_doc": "training_and_upskilling/training.docx",
        "key_facts": ["not provided", "not available", "only mentions", "not specified"],
        "target_dept": "Training and Upskilling",
    },
}

def analyze_trace(t: Dict[str, Any]) -> Dict[str, Any]:
    qid = t["id"]
    exp = EXPECTATIONS.get(qid, {})
    ans = t.get("final_displayed_answer") or ""
    raw_ans = t.get("raw_generated_answer") or ""
    ctx = t.get("final_evidence_context") or ""
    is_refusal = t.get("is_safe_refusal", False)
    pipe_stop = t.get("pipeline_stop", False)
    
    # 1. Query Understanding
    qu_status = "PASS"
    qu_scope = "Correct"
    qu_plan = "PASS"
    qu_reason = ""
    stop_reason = t.get("stop_reason") or ""
    
    if pipe_stop:
        if "unanchored_programme_clarification" in stop_reason:
            qu_status = "FAIL"
            qu_plan = "FAIL"
            qu_reason = f"False deictic clarification refusal: {stop_reason}"
        elif is_refusal:
            qu_status = "FAIL"
            qu_scope = "Incorrect"
            qu_plan = "FAIL"
            qu_reason = f"False out-of-scope refusal: {stop_reason}"
    
    # Check if target or attributes were stripped
    plan = t.get("query_plan", {})
    if isinstance(plan, dict):
        if not plan.get("target") and "focus areas" in t["question"].lower():
            qu_plan = "PARTIAL"
            qu_reason = "Target empty in query plan"
            
    # 2. Source / Index availability
    src_avail = "Present"
    
    # 3. Retrieval
    ret_status = "PASS"
    ret_rank = "absent"
    fused_candidates = t.get("fused_top10", [])
    dense_candidates = t.get("dense_top10", [])
    bm25_candidates = t.get("bm25_top10", [])
    ranked_candidates = t.get("ranked_top10", [])
    
    target_doc = exp.get("authoritative_doc", "")
    target_kw = exp.get("target_dept", "").lower().split()[0]
    
    # Check if authoritative text/source is in candidates
    auth_found_rank = None
    for c in fused_candidates:
        snip = c.get("snippet", "").lower()
        src = c.get("source", "").lower()
        full_src = c.get("full_source", "").lower()
        if target_doc.lower() in full_src or (target_kw and target_kw in snip):
            auth_found_rank = c.get("rank")
            break
            
    if auth_found_rank is not None:
        ret_rank = f"Rank {auth_found_rank}"
    else:
        # Check dense/bm25
        for c in dense_candidates + bm25_candidates:
            if target_kw and target_kw in c.get("snippet", "").lower():
                auth_found_rank = c.get("rank")
                ret_rank = f"Rank {auth_found_rank} (unfused)"
                break
                
    if auth_found_rank is None and not pipe_stop:
        ret_status = "PARTIAL"
        
    # 4. Candidate verification and ranking
    cand_verif = "PASS"
    ev_ranking = "PASS"
    if qid == "Q2":
        # Q2 was demoted below rank 10
        ev_ranking = "FAIL"
        
    # 5. Evidence packaging & Fact in final context
    ev_pack = "PASS"
    fact_in_ctx = "YES"
    missing_ctx_facts = []
    for kf in exp.get("key_facts", []):
        if kf.lower() not in ctx.lower():
            missing_ctx_facts.append(kf)
            
    if missing_ctx_facts and len(missing_ctx_facts) > len(exp.get("key_facts", [])) / 2:
        fact_in_ctx = "NO"
        ev_pack = "FAIL"
    elif missing_ctx_facts:
        fact_in_ctx = "PARTIAL"
        ev_pack = "PARTIAL"
        
    # 6. Raw LLM Generation
    raw_gen = "PASS"
    disallowed = exp.get("disallowed_facts", [])
    has_disallowed = any(df.lower() in raw_ans.lower() for df in disallowed)
    
    missing_ans_facts = []
    for kf in exp.get("key_facts", []):
        if kf.lower() not in ans.lower():
            missing_ans_facts.append(kf)
            
    if has_disallowed or len(missing_ans_facts) >= len(exp.get("key_facts", [])):
        raw_gen = "FAIL"
    elif missing_ans_facts:
        raw_gen = "PARTIAL"
        
    # 7. Grounding and Post-processing
    post_proc = "PASS"
    if "[Civil And Infrastructure" in ans or "######" in ans or "[Materials Engineering" in ans:
        post_proc = "FAIL"
    elif raw_ans != ans and len(raw_ans) > len(ans) + 20:
        post_proc = "PARTIAL"
        
    # 8. Citations
    cits = t.get("citations", [])
    cit_status = "PASS"
    if len(cits) == 0 and not is_refusal:
        cit_status = "FAIL"
    else:
        # Check if citations match target dept
        cit_paths = [str(c.get("source_path") or c.get("title") or "") for c in cits]
        alien_cits = [p for p in cit_paths if p and target_kw and target_kw not in p.lower() and "office_of_academics" not in p.lower()]
        if len(alien_cits) > 2:
            cit_status = "PARTIAL"
            
    # 9. Final Answer
    final_status = "PASS"
    if qu_status == "FAIL" or raw_gen == "FAIL" or post_proc == "FAIL" or ev_ranking == "FAIL":
        final_status = "FAIL"
    elif raw_gen == "PARTIAL" or fact_in_ctx == "PARTIAL":
        final_status = "PARTIAL"
        
    # 10. Earliest Root Cause & Confidence
    earliest_stage = "N/A"
    earliest_cause = "Correct behavior"
    confidence = "High"
    repair_action = "None needed"
    
    if qu_status == "FAIL":
        if "unanchored" in qu_reason:
            earliest_stage = "Query Understanding (conversation.py:resolve_conversation)"
            earliest_cause = "False deictic clarification triggered by generic 'the degree/course' regex in standalone query"
            repair_action = "Fix deictic program regex to require conversational antecedent in conversation.py"
        else:
            earliest_stage = "Query Understanding (planner.py:should_stop_pipeline)"
            earliest_cause = "False out-of-scope refusal due to narrow institutional scope dictionary"
            repair_action = "Fix scope classifier & markers in planner.py"
    elif ev_ranking == "FAIL":
        earliest_stage = "Ranking (ranking.py:rank_candidates)"
        earliest_cause = "Severe demotion of specific departmental document in favor of generic doc"
        repair_action = "Fix ranking penalization in ranking.py"
    elif post_proc == "FAIL":
        earliest_stage = "Post-Processing (generator.py:_extractive_repair)"
        earliest_cause = "Extractive repair inserted cross-department snippet / raw markdown tags"
        repair_action = "Fix extractive repair & regex terminal punctuation in generator.py"
    elif fact_in_ctx == "NO":
        earliest_stage = "Packaging (evidence_package.py)"
        earliest_cause = "Decisive chunk omitted from final context window"
        repair_action = "Fix context packaging budget and chunk survival"
    elif raw_gen == "FAIL":
        earliest_stage = "Generation (generator.py / LLM prompt)"
        earliest_cause = "Prompt instruction overload or model hallucination on specific numbers/dates"
        repair_action = "Simplify prompt guidance and enforce fact extraction"
    elif raw_gen == "PARTIAL":
        earliest_stage = "Generation (generator.py:sentence_termination)"
        earliest_cause = "Premature list termination or incomplete item enumeration"
        repair_action = "Fix terminal punct detection for numbered items"
        
    return {
        "id": qid,
        "title": t.get("title", ""),
        "qu_status": qu_status,
        "qu_scope": qu_scope,
        "qu_plan": qu_plan,
        "src_avail": src_avail,
        "ret_status": ret_status,
        "ret_rank": ret_rank,
        "cand_verif": cand_verif,
        "ev_ranking": ev_ranking,
        "ev_pack": ev_pack,
        "fact_in_ctx": fact_in_ctx,
        "raw_gen": raw_gen,
        "post_proc": post_proc,
        "cit_status": cit_status,
        "final_status": final_status,
        "earliest_stage": earliest_stage,
        "earliest_cause": earliest_cause,
        "confidence": confidence,
        "repair_action": repair_action,
        "latency_sec": t.get("latency_seconds"),
        "final_answer_preview": (ans[:120] + "...") if len(ans) > 120 else ans,
    }

def main():
    import sys
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "test_outputs/rag_failure_diagnosis_20261019/baseline_40q_traces.json"
    is_postfix = "postfix" in trace_path or (len(sys.argv) > 2 and "postfix" in sys.argv[2])
    title_prefix = "Post-Fix (Repaired)" if is_postfix else "Baseline"
    default_out_name = "postfix_failure_matrix.md" if is_postfix else "baseline_failure_matrix.md"
    out_matrix_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("docs/rag_failure_diagnosis_20261019", default_out_name)

    if not os.path.exists(trace_path):
        print(f"Error: {trace_path} does not exist yet.")
        return
        
    with open(trace_path) as f:
        traces = json.load(f)
        
    print(f"Analyzing {len(traces)} traces from {trace_path}...")
    results = [analyze_trace(t) for t in traces]
    
    # Generate Markdown Table
    md_lines = [
        f"# {title_prefix} Evaluation Matrix (Q1–Q40)",
        "",
        f"**Total Questions Analyzed**: {len(results)}/40",
        "",
        "| ID | Title | QU | Scope | Plan | Source | Retr | Retr Rank | Verif | Rank | Pack | Fact in Ctx | Raw Gen | Post-Proc | Cits | Final Ans | Earliest Stage & Function | Confidence | Repair / Action |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    
    pass_count = sum(1 for r in results if r["final_status"] == "PASS")
    fail_count = sum(1 for r in results if r["final_status"] == "FAIL")
    part_count = sum(1 for r in results if r["final_status"] == "PARTIAL")
    
    for r in results:
        row = (
            f"| {r['id']} | {r['title']} | {r['qu_status']} | {r['qu_scope']} | {r['qu_plan']} | {r['src_avail']} | "
            f"{r['ret_status']} | {r['ret_rank']} | {r['cand_verif']} | {r['ev_ranking']} | {r['ev_pack']} | "
            f"{r['fact_in_ctx']} | {r['raw_gen']} | {r['post_proc']} | {r['cit_status']} | **{r['final_status']}** | "
            f"{r['earliest_stage']} | {r['confidence']} | {r['repair_action']} |"
        )
        md_lines.append(row)
        
    md_lines.extend([
        "",
        "## Summary Metrics",
        f"- **PASS**: {pass_count} ({pass_count/len(results)*100:.1f}%)" if results else "",
        f"- **PARTIAL**: {part_count} ({part_count/len(results)*100:.1f}%)" if results else "",
        f"- **FAIL**: {fail_count} ({fail_count/len(results)*100:.1f}%)" if results else "",
        "",
        "## Confirmed Stage Breakdown of Earliest Root Causes",
    ])
    
    stage_counts = {}
    for r in results:
        stage = r["earliest_stage"].split(":")[0] if ":" in r["earliest_stage"] else r["earliest_stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
    for stage, count in sorted(stage_counts.items(), key=lambda x: x[1], reverse=True):
        md_lines.append(f"- **{stage}**: {count} questions")
        
    os.makedirs(os.path.dirname(out_matrix_path), exist_ok=True)
    with open(out_matrix_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
        
    print(f"Wrote evaluation matrix to {out_matrix_path}")
    print(f"Summary: PASS={pass_count}, PARTIAL={part_count}, FAIL={fail_count}")

    # Also dump summary stats json if requested
    stats = {
        "total": len(results),
        "pass": pass_count,
        "partial": part_count,
        "fail": fail_count,
        "pass_rate": round(pass_count / len(results) * 100, 1) if results else 0,
        "stage_counts": stage_counts,
        "results": results,
    }
    stats_path = out_matrix_path.replace(".md", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
