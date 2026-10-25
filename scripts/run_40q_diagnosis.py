#!/usr/bin/env python3
"""Run the frozen 40-question RAG diagnosis suite against the active IIT Jodhpur system.

Captures stage-by-stage traces for:
- Query understanding & planning
- Dense & lexical retrieval
- Candidate verification & ranking
- Evidence packaging & exact final LLM context
- Raw LLM generation & post-processing
- Streamlit citation rendering & final displayed response
- Stage latencies & root causes
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest

DIAGNOSTIC_QUESTIONS = [
    # Group A: Basic questions (Q1–Q10)
    {
        "id": "Q1",
        "group": "A",
        "title": "Aerospace vision",
        "question": "What is the vision of IIT Jodhpur's Aerospace Engineering Department? Explain its three main aims in your own words.",
    },
    {
        "id": "Q2",
        "group": "A",
        "title": "Drone and anti-drone programme",
        "question": "What is the proposed M.Tech. programme in Drone and Anti-Drone Technologies, why is it needed, and from which academic year does the document propose starting it?",
    },
    {
        "id": "Q3",
        "group": "A",
        "title": "Aerospace mission",
        "question": "What are the three main missions of the Aerospace Engineering Department? Cover education, research, and infrastructure.",
    },
    {
        "id": "Q4",
        "group": "A",
        "title": "Electronics focus areas",
        "question": "List the four major focus areas of the Department of Electronics Engineering.",
    },
    {
        "id": "Q5",
        "group": "A",
        "title": "Electronics research",
        "question": "Name five research areas pursued by the Department of Electronics Engineering, including both established and emerging technologies.",
    },
    {
        "id": "Q6",
        "group": "A",
        "title": "Materials Engineering",
        "question": "When was the Department of Materials Engineering established, which degree programmes does it offer, and what are its four major technical streams?",
    },
    {
        "id": "Q7",
        "group": "A",
        "title": "Materials Engineering minor options",
        "question": "Which two minor options are explicitly listed for B.Tech. students in Materials Engineering?",
    },
    {
        "id": "Q8",
        "group": "A",
        "title": "Training and Upskilling",
        "question": "What is the role of the Office of Vocational Training & Upskilling (VTU) at IIT Jodhpur?",
    },
    {
        "id": "Q9",
        "group": "A",
        "title": "Mathematics and Computing",
        "question": "How does the B.S. in Mathematics and Computing combine mathematics with computing, and how is its stated approach different from a conventional computing degree?",
    },
    {
        "id": "Q10",
        "group": "A",
        "title": "Career pathways",
        "question": "Give four career pathways specifically listed for graduates of the B.S. in Mathematics and Computing.",
    },

    # Group B: Medium and multi-attribute questions (Q11–Q15)
    {
        "id": "Q11",
        "group": "B",
        "title": "GeM training",
        "question": "Describe the Specialized Training in Government e-Marketplace (GeM). Include its dates, duration, mode, intended audience, and three hands-on modules.",
    },
    {
        "id": "Q12",
        "group": "B",
        "title": "GFR training",
        "question": "Who was the GFR 2017 training intended for? Give its registration date, programme dates, duration, daily schedule, mode, and the main topics covered.",
    },
    {
        "id": "Q13",
        "group": "B",
        "title": "PMKVY training",
        "question": "For the four job roles listed under PMKVY 4.0 Phase I, provide participant and certification counts. Include the training duration wherever the document explicitly specifies it.",
    },
    {
        "id": "Q14",
        "group": "B",
        "title": "Materials Engineering curriculum",
        "question": "How does the B.Tech. in Materials Engineering combine its four traditional technical streams with newer computational and interdisciplinary areas?",
    },
    {
        "id": "Q15",
        "group": "B",
        "title": "Mathematics learning outcomes",
        "question": "What mathematical capabilities are students expected to develop, and which symbolic-computation tools are explicitly named in the Mathematics programme document?",
    },

    # Group C: Hard and cross-document questions (Q16–Q30)
    {
        "id": "Q16",
        "group": "C",
        "title": "DADT learning outcomes",
        "question": "According to the DADT document, what should students learn about physical principles, subsystem trade-offs, mission requirements, hands-on practical work, and future careers? Organize your answer by learning outcome.",
    },
    {
        "id": "Q17",
        "group": "C",
        "title": "Aerospace objectives versus DADT outcomes",
        "question": "The Aerospace documents discuss AI/ML, IoT, and digital twins, while the DADT document discusses drone and anti-drone systems. Which items appear in the general Aerospace objectives, and which are explicitly described as DADT learning outcomes?",
    },
    {
        "id": "Q18",
        "group": "C",
        "title": "Electronics focus-area mapping",
        "question": "Map each of the four Electronics Engineering focus areas to the relevant technical topics in the documents. Include examples such as chip design, wireless or optical communications, RF systems, photonics, and embedded computing where supported. If the source does not explicitly establish a one-to-one mapping between a focus area and a specific topic, say so rather than inventing the mapping.",
    },
    {
        "id": "Q19",
        "group": "C",
        "title": "Materials departmental versus interdisciplinary specialization",
        "question": "For B.Tech. Materials Engineering, distinguish the four departmental specialization streams from the listed interdisciplinary specializations. Where does Artificial Intelligence appear, and is it listed as one of the four departmental streams?",
    },
    {
        "id": "Q20",
        "group": "C",
        "title": "Institute UG credit structure",
        "question": "According to the Mathematics General Information document's Institute UG structure table, what is the total graded credit count, the total credit count for the degree, and the difference between them? Identify the non-graded components that account for that difference.",
    },
    {
        "id": "Q21",
        "group": "C",
        "title": "B.S. Mathematics and Computing credits",
        "question": "Report the minimum credits listed for Program Core, Program Elective, Open Elective, Industry-Academia Summer Internship, and Thesis. What total does the document itself state?",
    },
    {
        "id": "Q22",
        "group": "C",
        "title": "Comparing two 72-credit totals",
        "question": "The Mathematics Curriculum document lists a grand total of 72 credits, while the B.S. in Mathematics and Computing programme document also lists 72 total credits. Are the documents sufficient to conclude that these are the same curriculum? Explain which programme context is stated in each source.",
    },
    {
        "id": "Q23",
        "group": "C",
        "title": "Mathematics elective groups",
        "question": "Name one course from each of the three elective groups—Mathematics, Computing, and Statistics & Quantum Computing—in the Mathematics Curriculum document. Preserve the course names as written.",
    },
    {
        "id": "Q24",
        "group": "C",
        "title": "Course codes",
        "question": "What course codes are shown for Advanced Algorithms, Computer Vision, and Quantum Computing? Do not fill in any missing parts of a code.",
    },
    {
        "id": "Q25",
        "group": "C",
        "title": "PMKVY aggregate calculation",
        "question": "Add the enrolment counts and certification counts across all four PMKVY job roles. Do your totals agree with the reported aggregate impact?",
    },
    {
        "id": "Q26",
        "group": "C",
        "title": "GeM versus GFR",
        "question": "Compare the GeM and GFR 2017 training programmes. Which programme ran on 10–11 June 2023, which ran on 17–18 June 2023, and which one has 19 May as its registration date? Also compare their stated audiences and coverage.",
    },
    {
        "id": "Q27",
        "group": "C",
        "title": "Electronics graduate attributes",
        "question": "What does the Electronics Engineering document say about designing electronic systems under performance, power, and cost constraints? Include the other professional attributes the graduates are expected to demonstrate.",
    },
    {
        "id": "Q28",
        "group": "C",
        "title": "Materials B.Tech. versus Ph.D.",
        "question": "Compare the B.Tech. and Ph.D. descriptions in Materials Engineering. How do their stated learning or training approaches differ, and what role do experimental and computational methods play?",
    },
    {
        "id": "Q29",
        "group": "C",
        "title": "Mathematics objectives and career pathways",
        "question": "Explain how the B.S. in Mathematics and Computing connects mathematical foundations, algorithmic thinking, industry readiness, and research readiness. Give at least four relevant industry pathways from the source.",
    },
    {
        "id": "Q30",
        "group": "C",
        "title": "Electronics versus Materials Engineering",
        "question": "Compare Electronics Engineering and Materials Engineering research themes. Identify at least one meaningful overlap and three areas that distinguish the departments, using only the uploaded documents.",
    },

    # Group D: Ambiguity, missing evidence and difficult validation (Q31–Q40)
    {
        "id": "Q31",
        "group": "D",
        "title": "Missing DADT admission details",
        "question": "What are the exact admission eligibility criteria, application deadline, and tuition fees for the proposed M.Tech. in Drone and Anti-Drone Technologies?",
    },
    {
        "id": "Q32",
        "group": "D",
        "title": "B.S. Mathematics and Computing admissions",
        "question": "What are the minimum admission marks, application deadline, and tuition fees for the B.S. in Mathematics and Computing?",
    },
    {
        "id": "Q33",
        "group": "D",
        "title": "Historical versus current training dates",
        "question": "When are the next GeM and GFR training programmes scheduled in 2026? Use the uploaded documents to establish the dates.",
    },
    {
        "id": "Q34",
        "group": "D",
        "title": "Ambiguous graduation-credit query",
        "question": "How many credits do I need to graduate from IIT Jodhpur? Give the exact number.",
    },
    {
        "id": "Q35",
        "group": "D",
        "title": "Materials M.Tech. eligibility",
        "question": "What is the minimum qualifying degree, required marks, course duration, and admission procedure for M.Tech. in Materials Engineering?",
    },
    {
        "id": "Q36",
        "group": "D",
        "title": "PMKVY duration false premise",
        "question": "Does every PMKVY job role have a duration of 450 hours plus 150 hours of on-the-job training? Give the evidence for each role and distinguish missing information from confirmed durations.",
    },
    {
        "id": "Q37",
        "group": "D",
        "title": "Incomplete Mathematics course codes",
        "question": "Give the complete course codes for all elective courses in the Mathematics Curriculum document. If the complete code is not provided, tell me which entries contain placeholders or incomplete codes.",
    },
    {
        "id": "Q38",
        "group": "D",
        "title": "Arithmetic trap",
        "question": "The B.S. in Mathematics and Computing table lists 42 core credits, 12 programme elective credits, 6 open elective credits, 2 internship credits, and 12 thesis credits. These add up to 72, correct?",
    },
    {
        "id": "Q39",
        "group": "D",
        "title": "Semiconductor chip-design recommendation",
        "question": "Which is the better department for someone interested in semiconductor chip design: Electronics Engineering or Materials Engineering? Give a definitive winner based only on these documents.",
    },
    {
        "id": "Q40",
        "group": "D",
        "title": "Incomplete Training programme information",
        "question": "Give the eligibility criteria, fees, duration, registration deadline, and certificate details for the “Mastering Technical English: Academic & Professional Excellence” programme.",
    },
]


def serialize_candidate(c: Any, rank: int | None = None) -> dict[str, Any]:
    if c is None:
        return {}
    source = str(getattr(c, "source", None) or getattr(getattr(c, "document", None), "metadata", {}).get("source", "") or "")
    score = getattr(c, "final_score", None) or getattr(c, "score", None)
    doc = getattr(c, "document", None)
    content = ""
    if doc is not None:
        content = getattr(doc, "page_content", "") or ""
    elif hasattr(c, "page_content"):
        content = getattr(c, "page_content", "") or ""

    prov = getattr(c, "provenance", None)
    rrf_score = getattr(prov, "rrf_score", None) if prov else None
    
    return {
        "rank": rank,
        "source": source.replace("\\", "/").split("/")[-1],
        "full_source": source.replace("\\", "/"),
        "score": round(float(score), 4) if score is not None else None,
        "rrf_score": round(float(rrf_score), 5) if rrf_score is not None else None,
        "snippet": content[:300].strip(),
    }


def serialize_query_plan(qp: Any) -> dict[str, Any]:
    if qp is None:
        return {}
    if isinstance(qp, dict):
        return {
            "route": qp.get("route"),
            "intents": qp.get("intents"),
            "target": qp.get("target"),
            "entity": qp.get("entity"),
            "attributes": qp.get("attributes"),
        }
    return {
        "route": str(getattr(qp, "route", "")),
        "target": str(getattr(qp, "target", "")),
        "entity": str(getattr(qp, "entity", "")),
        "intents": [str(i) for i in getattr(qp, "intents", [])],
        "attributes": list(getattr(qp, "attributes", [])),
    }


def run_diagnosis(output_dir: Path, max_questions: int | None = None, output_filename: str = "baseline_40q_traces.json") -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_file = output_dir / output_filename
    
    existing_results: dict[str, dict[str, Any]] = {}
    if traces_file.exists():
        try:
            with open(traces_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    for item in loaded:
                        existing_results[item["id"]] = item
                elif isinstance(loaded, dict):
                    existing_results = loaded
        except Exception as e:
            print(f"Warning: Could not read existing traces: {e}")

    results = []
    questions_to_run = DIAGNOSTIC_QUESTIONS[:max_questions] if max_questions else DIAGNOSTIC_QUESTIONS

    print(f"=== Starting 40-Question RAG Baseline Diagnostic Run ===")
    print(f"Output directory: {output_dir}")
    print(f"Total questions: {len(questions_to_run)}")

    for idx, q_info in enumerate(questions_to_run, 1):
        q_id = q_info["id"]
        q_text = q_info["question"]
        q_title = q_info["title"]
        q_group = q_info["group"]

        if q_id in existing_results and existing_results[q_id].get("completed"):
            print(f"\n[{idx}/{len(questions_to_run)}] {q_id} ({q_title}) - Already completed, loading cached trace.")
            results.append(existing_results[q_id])
            continue

        print(f"\n================================================================================")
        print(f"[{idx}/{len(questions_to_run)}] Running {q_id} [Group {q_group}: {q_title}]")
        print(f"Question: {q_text}")
        print(f"================================================================================")

        t_start = time.perf_counter()

        req = ChatTurnRequest(
            question=q_text,
            chat_history=(),  # fresh conversation for each diagnostic question
            session_id=f"diag_baseline_{q_id.lower()}",
            institution_id="iitj",
        )

        try:
            resp = handle_chat_turn(req)
            total_time = time.perf_counter() - t_start
            diag = resp.diagnostics or {}
            raw_state = diag.get("raw_state") or diag

            # Parse query understanding & planning
            qp = raw_state.get("query_plan") or diag.get("query_plan")
            qp_serialized = serialize_query_plan(qp)
            resolved_q = str(raw_state.get("resolved_question") or diag.get("resolved_question") or q_text)
            retrieval_queries = list(raw_state.get("retrieval_queries") or diag.get("retrieval_queries") or [resolved_q])

            # Scope & guard
            pipeline_stop = bool(raw_state.get("pipeline_stop"))
            stop_reason = str(raw_state.get("stop_reason") or "")
            query_scope = str(raw_state.get("query_scope") or "")
            query_scope_conf = float(raw_state.get("query_scope_confidence") or 1.0)
            query_scope_reason = str(raw_state.get("query_scope_reason") or "")

            # Retrieval candidates
            ret_results = raw_state.get("retrieval_results") or ()
            dense_candidates = []
            bm25_candidates = []
            if len(ret_results) > 0 and isinstance(ret_results[0], (list, tuple)):
                dense_candidates = [serialize_candidate(c, r) for r, c in enumerate(ret_results[0][:10], 1)]
            if len(ret_results) > 1 and isinstance(ret_results[1], (list, tuple)):
                bm25_candidates = [serialize_candidate(c, r) for r, c in enumerate(ret_results[1][:10], 1)]

            # Fused, ranked, verified, rejected
            fused_cands = raw_state.get("fused_candidates") or diag.get("fused_candidates") or []
            verified_cands = raw_state.get("verified_candidates") or diag.get("verified_candidates") or []
            ranked_cands = raw_state.get("ranked_candidates") or diag.get("ranked_candidates") or []
            rejected_cands = raw_state.get("rejected_candidates") or diag.get("rejected_candidates") or []

            fused_candidates = [serialize_candidate(c, r) for r, c in enumerate(fused_cands[:10], 1)]
            verified_candidates = [serialize_candidate(c, r) for r, c in enumerate(verified_cands[:10], 1)]
            ranked_candidates = [serialize_candidate(c, r) for r, c in enumerate(ranked_cands[:10], 1)]
            rejected_candidates = [serialize_candidate(c) for c in rejected_cands[:10]]

            # Fallback to section_a_candidates if raw candidates were not accessible
            if not fused_candidates and diag.get("section_a_candidates"):
                fused_candidates = diag.get("section_a_candidates")[:10]

            # Evidence context
            package = raw_state.get("evidence_package")
            evidence_context = str(
                raw_state.get("final_evidence_context")
                or (getattr(package, "context", "") if package else "")
                or raw_state.get("context")
                or ""
            )
            package_status = str(raw_state.get("evidence_package_status") or "")

            # LLM generation & post-processing
            gen = raw_state.get("answer_generation")
            raw_answer = ""
            if gen is not None:
                raw_answer = str(getattr(gen, "answer", "") or getattr(gen, "content", "") or "")
            if not raw_answer:
                raw_answer = resp.answer

            grounding_assessment = raw_state.get("grounding_assessment") or raw_state.get("answer_grounding")
            grounding_status = str(raw_state.get("answer_grounding_status") or "")
            guard_status = str(raw_state.get("answer_guard_status") or "")
            guard_reason = str(raw_state.get("answer_guard_reason") or "")

            item_trace = {
                "id": q_id,
                "group": q_group,
                "title": q_title,
                "question": q_text,
                "history": [],
                "completed": True,
                "latency_seconds": round(total_time, 2),
                
                # Query understanding & planning
                "resolved_question": resolved_q,
                "retrieval_queries": retrieval_queries,
                "query_plan": qp_serialized,
                "is_multi_intent": bool(diag.get("is_multi_intent")),
                "intent_questions": list(diag.get("intent_questions") or []),
                "intent_count": int(diag.get("intent_count") or 1),
                
                # Guard / Scope
                "pipeline_stop": pipeline_stop,
                "stop_reason": stop_reason,
                "query_scope": query_scope,
                "query_scope_confidence": query_scope_conf,
                "query_scope_reason": query_scope_reason,

                # Retrieval
                "dense_top10": dense_candidates,
                "bm25_top10": bm25_candidates,
                "fused_top10": fused_candidates,
                "ranked_top10": ranked_candidates,
                "verified_count": len(diag.get("verified_candidates") or []),
                "rejected_count": len(diag.get("rejected_candidates") or []),
                "rejected_sample": rejected_candidates[:5],

                # Evidence & context
                "context_char_count": len(evidence_context),
                "final_evidence_context": evidence_context,
                "evidence_package_status": package_status,

                # Generation & Answering
                "raw_generated_answer": raw_answer,
                "final_displayed_answer": resp.answer,
                "formatted_response": resp.formatted_response,
                "is_safe_refusal": resp.is_safe_refusal,
                "evidence_status": resp.evidence_status,
                "grounding_status": grounding_status,
                "guard_status": guard_status,
                "guard_reason": guard_reason,
                "citations": resp.sources,
            }

            print(f"  Latency: {total_time:.2f}s | Citations: {len(resp.sources)} | Refusal: {resp.is_safe_refusal}")
            print(f"  Resolved: '{resolved_q}'")
            print(f"  Context chars: {len(evidence_context)}")
            print(f"  Answer preview: {resp.answer[:140]}...")

            results.append(item_trace)
            existing_results[q_id] = item_trace

            # Save incremental progress
            with open(traces_file, "w", encoding="utf-8") as f:
                json.dump(list(existing_results.values()), f, indent=2)

        except Exception as exc:
            print(f"ERROR processing {q_id}: {exc}")
            import traceback
            traceback.print_exc()
            error_trace = {
                "id": q_id,
                "group": q_group,
                "title": q_title,
                "question": q_text,
                "completed": False,
                "error": str(exc),
                "latency_seconds": round(time.perf_counter() - t_start, 2),
            }
            results.append(error_trace)
            existing_results[q_id] = error_trace
            with open(traces_file, "w", encoding="utf-8") as f:
                json.dump(list(existing_results.values()), f, indent=2)

    print(f"\nCompleted all {len(results)} diagnostic questions. Saved to {traces_file}")
    return results


if __name__ == "__main__":
    out_dir = WORKSPACE_ROOT / "test_outputs" / "rag_failure_diagnosis_20261019"
    filename = sys.argv[1] if len(sys.argv) > 1 else "baseline_40q_traces.json"
    run_diagnosis(out_dir, output_filename=filename)
