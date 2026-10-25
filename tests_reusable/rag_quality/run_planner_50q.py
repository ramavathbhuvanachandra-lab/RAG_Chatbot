"""50-Scenario Query Understanding and Planning Benchmark Harness and Runner.

Evaluates 50 realistic college RAG scenarios covering:
- Short direct questions
- Long narrative eligibility questions
- Conversational follow-ups
- Linked answer facets
- Independent multi-intent questions
- Programme changes within a sentence
- Applicant categories and numerical thresholds
- Minimums, maximums, units and dates
- Negation and exceptions
- AND/OR qualifying paths
- Facilities, fees, governance and institutional policies
- Procedures, consequences and deadlines
- Hindi/Hinglish student wording
- Ambiguity and out-of-scope requests
- Adversarial constraints

Produces:
- planner_50q_results.json
- planner_50q_report.txt
- planner_50q_failures.txt
- manifest.json
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_platform.core.conversation.conversation import resolve_conversation
from ai_platform.core.query.planner import (
    build_query_plan,
    plan_to_frame,
    retrieval_queries_from_plan,
    should_stop_pipeline,
)
from ai_platform.core.query.lsqc import build_lossless_query

DEFAULT_SCENARIOS_FILE = ROOT / "tests_reusable" / "rag_quality" / "planner_50q_scenarios.json"
DEFAULT_OUTPUT_BASE = ROOT / "test_outputs" / "rag_quality"


def evaluate_plan_record(item: dict[str, Any], plan_data: dict[str, Any]) -> dict[str, Any]:
    expected = item["expected"]
    failures: list[str] = []

    # 1. Scope / Domain check
    actual_domain = plan_data["domain_decision"]
    if actual_domain != expected["domain_decision"]:
        failures.append(f"Domain mismatch: expected '{expected['domain_decision']}', got '{actual_domain}'")

    actual_stop = plan_data["pipeline_stop"]
    if actual_stop != expected["pipeline_stop"]:
        failures.append(f"Pipeline stop mismatch: expected {expected['pipeline_stop']}, got {actual_stop}")

    # 2. Multi-intent classification check
    actual_multi = plan_data["is_multi_intent"]
    if actual_multi != expected["is_multi_intent"]:
        failures.append(f"Multi-intent mismatch: expected {expected['is_multi_intent']}, got {actual_multi} (intents count={plan_data['intent_count']})")

    # 3. Target entity check
    expected_target = expected.get("target")
    actual_target = plan_data.get("target")
    if expected_target is not None:
        def _norm_tgt(s: str) -> str:
            return re.sub(r"[.\s]", "", s).lower()

        all_targets = [str(actual_target or "")] + [str(i.get("target") or "") for i in plan_data.get("intents", [])]
        exp_parts = [p.strip() for p in re.split(r"\band\b", expected_target, flags=re.I)]
        if len(exp_parts) > 1:
            target_match = any(all(_norm_tgt(p) in _norm_tgt(t) for p in exp_parts) for t in all_targets if t)
            # In multi-intent with multiple degree targets, verify both are present across intents
            if not target_match:
                found_parts = [any(_norm_tgt(p) in _norm_tgt(t) for t in all_targets if t) for p in exp_parts]
                target_match = all(found_parts)
        else:
            target_match = any(_norm_tgt(expected_target) in _norm_tgt(t) or _norm_tgt(t) in _norm_tgt(expected_target) for t in all_targets if t)
        if not target_match:
            failures.append(f"Target missing/mismatch: expected '{expected_target}', got '{actual_target}'")

    # 4. Applicant categories check
    expected_cats = expected.get("categories", [])
    actual_cats = plan_data.get("applicant_categories", [])
    for exp_c in expected_cats:
        if not any(exp_c.upper() == c.upper() for c in actual_cats):
            failures.append(f"Category missing: expected '{exp_c}', actual={actual_cats}")

    # 5. Numerical constraints check
    expected_nums = expected.get("numerics", [])
    all_num_text = " ".join(str(n.get("value", "")) for n in plan_data.get("numeric_constraints", [])) + " " + " ".join(plan_data.get("preserved_terms", [])) + " " + " ".join(str(it.get("constraints", [])) for it in plan_data.get("intents", []))
    for num in expected_nums:
        if num not in all_num_text and f"{num}.0" not in all_num_text:
            failures.append(f"Numerical constraint missing: '{num}' not found in {all_num_text}")

    # 6. Preserved key terms
    expected_terms = expected.get("preserved_terms", [])
    all_preserved_text = " ".join(plan_data.get("preserved_terms", [])).lower() + " " + plan_data.get("resolved_query", "").lower()
    for term in expected_terms:
        if term.lower() not in all_preserved_text:
            failures.append(f"Preserved term missing: '{term}'")

    # 7. Forbidden retrieval queries (e.g. isolated 'without', 'documents')
    forbidden = expected.get("forbidden_queries", [])
    actual_retrieval_queries = plan_data.get("retrieval_queries", [])
    for fb in forbidden:
        for rq in actual_retrieval_queries:
            if rq.strip().lower() == fb.strip().lower():
                failures.append(f"Forbidden retrieval query generated: '{rq}'")

    passed = len(failures) == 0
    return {
        "id": item["id"],
        "class": item["class"],
        "question": item["question"],
        "passed": passed,
        "failures": failures,
        "actual": plan_data,
        "expected": expected,
    }


def run_one_scenario(item: dict[str, Any]) -> dict[str, Any]:
    question = item["question"]
    chat_history = item.get("chat_history", [])

    # Step 1: Conversation resolution
    if chat_history:
        res_info = resolve_conversation(question=question, chat_history=chat_history)
        resolved_query = res_info.get("resolved_question") or question
        conv_mode = res_info.get("mode", "follow_up")
    else:
        resolved_query = question
        conv_mode = "standalone"

    # Step 2: Build query plan
    plan = build_query_plan(question=resolved_query)
    frame = plan_to_frame(plan)
    retrieval_queries = retrieval_queries_from_plan(plan, limit=4)
    pipeline_stop = should_stop_pipeline(plan)

    # Step 3: Build lossless query
    try:
        lossless = build_lossless_query(plan, question=resolved_query)
    except Exception:
        lossless = None

    intents_summary = []
    for idx, it in enumerate(plan.intents):
        intents_summary.append({
            "intent_index": idx,
            "question": it.question,
            "request_type": it.request_type,
            "target": it.target,
            "requested_attributes": it.requested_attributes,
            "constraints": it.constraints,
            "retrieval_query": it.retrieval_query,
            "negated": it.negated,
        })

    numeric_constraints = []
    if lossless and lossless.numeric_constraints:
        for nc in lossless.numeric_constraints:
            numeric_constraints.append({
                "name": nc.name,
                "value": nc.value,
                "operator": nc.operator,
                "unit": nc.unit,
            })

    applicant_categories = list(lossless.applicant_categories) if lossless else []

    target = plan.intents[0].target if plan.intents else None
    if not target and lossless:
        target = lossless.program_binding or lossless.degree_binding

    return {
        "original_query": question,
        "resolved_query": resolved_query,
        "conversation_mode": conv_mode,
        "domain_decision": plan.domain_decision,
        "domain_confidence": plan.domain_confidence,
        "domain_reason": plan.domain_reason,
        "pipeline_stop": pipeline_stop,
        "is_multi_intent": plan.is_multi_intent,
        "intent_count": len(plan.intents),
        "target": target,
        "intents": intents_summary,
        "retrieval_queries": list(retrieval_queries),
        "preserved_terms": list(plan.preserved_terms),
        "applicant_categories": applicant_categories,
        "numeric_constraints": numeric_constraints,
        "qualifications": list(lossless.qualifications) if lossless else [],
        "route": plan.route,
        "confidence": plan.confidence,
    }


def execute_suite(scenarios_file: Path, output_dir: Path, tag: str = "redesign") -> list[dict[str, Any]]:
    with open(scenarios_file, "r", encoding="utf-8") as f:
        scenarios: list[dict[str, Any]] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now()
    start_time_perf = time.perf_counter()

    print(f"\n=======================================================")
    print(f"RUNNING 50-SCENARIO QUERY UNDERSTANDING SUITE [{tag.upper()}]")
    print(f"Scenarios: {len(scenarios)} | Output: {output_dir}")
    print(f"Execution Mode: {'OFFLINE (Deterministic)' if os.environ.get('PLANNER_OFFLINE_MODE') == '1' else 'DEFAULT (Local Runtime)'}")
    print(f"=======================================================\n")

    evaluations: list[dict[str, Any]] = []

    for idx, sc in enumerate(scenarios, 1):
        sid = sc["id"]
        print(f"[{idx:02d}/50] {sid:<48} ... ", end="", flush=True)

        t0 = time.perf_counter()
        plan_data = run_one_scenario(sc)
        lat = time.perf_counter() - t0

        res = evaluate_plan_record(sc, plan_data)
        res["latency_sec"] = round(lat, 4)

        status_str = "PASS" if res["passed"] else "FAIL"
        print(f"{status_str} ({res['latency_sec']}s)")
        if not res["passed"]:
            for f in res["failures"]:
                print(f"       -> {f}")

        evaluations.append(res)

    total_duration = time.perf_counter() - start_time_perf
    passed_count = sum(1 for e in evaluations if e["passed"])
    pass_rate = (passed_count / len(evaluations)) * 100.0

    # 1. Write planner_50q_results.json
    results_path = output_dir / "planner_50q_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)

    # 2. Write planner_50q_failures.txt
    failures_path = output_dir / "planner_50q_failures.txt"
    failures_list = [e for e in evaluations if not e["passed"]]
    with open(failures_path, "w", encoding="utf-8") as f:
        f.write(f"50-Scenario Planner Evaluation Failure Audit\n")
        f.write(f"Timestamp: {t_start.isoformat()}\n")
        f.write(f"Total Failures: {len(failures_list)} / {len(evaluations)}\n\n")
        if not failures_list:
            f.write("CONGRATULATIONS: ZERO FAILURES DETECTED! All 50 scenarios passed semantic validation.\n")
        else:
            for idx, item in enumerate(failures_list, 1):
                f.write(f"Failure #{idx}: {item['id']} ({item['class']})\n")
                f.write(f"  Question: {item['question']}\n")
                for err in item["failures"]:
                    f.write(f"  Reason: {err}\n")
                f.write("\n")

    # 3. Class-level metrics
    classes: dict[str, list[dict[str, Any]]] = {}
    for e in evaluations:
        classes.setdefault(e["class"], []).append(e)

    # 4. Write planner_50q_report.txt
    report_path = output_dir / "planner_50q_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("50-SCENARIO INSTITUTIONAL QUERY PLANNER EVALUATION REPORT\n")
        f.write("================================================================================\n")
        f.write(f"Timestamp:        {t_start.strftime('%Y-%m-%d %H:%M:%S IST')}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Execution Mode:   {'Deterministic Offline Mode' if os.environ.get('PLANNER_OFFLINE_MODE') == '1' else 'Local Runtime'}\n")
        f.write(f"Total Scenarios:  {len(evaluations)}\n")
        f.write(f"Passed:           {passed_count} / {len(evaluations)} ({pass_rate:.1f}%)\n")
        f.write(f"Failed:           {len(failures_list)} / {len(evaluations)}\n")
        f.write(f"Total Latency:    {total_duration:.2f}s (Avg: {total_duration/len(evaluations):.3f}s/q)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("PERFORMANCE BY SCENARIO CLASS\n")
        f.write("--------------------------------------------------------------------------------\n")
        for cname, citems in sorted(classes.items()):
            cpass = sum(1 for x in citems if x["passed"])
            crate = (cpass / len(citems)) * 100.0
            f.write(f"- {cname:<52}: {cpass:>2}/{len(citems):<2} ({crate:>5.1f}%)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("CRITICAL RESIDUAL FAILURE CHECKS (7 HISTORICAL RESIDUAL CASES)\n")
        f.write("--------------------------------------------------------------------------------\n")
        orig_7_ids = [
            "scen_04_story_exec_mtech_working_engineer",
            "scen_05_story_provisional_admission_delay",
            "scen_06_story_phd_direct_from_btech_sc",
            "scen_10_multi_msc_jam_and_phd_duration",
            "scen_11_multi_hostel_charges_and_anti_ragging",
            "scen_12_multi_part_time_mtech_exp_and_phd_assistantship",
            "scen_15_probation_max_credits",
        ]
        all_7_passed = True
        for o_id in orig_7_ids:
            rec = next((x for x in evaluations if x["id"] == o_id), None)
            if rec:
                status = "PASS" if rec["passed"] else "FAIL"
                if not rec["passed"]:
                    all_7_passed = False
                f.write(f"- {o_id:<55}: {status}\n")
        f.write(f"\nAll 7 Original Residual Failures Resolved: {'YES' if all_7_passed else 'NO'}\n")
        f.write(f"Acceptance Gate Passed (>= 48/50 and all 7 passed): {'YES' if (pass_rate >= 96.0 and all_7_passed) else 'NO'}\n")
        f.write("================================================================================\n")

    # 5. Write manifest.json
    git_rev = "unknown"
    git_branch = "ai-platform"
    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass

    manifest = {
        "dataset_version": "50q_development_regression_v1",
        "dataset_provenance": "30 reviewed historical scenarios + 20 realistic college RAG scenarios",
        "timestamp": t_start.isoformat(),
        "total_scenarios": len(evaluations),
        "passed_scenarios": passed_count,
        "pass_rate_percent": round(pass_rate, 2),
        "execution_mode": "offline_deterministic" if os.environ.get("PLANNER_OFFLINE_MODE") == "1" else "local_model_runtime",
        "scoring_rules": {
            "domain_decision_match": True,
            "pipeline_stop_match": True,
            "multi_intent_classification_match": True,
            "target_entity_exact_or_substring": True,
            "applicant_categories_coverage": True,
            "numerical_constraints_coverage": True,
            "preserved_terms_coverage": True,
            "forbidden_queries_zero_tolerance": True,
        },
        "all_7_historical_failures_passed": all_7_passed,
        "acceptance_gate_passed": pass_rate >= 96.0 and all_7_passed,
        "git_commit": git_rev,
        "git_branch": git_branch,
        "duration_seconds": round(total_duration, 3),
        "artifacts": {
            "results_json": str(results_path),
            "report_txt": str(report_path),
            "failures_txt": str(failures_path),
        }
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=======================================================")
    print(f"50Q BENCHMARK COMPLETED: {passed_count}/{len(evaluations)} PASSED ({pass_rate:.1f}%)")
    print(f"All 7 Original Failures Resolved: {'YES' if all_7_passed else 'NO'}")
    print(f"Acceptance Gate Status: {'PASSED' if (pass_rate >= 96.0 and all_7_passed) else 'FAILED'}")
    print(f"Output saved to: {output_dir}")
    print(f"=======================================================\n")

    return evaluations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50-scenario query planning harness.")
    parser.add_argument("--scenarios", type=str, default=str(DEFAULT_SCENARIOS_FILE))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="redesign")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_BASE / f"planner_redesign_{ts}"

    execute_suite(Path(args.scenarios), out_dir, tag=args.tag)
