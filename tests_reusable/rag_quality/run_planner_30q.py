"""30-Scenario Query Understanding and Planning Test Harness and Runner.

Runs the 30 realistic student-question scenarios through the canonical
query understanding path:
1. resolve_conversation (for conversational history)
2. build_query_plan (core planning)
3. plan_to_frame (semantic query frame construction)
4. build_lossless_query (LSQC lossless extraction)
5. retrieval_queries_from_plan (retrieval query generation)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
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

DEFAULT_SCENARIOS_FILE = ROOT / "tests_reusable" / "rag_quality" / "planner_30q_scenarios.json"
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


def execute_suite(scenarios_file: Path, output_dir: Path, tag: str = "run") -> list[dict[str, Any]]:
    with open(scenarios_file, "r", encoding="utf-8") as f:
        scenarios: list[dict[str, Any]] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f"RUNNING 30-SCENARIO QUERY UNDERSTANDING SUITE [{tag.upper()}]")
    print(f"Scenarios: {len(scenarios)} | Output: {output_dir}")
    print(f"=======================================================\n")

    evaluations: list[dict[str, Any]] = []

    for idx, sc in enumerate(scenarios, 1):
        sid = sc["id"]
        sclass = sc["class"]
        print(f"[{idx:02d}/30] {sid:<45} ... ", end="", flush=True)

        t0 = time.perf_counter()
        plan_data = run_one_scenario(sc)
        lat = time.perf_counter() - t0

        res = evaluate_plan_record(sc, plan_data)
        res["latency_sec"] = round(lat, 3)

        status_str = "PASS" if res["passed"] else "FAIL"
        print(f"{status_str} ({res['latency_sec']}s)")
        if not res["passed"]:
            for f in res["failures"]:
                print(f"       -> {f}")

        evaluations.append(res)

    out_json = output_dir / f"evaluations_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)

    passed_count = sum(1 for e in evaluations if e["passed"])
    print(f"\n{tag.upper()} RESULT: {passed_count}/{len(evaluations)} PASSED ({passed_count/len(evaluations)*100:.1f}%)")
    return evaluations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 30-scenario query planning harness.")
    parser.add_argument("--scenarios", type=str, default=str(DEFAULT_SCENARIOS_FILE))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_BASE / f"planner_30q_{ts}"

    execute_suite(Path(args.scenarios), out_dir, tag=args.tag)
