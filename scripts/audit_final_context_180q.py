#!/usr/bin/env python3
"""
Zero-Inference Final-Context Audit Tool for the 180-Question Frozen Benchmark.

Audits and records the exact pre-generation evidence context sent to the answer
LLM for all 180 benchmark scenarios in:
    tests_reusable/rag_quality/generalization_180q_scenarios.json

STRICT SAFETY GUARANTEES:
- Zero LLM answer generation calls (no Ollama, no API inference).
- Frozen benchmark scenarios JSON is loaded read-only with SHA-256 validation.
- Per-question incremental checkpointing (resumable without duplicate runs).
- Generates human-readable TXT records across 5 batches and combined TXT.
- Generates machine-readable progress.json, results.jsonl, and audit_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure offline deterministic mode for query planner
os.environ["PLANNER_OFFLINE_MODE"] = "1"

from ai_platform.runtime.engine import inspect_context, get_inspect_nodes

# Frozen benchmark dataset path and authoritative SHA-256 hash
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "tests_reusable/rag_quality/generalization_180q_scenarios.json"
EXPECTED_DATASET_SHA256 = "90c8b46a4310a4bcc8d25bb7f256443d3212efc6184e53a1b17e49f49290e47a"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "test_outputs/rag_quality/final_context_audit_180q"

BATCH_DEFINITIONS = [
    {"batch": 1, "start": 0, "end": 40, "filename": "batch_01_questions_001_040.txt", "label": "Questions 001–040"},
    {"batch": 2, "start": 40, "end": 80, "filename": "batch_02_questions_041_080.txt", "label": "Questions 041–080"},
    {"batch": 3, "start": 80, "end": 120, "filename": "batch_03_questions_081_120.txt", "label": "Questions 081–120"},
    {"batch": 4, "start": 120, "end": 160, "filename": "batch_04_questions_121_160.txt", "label": "Questions 121–160"},
    {"batch": 5, "start": 160, "end": 180, "filename": "batch_05_questions_161_180.txt", "label": "Questions 161–180"},
]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset_hash(file_path: Path, expected_hash: str = EXPECTED_DATASET_SHA256) -> str:
    """Verify that dataset hash matches the expected frozen benchmark hash."""
    actual_hash = compute_sha256(file_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Dataset hash mismatch for {file_path}!\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            "Stopping immediately to protect benchmark integrity."
        )
    return actual_hash


def _matches_gold_source_or_facet(
    text: str,
    source: str,
    gold_sources: Sequence[str],
    gold_facets: Sequence[str],
    require_facets: bool = True,
) -> bool:
    """Check whether a text snippet and source align with gold scenario annotations."""
    src_clean = str(source or "").strip()
    src_matched = False
    for g_src in gold_sources:
        g_name = Path(g_src).name
        if g_name and (g_name in src_clean or Path(src_clean).name == g_name):
            src_matched = True
            break

    if not src_matched:
        return False

    if not require_facets or not gold_facets:
        return True

    text_low = text.lower()
    for facet in gold_facets:
        f_low = str(facet).lower()
        if f_low in text_low:
            return True
        tokens = [
            w for w in re.findall(r"[a-zA-Z0-9%₹]+", f_low)
            if w not in ("for", "of", "on", "the", "a", "an", "in", "to", "with", "and", "is", "at")
        ]
        if len(tokens) >= 2 and all(t in text_low for t in tokens):
            return True
    return False


def _analyze_evidence_loss(
    scenario: dict[str, Any],
    diag: dict[str, Any],
) -> tuple[str, str]:
    """Determine whether gold evidence survived into final context or where it was lost."""
    if bool(scenario.get("is_control")):
        if diag.get("will_abstain") or diag.get("pipeline_stop"):
            return "CONTROL_SAFE_ABSTENTION", "Control question successfully abstained / rejected."
        return "CONTROL_FALSE_ACCEPTANCE", "Control query was not rejected by guard or insufficient evidence gate."

    intents = scenario.get("intents", [])
    if not intents:
        return "NO_GOLD_INTENTS", "Scenario has no gold intent annotations."

    pkg_context = str(diag.get("final_evidence_context", "") or "")
    if not pkg_context and diag.get("will_abstain"):
        abstention_reason = diag.get("abstention_reason") or "insufficient_evidence"
        return "PIPELINE_ABSTAINED", f"Pipeline abstained before generation ({abstention_reason})."

    all_intents_survived = True
    loss_details: list[str] = []

    for intent in intents:
        gold_sources = intent.get("gold_sources", [])
        gold_facets = intent.get("facets", [])

        # Check final context inclusion
        pkg_survived = False
        for u in diag.get("evidence_units", []):
            if _matches_gold_source_or_facet(u.get("text", ""), u.get("source", ""), gold_sources, gold_facets, require_facets=True):
                pkg_survived = True
                break

        # Check prompt text inclusion if not found in individual units
        if not pkg_survived and gold_facets:
            ctx_low = pkg_context.lower()
            pkg_survived = any(
                f.lower() in ctx_low or (
                    len([w for w in re.findall(r"[a-zA-Z0-9%₹]+", f.lower()) if w not in ("for", "of", "on", "the", "a", "an", "in", "to", "with", "and", "is", "at")]) >= 2
                    and all(w in ctx_low for w in re.findall(r"[a-zA-Z0-9%₹]+", f.lower()) if w not in ("for", "of", "on", "the", "a", "an", "in", "to", "with", "and", "is", "at"))
                )
                for f in gold_facets
            )

        if not pkg_survived:
            all_intents_survived = False
            # Check verification trace
            ver_survived = any(
                _matches_gold_source_or_facet(
                    str(v.get("text") or v.get("snippet") or ""),
                    str(v.get("source") or v.get("document_id") or ""),
                    gold_sources,
                    gold_facets,
                    require_facets=False,
                )
                for v in diag.get("verification_trace", ())
                if v.get("decision") in {"accept", "verified"}
            )
            raw_survived = (diag.get("raw_retrieval_count", 0) > 0)

            if not raw_survived:
                loss_details.append(f"Intent {intent.get('intent_index', 0)}: RAW_RETRIEVAL_MISS")
            elif not ver_survived:
                loss_details.append(f"Intent {intent.get('intent_index', 0)}: VERIFICATION_FALSE_REJECTION")
            else:
                loss_details.append(f"Intent {intent.get('intent_index', 0)}: PACKAGING_BUDGET_DROP")

    if all_intents_survived:
        return "FULLY_PRESERVED", "All required gold evidence facets successfully survived into final context."
    return "EVIDENCE_LOST", "; ".join(loss_details) if loss_details else "Required evidence was lost before final packaging."


def format_txt_record(
    scenario: dict[str, Any],
    batch_num: int,
    diag: dict[str, Any],
    error_msg: str | None = None,
) -> str:
    """Format one scenario record strictly conforming to Step 4 specification."""
    s_id = scenario.get("id", "UNKNOWN_ID")
    group = scenario.get("group", "UNSPECIFIED")
    question = scenario.get("question", "")
    resolved_q = diag.get("resolved_question") or "NOT AVAILABLE"
    is_multi = diag.get("is_multi_intent", False)
    intent_cls = "Multi-Intent" if is_multi else "Single-Intent"
    if is_multi and diag.get("intent_questions"):
        intent_cls += f" ({len(diag['intent_questions'])} sub-intents: {list(diag['intent_questions'])})"

    ready_for_gen = bool(diag.get("ready_for_generation", False))
    will_abstain = bool(diag.get("will_abstain", False))
    pipeline_stop = bool(diag.get("pipeline_stop", False))
    abstention_reason = diag.get("abstention_reason") or "NONE"
    stop_reason = diag.get("stop_reason") or "NONE"
    evidence_status = diag.get("evidence_status", "unknown")
    coverage_status = diag.get("evidence_coverage_status", "unknown")

    if error_msg:
        status_str = "ERROR"
    elif will_abstain or pipeline_stop:
        status_str = "ABSTAIN"
    else:
        status_str = "SUCCESS"

    lines: list[str] = [
        "=" * 60,
        f"QUESTION ID: {s_id}",
        f"BATCH: {batch_num}",
        f"BENCHMARK GROUP: {group}",
        f"STATUS: {status_str}",
        "ANSWER: NOT GENERATED — ZERO-INFERENCE MODE",
        "=" * 60,
        "",
        "ORIGINAL QUESTION:",
        question,
        "",
        "RESOLVED QUESTION:",
        resolved_q,
        "",
        "INTENT CLASSIFICATION:",
        intent_cls,
        "",
        "PIPELINE / EVIDENCE STATUS:",
        f"- Pipeline stopped early: {pipeline_stop} (Reason: {stop_reason})",
        f"- Evidence status: {evidence_status}",
        f"- Coverage status: {coverage_status}",
        f"- Ready for generation: {ready_for_gen}",
        f"- Will abstain: {will_abstain} (Reason: {abstention_reason})",
        f"- Fallback response: {diag.get('fallback_response') or 'NONE'}",
        "",
        "FINAL CONTEXT THAT WOULD BE SENT TO THE LLM:",
        "-" * 60,
    ]

    final_ctx = diag.get("final_evidence_context", "").strip()
    if not final_ctx or will_abstain or pipeline_stop:
        lines.append("FINAL CONTEXT: EMPTY — PIPELINE ABSTAINED OR FAILED CLOSED")
    else:
        lines.append(final_ctx)
    lines.append("-" * 60)
    lines.append("")

    # Evidence included
    lines.append("EVIDENCE INCLUDED:")
    units = diag.get("evidence_units", [])
    if not units or will_abstain:
        lines.append("(None — context is empty or pipeline abstained)")
    else:
        for idx, u in enumerate(units, start=1):
            eid = u.get("evidence_id", f"unit_{idx}")
            src = u.get("source", "unknown")
            hdg = u.get("heading") or "No Heading"
            txt_snippet = " ".join(u.get("text", "").split())
            if len(txt_snippet) > 200:
                txt_snippet = txt_snippet[:200] + "..."
            lines.append(f"{idx}. [{eid}] [{src}] [{hdg}]")
            lines.append(f"   Snippet: {txt_snippet}")
    lines.append("")

    # Evidence excluded or lost
    loss_stage, loss_explanation = _analyze_evidence_loss(scenario, diag)
    lines.append("EVIDENCE EXCLUDED OR LOST:")
    lines.append(f"- Loss / Preservation Stage: {loss_stage}")
    lines.append(f"- Finding: {loss_explanation}")
    reasons = diag.get("package_reasons", ())
    if reasons:
        lines.append(f"- Packaging exclusion reasons: {', '.join(reasons)}")
    omitted_count = diag.get("omitted_units_count", 0)
    if omitted_count > 0:
        lines.append(f"- Units omitted during packaging: {omitted_count}")
    omitted_claims = diag.get("omitted_claim_ids", ())
    if omitted_claims:
        lines.append(f"- Omitted claim IDs: {list(omitted_claims)}")
    rejected_count = diag.get("rejected_candidate_count", 0)
    if rejected_count > 0:
        lines.append(f"- Candidates rejected during verification: {rejected_count}")
    unsupported_years = diag.get("unsupported_years", ())
    if unsupported_years:
        lines.append(f"- Unsupported temporal years in query: {list(unsupported_years)}")
    lines.append("")

    # Context Statistics
    char_count = diag.get("context_character_count", 0)
    token_est = diag.get("estimated_token_count", 0)
    unit_count = diag.get("evidence_unit_count", 0)
    lines.extend([
        "CONTEXT STATISTICS:",
        f"- Character count: {char_count}",
        f"- Token estimate: {token_est}",
        f"- Evidence-unit count: {unit_count}",
        f"- Ready for generation: {ready_for_gen}",
        f"- Will abstain: {will_abstain}",
        f"- Abstention reason: {abstention_reason}",
        "",
        "DIAGNOSTIC ERRORS / WARNINGS:",
        error_msg or ("NONE" if not diag.get("unsupported_years") else f"Temporal year mismatch: {list(diag['unsupported_years'])}"),
        "",
    ])

    return "\n".join(lines)


def run_audit(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_selector: int | str = "all",
    resume: bool = True,
    max_questions: int | None = None,
    inspect_fn: Callable[..., dict[str, Any]] = inspect_context,
    nodes: Any = None,
    expected_hash: str | None = EXPECTED_DATASET_SHA256,
) -> dict[str, Any]:
    """Execute the full 180-question zero-inference context audit with incremental checkpoints."""
    start_time = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Zero-Inference Final-Context Audit Runner ===")
    print(f"Benchmark File: {benchmark_path}")
    print(f"Output Directory: {out_dir}")

    # Step 1: Validate hash
    if expected_hash is not None:
        dataset_hash = validate_dataset_hash(benchmark_path, expected_hash=expected_hash)
    else:
        dataset_hash = compute_sha256(benchmark_path)
    print(f"Verified Dataset SHA-256: {dataset_hash}")

    with open(benchmark_path, encoding="utf-8") as f:
        all_scenarios: list[dict[str, Any]] = json.load(f)

    total_scenarios = len(all_scenarios)
    print(f"Loaded {total_scenarios} scenarios from benchmark.")

    # Initialize shared inspection nodes once to reuse cache
    if nodes is None and inspect_fn is inspect_context:
        print("Initializing cached inspection nodes (warming embedding models)...")
        nodes = get_inspect_nodes()

    # Determine batch filter
    target_batches: list[dict[str, Any]] = []
    if str(batch_selector).lower() == "all":
        target_batches = list(BATCH_DEFINITIONS)
    else:
        try:
            b_num = int(batch_selector)
            matched = [b for b in BATCH_DEFINITIONS if b["batch"] == b_num]
            if not matched:
                raise ValueError(f"Invalid batch number {b_num}. Must be 1 to 5.")
            target_batches = matched
        except ValueError as e:
            raise ValueError(f"Invalid batch selector '{batch_selector}'. Choose 1, 2, 3, 4, 5, or 'all'.") from e

    # Checkpoint paths
    results_jsonl_path = out_dir / "results.jsonl"
    progress_path = out_dir / "progress.json"
    failures_txt_path = out_dir / "failures_and_warnings.txt"
    all_txt_path = out_dir / "all_180_questions_final_context.txt"

    completed_ids: set[str] = set()
    existing_results: dict[str, dict[str, Any]] = {}

    if resume and results_jsonl_path.exists():
        print(f"Resuming from existing checkpoint: {results_jsonl_path}")
        with open(results_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                    s_id = rec.get("id")
                    if s_id:
                        completed_ids.add(s_id)
                        existing_results[s_id] = rec
                except Exception:
                    pass
        print(f"Loaded {len(completed_ids)} previously completed scenarios.")

    # Track audit statistics
    stats = {
        "total_benchmark_questions": total_scenarios,
        "scenarios_inspected": len(completed_ids),
        "success_count": 0,
        "abstain_count": 0,
        "error_count": 0,
        "answerable_with_context": 0,
        "answerable_empty_context": 0,
        "controls_safe_abstain": 0,
        "controls_false_accept": 0,
        "multi_intent_total": 0,
        "multi_intent_supported": 0,
        "evidence_lost_cases": 0,
        "char_counts": [],
        "token_estimates": [],
        "unit_counts": [],
        "batches_status": {},
    }

    questions_processed_this_run = 0

    # Process each target batch sequentially
    for b_def in target_batches:
        b_idx = b_def["batch"]
        b_start = b_def["start"]
        b_end = b_def["end"]
        b_file = out_dir / b_def["filename"]
        b_scenarios = all_scenarios[b_start:b_end]

        print(f"\n--- Processing Batch {b_idx}: {b_def['label']} ({len(b_scenarios)} questions) ---")

        b_txt_records: list[str] = []

        # If batch TXT already exists and we're resuming, we'll re-read or regenerate cleanly
        for s in b_scenarios:
            if max_questions is not None and questions_processed_this_run >= max_questions:
                print(f"[LIMIT_REACHED] Reached maximum requested questions ({max_questions}).")
                break

            s_id = s["id"]
            question = s["question"]

            if s_id in existing_results:
                diag = existing_results[s_id].get("diagnostic", {})
                error_msg = existing_results[s_id].get("error")
                rec_txt = format_txt_record(s, b_idx, diag, error_msg)
                b_txt_records.append(rec_txt)
                continue

            # Run zero-inference diagnostic
            error_msg: str | None = None
            diag: dict[str, Any] = {}
            try:
                diag = inspect_fn(
                    question,
                    chat_history=tuple(s.get("chat_history", ())),
                    nodes=nodes,
                )
            except Exception as exc:
                error_msg = f"Diagnostic Exception: {type(exc).__name__}: {str(exc)}"
                diag = {
                    "question": question,
                    "resolved_question": question,
                    "pipeline_stop": False,
                    "stop_reason": "DIAGNOSTIC_EXCEPTION",
                    "is_multi_intent": bool(s.get("is_multi_intent", False)),
                    "intent_questions": (),
                    "evidence_status": "error",
                    "evidence_coverage_status": "error",
                    "ready_for_generation": False,
                    "will_abstain": True,
                    "abstention_reason": "diagnostic_exception",
                    "fallback_response": "Error during diagnostic execution",
                    "final_evidence_context": "",
                    "context_character_count": 0,
                    "estimated_token_count": 0,
                    "evidence_units": [],
                    "evidence_unit_count": 0,
                    "prompt_messages": [],
                    "candidate_count": 0,
                    "verified_candidate_count": 0,
                    "verification_trace": (),
                    "unsupported_years": (),
                    "rejected_candidate_count": 0,
                    "rejected_candidates": (),
                    "package_reasons": (),
                    "omitted_units_count": 0,
                    "omitted_claim_ids": (),
                    "supported_claim_ids": (),
                    "partial_claim_ids": (),
                    "claim_audit_status": "error",
                    "raw_retrieval_count": 0,
                }

            # Create jsonl record
            loss_stage, loss_explanation = _analyze_evidence_loss(s, diag)
            rec_json = {
                "id": s_id,
                "batch": b_idx,
                "group": s.get("group"),
                "question": question,
                "is_control": bool(s.get("is_control", False)),
                "is_multi_intent": bool(diag.get("is_multi_intent", False)),
                "status": "ERROR" if error_msg else ("ABSTAIN" if diag.get("will_abstain") else "SUCCESS"),
                "ready_for_generation": bool(diag.get("ready_for_generation", False)),
                "will_abstain": bool(diag.get("will_abstain", False)),
                "abstention_reason": diag.get("abstention_reason"),
                "context_character_count": diag.get("context_character_count", 0),
                "estimated_token_count": diag.get("estimated_token_count", 0),
                "evidence_unit_count": diag.get("evidence_unit_count", 0),
                "loss_stage": loss_stage,
                "loss_explanation": loss_explanation,
                "error": error_msg,
                "diagnostic": diag,
            }

            # Incremental checkpoint to results.jsonl
            with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(rec_json, ensure_ascii=False) + "\n")

            existing_results[s_id] = rec_json
            completed_ids.add(s_id)
            questions_processed_this_run += 1

            # Format TXT record
            rec_txt = format_txt_record(s, b_idx, diag, error_msg)
            b_txt_records.append(rec_txt)

            # Record failure / warning if applicable
            if error_msg or (s.get("is_control") and not diag.get("will_abstain")) or (not s.get("is_control") and loss_stage != "FULLY_PRESERVED"):
                with open(failures_txt_path, "a", encoding="utf-8") as f_fail:
                    f_fail.write(f"[{s_id}] Batch {b_idx} | Group: {s.get('group')} | Control: {s.get('is_control')}\n")
                    f_fail.write(f"Question: {question}\n")
                    f_fail.write(f"Loss Stage: {loss_stage} | Finding: {loss_explanation}\n")
                    if error_msg:
                        f_fail.write(f"Error: {error_msg}\n")
                    f_fail.write("-" * 50 + "\n")

            # Update progress.json after every single question
            progress_data = {
                "benchmark_hash": dataset_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_scenarios": total_scenarios,
                "completed_count": len(completed_ids),
                "remaining_count": total_scenarios - len(completed_ids),
                "completed_ids": sorted(completed_ids),
                "current_batch": b_idx,
            }
            with open(progress_path, "w", encoding="utf-8") as f_prog:
                json.dump(progress_data, f_prog, indent=2)

        # Write out complete batch TXT file
        with open(b_file, "w", encoding="utf-8") as f_batch:
            f_batch.write(f"============================================================\n")
            f_batch.write(f"ZERO-INFERENCE FINAL-CONTEXT AUDIT — BATCH {b_idx:02d}\n")
            f_batch.write(f"SCOPE: {b_def['label']} | TOTAL QUESTIONS: {len(b_scenarios)}\n")
            f_batch.write(f"DATASET SHA-256: {dataset_hash}\n")
            f_batch.write(f"============================================================\n\n")
            for r_txt in b_txt_records:
                f_batch.write(r_txt + "\n\n")

        print(f"Saved Batch {b_idx} TXT file: {b_file}")

    # Build combined all_180_questions_final_context.txt in original benchmark order
    print("\nAssembling complete 180-question combined TXT file...")
    with open(all_txt_path, "w", encoding="utf-8") as f_all:
        f_all.write(f"============================================================\n")
        f_all.write(f"ZERO-INFERENCE FINAL-CONTEXT AUDIT — ALL 180 QUESTIONS\n")
        f_all.write(f"BENCHMARK: {benchmark_path.name}\n")
        f_all.write(f"DATASET SHA-256: {dataset_hash}\n")
        f_all.write(f"EXECUTION MODE: ZERO-INFERENCE (NO LLM GENERATION CALLS)\n")
        f_all.write(f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_all.write(f"============================================================\n\n")

        for idx, s in enumerate(all_scenarios):
            s_id = s["id"]
            rec = existing_results.get(s_id)
            b_num = next((b["batch"] for b in BATCH_DEFINITIONS if b["start"] <= idx < b["end"]), 0)
            if rec:
                diag = rec.get("diagnostic", {})
                err = rec.get("error")
                txt_entry = format_txt_record(s, b_num, diag, err)
            else:
                txt_entry = (
                    f"============================================================\n"
                    f"QUESTION ID: {s_id}\n"
                    f"BATCH: {b_num}\n"
                    f"STATUS: PENDING_RESUMPTION\n"
                    f"============================================================\n"
                    f"ORIGINAL QUESTION: {s['question']}\n"
                    f"FINAL CONTEXT: NOT RUN YET\n"
                )
            f_all.write(txt_entry + "\n\n")

    print(f"Saved combined 180-question TXT file: {all_txt_path}")

    # Calculate final aggregate statistics across all completed scenarios
    for idx, s in enumerate(all_scenarios):
        s_id = s["id"]
        rec = existing_results.get(s_id)
        if not rec:
            continue
        diag = rec.get("diagnostic", {})
        err = rec.get("error")
        is_ctrl = bool(s.get("is_control", False))
        is_multi = bool(diag.get("is_multi_intent", False))
        will_abstain = bool(diag.get("will_abstain", False))
        ready_for_gen = bool(diag.get("ready_for_generation", False))
        char_c = diag.get("context_character_count", 0)
        tok_e = diag.get("estimated_token_count", 0)
        units_c = diag.get("evidence_unit_count", 0)

        if err:
            stats["error_count"] += 1
        elif will_abstain:
            stats["abstain_count"] += 1
        else:
            stats["success_count"] += 1

        if is_ctrl:
            if will_abstain or diag.get("pipeline_stop"):
                stats["controls_safe_abstain"] += 1
            else:
                stats["controls_false_accept"] += 1
        else:
            if ready_for_gen and char_c > 0:
                stats["answerable_with_context"] += 1
            else:
                stats["answerable_empty_context"] += 1

        if is_multi:
            stats["multi_intent_total"] += 1
            if ready_for_gen and char_c > 0:
                stats["multi_intent_supported"] += 1

        if rec.get("loss_stage") not in {"FULLY_PRESERVED", "CONTROL_SAFE_ABSTENTION"}:
            stats["evidence_lost_cases"] += 1

        if char_c > 0:
            stats["char_counts"].append(char_c)
            stats["token_estimates"].append(tok_e)
            stats["unit_counts"].append(units_c)

    # Compute summaries
    chars = stats["char_counts"]
    avg_chars = round(sum(chars) / len(chars), 1) if chars else 0
    max_chars = max(chars) if chars else 0
    min_chars = min(chars) if chars else 0
    avg_tokens = round(sum(stats["token_estimates"]) / len(stats["token_estimates"]), 1) if stats["token_estimates"] else 0
    avg_units = round(sum(stats["unit_counts"]) / len(stats["unit_counts"]), 1) if stats["unit_counts"] else 0

    elapsed_time = round(time.monotonic() - start_time, 2)

    summary_json = {
        "benchmark_file": str(benchmark_path),
        "benchmark_hash": dataset_hash,
        "total_benchmark_questions": total_scenarios,
        "total_completed_questions": len(completed_ids),
        "total_remaining_questions": total_scenarios - len(completed_ids),
        "completion_rate_pct": round(len(completed_ids) / total_scenarios * 100, 1),
        "execution_mode": "ZERO_INFERENCE_NO_LLM",
        "total_elapsed_seconds": elapsed_time,
        "counts": {
            "success_ready_for_generation": stats["success_count"],
            "abstain_or_failed_closed": stats["abstain_count"],
            "diagnostic_errors": stats["error_count"],
            "answerable_with_context": stats["answerable_with_context"],
            "answerable_empty_context": stats["answerable_empty_context"],
            "controls_safe_abstain": stats["controls_safe_abstain"],
            "controls_false_accept": stats["controls_false_accept"],
            "multi_intent_total": stats["multi_intent_total"],
            "multi_intent_supported": stats["multi_intent_supported"],
            "evidence_lost_cases": stats["evidence_lost_cases"],
        },
        "context_statistics": {
            "average_characters": avg_chars,
            "min_characters": min_chars,
            "max_characters": max_chars,
            "average_token_estimate": avg_tokens,
            "average_evidence_units": avg_units,
        },
    }

    # Write audit_summary.json
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f_sum_j:
        json.dump(summary_json, f_sum_j, indent=2)

    # Write audit_summary.txt
    summary_txt_path = out_dir / "audit_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f_sum:
        f_sum.write("=" * 60 + "\n")
        f_sum.write("ZERO-INFERENCE FINAL-CONTEXT AUDIT SUMMARY (180 QUESTIONS)\n")
        f_sum.write("=" * 60 + "\n\n")
        f_sum.write(f"- Benchmark File: {benchmark_path.name}\n")
        f_sum.write(f"- Benchmark SHA-256: {dataset_hash}\n")
        f_sum.write(f"- Execution Mode: ZERO-INFERENCE (No LLM generation calls)\n")
        f_sum.write(f"- Total Questions in Dataset: {total_scenarios}\n")
        f_sum.write(f"- Completed Questions: {len(completed_ids)} / {total_scenarios} ({summary_json['completion_rate_pct']}%)\n")
        f_sum.write(f"- Total Execution Time: {elapsed_time}s\n\n")
        f_sum.write("OUTCOME BREAKDOWN:\n")
        f_sum.write(f"- Ready for Generation (Context Supplied): {stats['success_count']}\n")
        f_sum.write(f"- Abstained / Failed Closed (Safe Empty Context): {stats['abstain_count']}\n")
        f_sum.write(f"- Diagnostic Errors / Exceptions: {stats['error_count']}\n")
        f_sum.write(f"- Controls Safely Abstained / Refused: {stats['controls_safe_abstain']}\n")
        f_sum.write(f"- Controls Falsely Accepted: {stats['controls_false_accept']}\n")
        f_sum.write(f"- Multi-Intent Questions Evaluated: {stats['multi_intent_total']}\n")
        f_sum.write(f"- Multi-Intent with Verified Context: {stats['multi_intent_supported']}\n")
        f_sum.write(f"- Answerable Cases with Lost Gold Evidence: {stats['evidence_lost_cases']}\n\n")
        f_sum.write("CONTEXT SIZE DISTRIBUTION:\n")
        f_sum.write(f"- Average Context Characters: {avg_chars}\n")
        f_sum.write(f"- Min Context Characters: {min_chars}\n")
        f_sum.write(f"- Max Context Characters: {max_chars}\n")
        f_sum.write(f"- Average Token Estimate: {avg_tokens}\n")
        f_sum.write(f"- Average Evidence Units: {avg_units}\n\n")
        f_sum.write("ARTIFACT LOCATIONS:\n")
        f_sum.write(f"- Batch 01 (001–040): {out_dir / 'batch_01_questions_001_040.txt'}\n")
        f_sum.write(f"- Batch 02 (041–080): {out_dir / 'batch_02_questions_041_080.txt'}\n")
        f_sum.write(f"- Batch 03 (081–120): {out_dir / 'batch_03_questions_081_120.txt'}\n")
        f_sum.write(f"- Batch 04 (121–160): {out_dir / 'batch_04_questions_121_160.txt'}\n")
        f_sum.write(f"- Batch 05 (161–180): {out_dir / 'batch_05_questions_161_180.txt'}\n")
        f_sum.write(f"- Combined All 180:   {all_txt_path}\n")
        f_sum.write(f"- Failures & Lost:   {failures_txt_path}\n")
        f_sum.write(f"- Machine Summary:   {out_dir / 'audit_summary.json'}\n")
        f_sum.write(f"- Raw JSONL Results: {results_jsonl_path}\n")

    print(f"Saved Audit Summary TXT: {summary_txt_path}")
    print(f"Saved Audit Summary JSON: {out_dir / 'audit_summary.json'}")

    return summary_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-Inference Final-Context Audit for 180 Questions")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH, help="Path to scenarios JSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--batch", default="all", help="Batch to audit: 1, 2, 3, 4, 5, or 'all'")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoints")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Overwrite existing checkpoints")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional max questions to process")
    args = parser.parse_args()

    try:
        run_audit(
            benchmark_path=args.benchmark,
            out_dir=args.out_dir,
            batch_selector=args.batch,
            resume=args.resume,
            max_questions=args.max_questions,
        )
        return 0
    except Exception as exc:
        print(f"\n[FATAL_AUDIT_ERROR] {type(exc).__name__}: {str(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
