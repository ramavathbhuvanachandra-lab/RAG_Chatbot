#!/usr/bin/env python3
"""Stage-2 forensic gate for the V1 diagnosis report.

Diagnostic only. Reads diagnosis_v2.json and never imports backend or modifies
production code. The purpose is to decide whether verification is destroying
candidate identity before ranking, rather than relying on broad oracle labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def clean(value: Any) -> str:
    return str(value or "").strip()


def unique_keys(items: list[dict[str, Any]]) -> set[str]:
    return {
        clean(item.get("key") or item.get("candidate_id"))
        for item in items
        if clean(item.get("key") or item.get("candidate_id"))
    }


def candidate_status(decision: dict[str, Any]) -> str:
    return clean(decision.get("verification_status")) or "missing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        default="tests_reusable/test_diagnosis/reports/stage_diagnosis_v2/diagnosis_v2.json",
    )
    ap.add_argument(
        "--out",
        default="tests_reusable/test_diagnosis/reports/stage_diagnosis_v2/stage2_forensic_gate.md",
    )
    args = ap.parse_args()

    report_path = Path(args.report)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = list(data.get("results") or [])

    reason_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    destructive_cases: list[str] = []
    expected_cases_with_rejection: list[str] = []
    stage4_scope_wipe_cases: list[str] = []

    rows: list[dict[str, Any]] = []

    for result in results:
        case_id = clean(result.get("case_id"))
        stages = result.get("stages") or {}
        verification = (
            stages.get("verify_and_rank_node", {}).get("verification") or {}
        )
        decisions = list(verification.get("relevant_candidate_decisions") or [])

        rejected_trace = []
        for item in verification.get("verification_trace") or []:
            if not isinstance(item, dict):
                continue
            status = clean(item.get("status")).lower()
            if status in {"rejected", "uncertain"}:
                rejected_trace.append(item)
                for reason in item.get("reasons") or []:
                    reason_counts[clean(reason)] += 1

        ranked_items = (
            stages.get("verify_and_rank_node", {})
            .get("ranking", {})
            .get("items", [])
        )
        # The current runner exposes ranked candidates inside the verification
        # trace summary only indirectly, so use its diagnosis + decision records
        # where available.
        ranked_keys = {
            clean(d.get("key"))
            for d in decisions
            if clean(d.get("key")) and d.get("ranked_rank") is not None
        }

        decision_statuses = [candidate_status(d) for d in decisions]
        status_counts.update(decision_statuses)

        rejected_expected = [
            d for d in decisions if candidate_status(d) == "rejected"
        ]
        uncertain_expected = [
            d for d in decisions if candidate_status(d) == "uncertain"
        ]
        destructive = any(
            candidate_status(d) in {"rejected", "uncertain"}
            and d.get("ranked_rank") is None
            for d in decisions
        )

        if rejected_expected:
            expected_cases_with_rejection.append(case_id)
        if destructive:
            destructive_cases.append(case_id)

        internal = (
            stages.get("evidence_context_node", {})
            .get("internal_breakdown") or {}
        )
        grouped = internal.get("grouped_document_count")
        scope_filtered = internal.get("scope_filtered_document_count")
        if grouped is not None and grouped > 0 and scope_filtered == 0:
            stage4_scope_wipe_cases.append(case_id)

        rows.append(
            {
                "case": case_id,
                "expected_decisions": len(decisions),
                "expected_verified": sum(s == "verified" for s in decision_statuses),
                "expected_uncertain": len(uncertain_expected),
                "expected_rejected": len(rejected_expected),
                "destructive": destructive,
                "ranked_expected_same_key": len(ranked_keys),
                "grouped_docs": grouped,
                "scope_filtered_docs": scope_filtered,
            }
        )

    lines = [
        "# Stage 2 — Verification Forensic Gate",
        "",
        f"Cases analyzed: **{len(results)}**",
        "",
        "## Decision",
        "",
        "Stage 2 is treated as the active gate only if the same candidate key is tracked from RRF → verification → ranking.",
        "A broad benchmark label is not enough to call a verification failure, because another candidate may also match the benchmark oracle.",
        "",
        "## Per-case findings",
        "",
        "| Case | Expected candidates | Verified | Uncertain | Rejected | Same-key destructive? | Grouped docs | Scope-filtered docs |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['expected_decisions']} | {row['expected_verified']} | {row['expected_uncertain']} | {row['expected_rejected']} | {'YES' if row['destructive'] else 'no'} | {row['grouped_docs'] if row['grouped_docs'] is not None else '—'} | {row['scope_filtered_docs'] if row['scope_filtered_docs'] is not None else '—'} |"
        )

    lines += [
        "",
        "## Verification rejection reasons",
        "",
    ]
    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Gate metrics",
        "",
        f"- Expected-candidate statuses: {dict(status_counts)}",
        f"- Cases with an expected candidate rejected: **{len(expected_cases_with_rejection)}**",
        f"- Cases where an expected candidate was rejected/uncertain and did not survive as the same ranked key: **{len(destructive_cases)}**",
        f"- Cases where evidence grouping produced documents but scope filtering reduced the set to zero: **{len(stage4_scope_wipe_cases)}**",
        "",
        "## Stage-2 rebuild rule",
        "",
        "Do not loosen verification just to increase recall. Rebuild only after the destructive same-key failures are confirmed from candidate text + query frame + verification decision reasons.",
        "",
        "Recommended boundary after rebuild:",
        "`fused_candidates -> verify_candidates -> verified_candidates/uncertain_candidates/rejected_candidates -> rank verified candidates only`.",
        "",
        "## Stage-3 rule",
        "",
        "Ranking is frozen until Stage 2 passes. Once Stage 2 passes, rerun at least 15 adversarial cases and evaluate same-key top-1/top-k preservation without allowing ranking to resurrect rejected candidates.",
        "",
        "## Explicit non-goals",
        "",
        "- No backend imports.",
        "- No answer LLM.",
        "- No production code edits.",
        "- No benchmark changes to make failures disappear.",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("STAGE 2 FORENSIC GATE")
    print(f"Cases: {len(results)}")
    print(f"Expected candidate statuses: {dict(status_counts)}")
    print(f"Destructive same-key cases: {len(destructive_cases)}")
    if destructive_cases:
        print("Cases:", ", ".join(destructive_cases))
    print(f"Scope-to-zero cases: {len(stage4_scope_wipe_cases)}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
