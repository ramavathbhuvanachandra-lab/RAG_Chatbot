"""
Stage 2 — real verification audit.

Purpose
-------
Audit the real IITJ retrieval path at the exact boundary:

    fused_candidates
            ↓
    verification
            ↓
    verified / uncertain / rejected

This file is DIAGNOSTIC ONLY.
It does not modify production code and does not tune verification thresholds.

Run from the repository root:

    PYTHONPATH=. python tests_reusable/standalone/stage2_verification_audit.py

The report is printed to the terminal. Save the terminal output and use it
to make the targeted verification change in the next step.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ai_platform.core.graph.nodes import CoreNodes, default_dependencies


QUERY = "What are the M.Tech eligibility requirements?"
GOLD_SOURCE_SUFFIX = "/admissions/mtech_admissions.docx"


def _source(candidate: Any) -> str:
    value = getattr(candidate, "source", "")
    if value:
        return str(value)

    document = getattr(candidate, "document", candidate)
    metadata = getattr(document, "metadata", {}) or {}
    return str(metadata.get("source", "") or "")


def _text(candidate: Any) -> str:
    document = getattr(candidate, "document", candidate)
    return str(getattr(document, "page_content", "") or "")


def _id(candidate: Any) -> str:
    value = str(getattr(candidate, "document_id", "") or "")
    if value:
        return value
    return f"{_source(candidate)}|{_text(candidate)}"


def _decision_for_id(decision: Any) -> dict[str, Any]:
    if hasattr(decision, "to_dict"):
        data = decision.to_dict()
        if isinstance(data, dict):
            return data

    return {
        "status": getattr(decision, "status", ""),
        "accepted": getattr(decision, "accepted", False),
        "target_grounded": getattr(decision, "target_grounded", False),
        "attribute_grounded": getattr(decision, "attribute_grounded", False),
        "scope_compatible": getattr(decision, "scope_compatible", False),
        "semantic_compatible": getattr(decision, "semantic_compatible", False),
        "conflict_detected": getattr(decision, "conflict_detected", False),
        "coverage": getattr(decision, "coverage", 0.0),
        "score": getattr(decision, "score", 0.0),
        "reasons": list(getattr(decision, "reasons", ()) or ()),
        "candidate": getattr(decision, "candidate", None),
    }


def _candidate_from_decision(decision: Any) -> Any:
    return getattr(decision, "candidate", None)


def main() -> int:
    deps = default_dependencies()
    nodes = CoreNodes(deps)

    state: dict[str, Any] = {
        "question": QUERY,
        "resolved_question": QUERY,
        "chat_history": [],
    }

    understanding = nodes.understand_query_node(state)
    state.update(understanding)

    retrieval = nodes.hybrid_retrieve_node(state)
    state.update(retrieval)

    fused = nodes.fuse_retrieved_documents_node(state)
    state.update(fused)

    fused_candidates = tuple(state.get("fused_candidates", ()))

    if not fused_candidates:
        raise AssertionError("Stage 2 audit: fused_candidates is empty.")

    verified_rank = nodes.verify_and_rank_node(state)
    decisions_raw = tuple(
        verified_rank.get("verification_trace", ())
    )

    verified = tuple(verified_rank.get("verified_candidates", ()))
    uncertain = tuple(verified_rank.get("uncertain_candidates", ()))
    rejected = tuple(verified_rank.get("rejected_candidates", ()))

    print("=" * 92)
    print("STAGE 2 — REAL IITJ VERIFICATION AUDIT")
    print("=" * 92)
    print(f"QUERY: {QUERY}")
    print(f"FUSED CANDIDATES   : {len(fused_candidates)}")
    print(f"VERIFIED           : {len(verified)}")
    print(f"UNCERTAIN          : {len(uncertain)}")
    print(f"REJECTED           : {len(rejected)}")
    print(f"VERIFICATION TRACE : {len(decisions_raw)}")
    print()

    # Core conservation check. Recall rescue may add new verification
    # decisions, so trace length can exceed the original fused count.
    fused_ids = {_id(c) for c in fused_candidates}
    verified_ids = {_id(c) for c in verified}
    uncertain_ids = {_id(c) for c in uncertain}
    rejected_ids = {_id(c) for c in rejected}

    all_classified = verified_ids | uncertain_ids | rejected_ids

    print("--- BOUNDARY CONSERVATION ---")
    print(f"UNIQUE FUSED IDS       : {len(fused_ids)}")
    print(f"UNIQUE CLASSIFIED IDS  : {len(all_classified)}")
    print(f"FUSED IDS UNCLASSIFIED : {len(fused_ids - all_classified)}")
    print(f"CLASSIFIED NOT FUSED   : {len(all_classified - fused_ids)}")
    print()

    if fused_ids - all_classified:
        print("WARNING: Some fused candidates are not present in the final")
        print("verification buckets. Inspect recall-rescue/dedup behavior.")
    else:
        print("PASS: every fused candidate is represented in a final bucket.")

    # Gold-source integrity.
    gold_fused = [
        c for c in fused_candidates
        if _source(c).casefold().endswith(GOLD_SOURCE_SUFFIX)
    ]
    gold_verified = [
        c for c in verified
        if _source(c).casefold().endswith(GOLD_SOURCE_SUFFIX)
    ]
    gold_uncertain = [
        c for c in uncertain
        if _source(c).casefold().endswith(GOLD_SOURCE_SUFFIX)
    ]
    gold_rejected = [
        c for c in rejected
        if _source(c).casefold().endswith(GOLD_SOURCE_SUFFIX)
    ]

    print()
    print("--- GOLD-SOURCE CHECK ---")
    print(f"MTECH SOURCE IN FUSED    : {len(gold_fused)}")
    print(f"MTECH SOURCE VERIFIED    : {len(gold_verified)}")
    print(f"MTECH SOURCE UNCERTAIN   : {len(gold_uncertain)}")
    print(f"MTECH SOURCE REJECTED    : {len(gold_rejected)}")

    if gold_verified:
        print("PASS: actual M.Tech admissions source survives verification.")
    elif gold_fused and gold_rejected:
        print("FAIL: actual M.Tech admissions source was rejected.")
    elif gold_fused and gold_uncertain:
        print("REVIEW: actual M.Tech admissions source became uncertain.")
    else:
        print("FAIL: actual M.Tech admissions source did not reach verification.")

    # Analyze reasons from the verification trace when available.
    reason_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for raw in decisions_raw:
        data = raw if isinstance(raw, dict) else _decision_for_id(raw)
        status = str(data.get("status", "")).strip().casefold()
        if status:
            status_counts[status] += 1

        for reason in data.get("reasons", ()) or ():
            reason_counts[str(reason)] += 1

    print()
    print("--- DECISION STATUS COUNTS (TRACE) ---")
    for status, count in sorted(status_counts.items()):
        print(f"{status:14s}: {count}")

    print()
    print("--- REJECTION/DECISION REASONS ---")
    if reason_counts:
        for reason, count in reason_counts.most_common():
            print(f"{count:4d}  {reason}")
    else:
        print("No reason data available in verification_trace.")

    print()
    print("--- REJECTED CANDIDATES ---")
    if not rejected:
        print("None.")
    else:
        for index, candidate in enumerate(rejected, 1):
            print(f"\n[{index}] SOURCE")
            print(_source(candidate))
            print("TEXT")
            print(_text(candidate)[:450].replace("\n", " "))

    print()
    print("--- UNCERTAIN CANDIDATES ---")
    if not uncertain:
        print("None.")
    else:
        for index, candidate in enumerate(uncertain, 1):
            print(f"\n[{index}] SOURCE")
            print(_source(candidate))
            print("TEXT")
            print(_text(candidate)[:450].replace("\n", " "))

    print()
    print("=" * 92)
    print("STAGE 2 AUDIT COMPLETE")
    print("=" * 92)
    print("Do not change verification thresholds from this report alone.")
    print("Use the actual rejection reasons + gold-source result to make the")
    print("smallest targeted production change in retrieval/verification.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
