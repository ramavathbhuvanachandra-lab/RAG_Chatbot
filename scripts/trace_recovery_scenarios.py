"""Targeted stage-by-stage tracer for Scenarios A, B, D, E, F.

Pins down the EXACT first stage where ground truth evidence is lost,
demoted, rejected, or transformed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ai_platform.core.query.understanding import understand_query_frame
from ai_platform.core.query.planner import build_query_plan, plan_to_frame
from ai_platform.runtime.engine import get_app_graph
from ai_platform.core.retrieval.evidence_matcher import analyze_candidate
from ai_platform.core.retrieval.verification import verify_candidate
from ai_platform.core.retrieval.ranking import score_candidate, _source, _document_text

TARGET_SCENARIOS = [
    {
        "id": "A",
        "question": "What are the ITEP fees for academic year 2026-27, including tuition fee, admission fee, convocation fee, semester fee, refundable deposit, and experiential and professional development fee?",
        "gt_markers": ["₹50,000", "₹32,250", "3,800", "4,000", "8,000", "2,000"],
        "gt_source_snippet": "fees_and_finance",
    },
    {
        "id": "B",
        "question": "What are the applicable short-term accommodation rates for visitors or executive students at IIT Jodhpur, including single and double occupancy, with and without bedding?",
        "gt_markers": ["450", "750", "600", "900", "without bedding"],
        "gt_source_snippet": "fees_and_finance",
    },
    {
        "id": "D",
        "question": "What are the charges for longer accommodation bookings at IIT Jodhpur, and what basic facilities are provided in the hostel?",
        "gt_markers": ["2175", "11 days", "bedding", "air-conditioning"],
        "gt_source_snippet": "fees_and_finance",
    },
    {
        "id": "E",
        "question": "When does the postgraduate admission cycle typically occur for Semester 1 and Semester 2 at IIT Jodhpur?",
        "gt_markers": ["April-May", "November-December", "Postgraduate admissions"],
        "gt_source_snippet": "general_admissions",
    },
    {
        "id": "F",
        "question": "What does the Institute's Security and Transport office handle at IIT Jodhpur?",
        "gt_markers": ["Office of Security & Transports", "24/7 security patrols", "surveillance systems"],
        "gt_source_snippet": "overview",
    },
]


def trace_scenario(sc: dict):
    qid = sc["id"]
    q = sc["question"]
    gt_markers = sc["gt_markers"]
    gt_src = sc["gt_source_snippet"]

    print("\n" + "=" * 70)
    print(f"TRACING SCENARIO {qid}: '{q}'")
    print("=" * 70)

    # 1. Query Plan & Understanding
    plan = build_query_plan(q)
    frame = plan_to_frame(plan)
    print(f"\n[Stage 1: Query Plan]")
    print(f"Target: '{frame.target}' | Req: {getattr(frame, 'requirement', None)}")
    print(f"Facets: {getattr(frame, 'facets', ())}")
    print(f"Queries: {getattr(plan, 'retrieval_queries', (q,))}")

    # 2. Invoke Graph up to nodes
    graph = get_app_graph()
    state = {"question": q, "chat_history": ()}
    out = graph.invoke(state)

    fused = out.get("fused_candidates", ())
    verified = out.get("verified_candidates", ())
    ranked = out.get("ranked_candidates", ())
    pkg = out.get("evidence_package")
    items = getattr(pkg, "items", ())
    context = getattr(pkg, "context", "")
    ans = out.get("answer", "")

    # Check Fused candidates for GT
    print(f"\n[Stage 2: Fusion ({len(fused)} candidates)]")
    fused_gt = []
    for idx, c in enumerate(fused):
        text = _document_text(c)
        src = _source(c)
        hits = [m for m in gt_markers if m.casefold() in text.casefold()]
        if hits or (gt_src in src.casefold()):
            fused_gt.append((idx + 1, src, hits, text[:90].replace("\n", " ")))
    print(f"Found {len(fused_gt)} GT candidates in Fusion:")
    for rank, src, hits, snippet in fused_gt[:4]:
        print(f"  Rank #{rank}: {src} | hits={hits} | snippet: {snippet}...")

    # Check Verification for GT
    print(f"\n[Stage 3: Verification ({len(verified)} verified)]")
    verified_gt = []
    for idx, c in enumerate(verified):
        text = _document_text(c)
        src = _source(c)
        hits = [m for m in gt_markers if m.casefold() in text.casefold()]
        if hits or (gt_src in src.casefold()):
            verified_gt.append((idx + 1, src, hits))
    print(f"Found {len(verified_gt)} GT candidates in Verified:")
    for rank, src, hits in verified_gt[:4]:
        print(f"  Verified #{rank}: {src} | hits={hits}")

    # Check Ranked candidates for GT
    print(f"\n[Stage 4: Ranking ({len(ranked)} ranked)]")
    ranked_gt = []
    for idx, c in enumerate(ranked):
        text = _document_text(c)
        src = _source(c)
        hits = [m for m in gt_markers if m.casefold() in text.casefold()]
        if hits or (gt_src in src.casefold()):
            sc_val = score_candidate(q, c, query_frame=frame)
            ranked_gt.append((idx + 1, sc_val, src, hits, text[:90].replace("\n", " ")))
    print(f"Found {len(ranked_gt)} GT candidates in Ranked:")
    for rank, sc_val, src, hits, snippet in ranked_gt[:4]:
        print(f"  Rank #{rank} (score={sc_val:.2f}): {src} | hits={hits} | snippet: {snippet}...")

    # Check Final Evidence Package (Context sent to LLM)
    print(f"\n[Stage 5: Final Context ({len(items)} items, {len(context)} chars)]")
    pkg_gt = []
    for idx, it in enumerate(items):
        hits = [m for m in gt_markers if m.casefold() in it.text.casefold()]
        if hits or (gt_src in it.source.casefold()):
            pkg_gt.append((idx + 1, it.source, hits, it.text[:90].replace("\n", " ")))
    print(f"Found {len(pkg_gt)} GT items in Final Package:")
    for pos, src, hits, snippet in pkg_gt:
        print(f"  Item [{pos}]: {src} | hits={hits} | snippet: {snippet}...")

    # Check top 3 items in Package
    print(f"\nTop 3 items in Final Context sent to LLM:")
    for idx, it in enumerate(items[:3]):
        print(f"  [{idx+1}] {it.source}")
        print(f"      Text: {it.text[:120].replace(chr(10), ' ')}...")

    # Answer Output
    print(f"\n[Stage 6: Answer Generated]")
    print(f"{ans}\n")


def main():
    for sc in TARGET_SCENARIOS:
        trace_scenario(sc)


if __name__ == "__main__":
    main()
