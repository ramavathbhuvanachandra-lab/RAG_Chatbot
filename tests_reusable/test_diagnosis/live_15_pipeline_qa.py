from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from ai_platform.core.graph.nodes import CoreNodes, default_dependencies


QUESTIONS = [
    ("mtech_regular_eligibility",
     "What are the eligibility requirements for regular M.Tech admission?"),

    ("mtech_percentage",
     "What is the minimum percentage required for M.Tech admission?"),

    ("mtech_gate",
     "Is there any GATE requirement for M.Tech admission?"),

    ("mtech_qualification",
     "What qualification is required for M.Tech admission?"),

    ("mtech_written_test",
     "Is a written test required for M.Tech admission?"),

    ("mtech_work_experience",
     "Is work experience required for M.Tech admission?"),

    ("msc_eligibility",
     "What are the eligibility requirements for M.Sc admission?"),

    ("msc_percentage",
     "What percentage is required for M.Sc admission?"),

    ("msc_jam",
     "Is IIT JAM required for M.Sc admission?"),

    ("msc_bachelor_route",
     "What bachelor's degree is required for M.Sc admission?"),

    ("phd_eligibility",
     "What are the eligibility requirements for PhD admission?"),

    ("phd_application_fee",
     "What is the PhD application fee?"),

    ("mba_steps",
     "What are the steps for MBA admission?"),

    ("registration_documents",
     "What documents are required for registration?"),

    ("registration_provisional",
     "What are the rules for provisional registration?"),
]


def clean(value) -> str:
    return " ".join(str(value or "").split())


def candidate_text(candidate) -> str:
    document = getattr(candidate, "document", candidate)
    return clean(getattr(document, "page_content", ""))


def candidate_source(candidate) -> str:
    document = getattr(candidate, "document", candidate)
    metadata = getattr(document, "metadata", {}) or {}
    return clean(
        metadata.get("source")
        or getattr(candidate, "source", "")
        or ""
    )


def decision_counts(verifier, frame, candidates):
    batch = verifier(frame, candidates)

    decisions = getattr(batch, "decisions", None)
    if decisions is None:
        decisions = ()

    counts = Counter()

    for decision in decisions:
        status = getattr(decision, "status", "")
        status_value = getattr(status, "value", status)
        counts[str(status_value)] += 1

    return batch, counts


def main():
    nodes = CoreNodes(default_dependencies())
    verifier = nodes.deps.verify_candidates

    reports = []

    print("=" * 110)
    print("LIVE IITJ — 15 CASE RETRIEVAL QA")
    print("AI PLATFORM ONLY")
    print("NO ANSWER LLM IS CALLED")
    print("=" * 110)

    for case_id, question in QUESTIONS:
        started = time.perf_counter()

        print("\n" + "=" * 110)
        print(case_id)
        print(question)
        print("=" * 110)

        state = {
            "question": question,
            "chat_history": (),
        }

        # ----------------------------------------------------------
        # 1. Conversation resolution
        # ----------------------------------------------------------
        state.update(
            nodes.resolve_conversation_node(state)
        )

        # ----------------------------------------------------------
        # 2. Query understanding
        # ----------------------------------------------------------
        state.update(
            nodes.understand_query_node(state)
        )

        query = state.get("query")
        frame = state.get("query_frame")

        print("\nQUERY UNDERSTANDING")
        print("resolved_question:",
              state.get("resolved_question"))
        print("semantic_query:",
              state.get("semantic_query"))
        print("retrieval_queries:",
              state.get("retrieval_queries"))
        print("target:",
              getattr(frame, "target", None))
        print("request_type:",
              getattr(frame, "request_type", None))

        # ----------------------------------------------------------
        # 3. Multi-intent planning
        # ----------------------------------------------------------
        state.update(
            nodes.plan_multi_intent_node(state)
        )

        print("\nINTENT")
        print("intent_count:",
              state.get("intent_count"))
        print("is_multi_intent:",
              state.get("is_multi_intent"))

        # ----------------------------------------------------------
        # 4. Hybrid retrieval
        # ----------------------------------------------------------
        state.update(
            nodes.hybrid_retrieve_node(state)
        )

        retrieval_results = tuple(
            state.get("retrieval_results", ()) or ()
        )

        dense_count = (
            len(retrieval_results[0])
            if len(retrieval_results) >= 1
            else 0
        )

        bm25_count = (
            len(retrieval_results[1])
            if len(retrieval_results) >= 2
            else 0
        )

        lexical_count = sum(
            len(items)
            for items in retrieval_results[2::3]
        )

        print("\nHYBRID RETRIEVAL")
        print("retrieval_result_lists:",
              len(retrieval_results))
        print("primary_dense_hits:",
              dense_count)
        print("primary_bm25_hits:",
              bm25_count)
        print("lexical_recall_hits:",
              lexical_count)

        # ----------------------------------------------------------
        # 5. RRF
        # ----------------------------------------------------------
        state.update(
            nodes.fuse_retrieved_documents_node(state)
        )

        fused = tuple(
            state.get("fused_candidates", ()) or ()
        )

        print("\nRRF")
        print("fused_candidates:", len(fused))

        # ----------------------------------------------------------
        # 6. Verification + ranking
        # ----------------------------------------------------------
        state.update(
            nodes.verify_and_rank_node(state)
        )

        ranked = tuple(
            state.get("ranked_candidates", ()) or ()
        )

        # Run the exact active verifier once for explicit diagnostics.
        batch, verification_counts = decision_counts(
            verifier,
            frame,
            fused,
        )

        print("\nVERIFICATION")
        print(dict(verification_counts))

        print("\nTOP RANKED CANDIDATES")

        top_ranked = []

        for rank, candidate in enumerate(ranked[:8], start=1):
            text = candidate_text(candidate)
            source = candidate_source(candidate)
            score = getattr(candidate, "final_score", 0.0)

            item = {
                "rank": rank,
                "score": float(score or 0.0),
                "source": source,
                "text": text[:1200],
            }

            top_ranked.append(item)

            print(f"\n[{rank}] score={item['score']:.6f}")
            print("source:", source)
            print("text:", item["text"])

        # ----------------------------------------------------------
        # 7. Final context builder
        # ----------------------------------------------------------
        state.update(
            nodes.evidence_context_node(state)
        )

        package = state.get("final_context_package")

        final_context = clean(
            getattr(package, "context", "")
        )

        print("\nFINAL PRE-LLM CONTEXT")
        print("-" * 100)
        print(final_context)
        print("-" * 100)

        elapsed = time.perf_counter() - started

        ready = bool(
            final_context.strip()
        )

        print("\nCASE RESULT")
        print("ranked_candidates:", len(ranked))
        print("context_non_empty:", ready)
        print("context_chars:", len(final_context))
        print("elapsed_seconds:", round(elapsed, 3))

        reports.append({
            "case_id": case_id,
            "question": question,
            "resolved_question": state.get("resolved_question"),
            "semantic_query": state.get("semantic_query"),
            "retrieval_queries": list(
                state.get("retrieval_queries", ()) or ()
            ),
            "intent_count": state.get("intent_count"),
            "is_multi_intent": state.get("is_multi_intent"),
            "dense_hits": dense_count,
            "bm25_hits": bm25_count,
            "lexical_hits": lexical_count,
            "fused_candidates": len(fused),
            "verification_counts":
                dict(verification_counts),
            "ranked_candidates": len(ranked),
            "top_ranked": top_ranked,
            "final_context": final_context,
            "context_chars": len(final_context),
            "context_non_empty": ready,
            "elapsed_seconds": round(elapsed, 3),
        })

    # --------------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------------
    print("\n" + "=" * 110)
    print("AGGREGATE")
    print("=" * 110)

    total = len(reports)

    context_ready = sum(
        1 for r in reports
        if r["context_non_empty"]
    )

    ranked_ready = sum(
        1 for r in reports
        if r["ranked_candidates"] > 0
    )

    verified_total = sum(
        r["verification_counts"].get("verified", 0)
        for r in reports
    )

    rejected_total = sum(
        r["verification_counts"].get("rejected", 0)
        for r in reports
    )

    uncertain_total = sum(
        r["verification_counts"].get("uncertain", 0)
        for r in reports
    )

    print("cases:", total)
    print(f"ranked_non_empty: {ranked_ready}/{total}")
    print(f"final_context_non_empty: {context_ready}/{total}")
    print("verified_candidates:", verified_total)
    print("rejected_candidates:", rejected_total)
    print("uncertain_candidates:", uncertain_total)

    if total:
        print(
            "average_context_chars:",
            round(
                sum(r["context_chars"] for r in reports) / total,
                1,
            ),
        )
        print(
            "average_elapsed_seconds:",
            round(
                sum(r["elapsed_seconds"] for r in reports) / total,
                3,
            ),
        )

    output = Path(
        "tests_reusable/test_diagnosis/live_15_pipeline_report.json"
    )

    output.write_text(
        json.dumps(
            reports,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nREPORT:")
    print(output)

    if context_ready != total:
        raise AssertionError(
            f"Only {context_ready}/{total} cases reached final context."
        )


if __name__ == "__main__":
    main()
