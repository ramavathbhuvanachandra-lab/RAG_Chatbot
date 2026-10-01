"""Real IITJ integration gate for the new core Node 1.

This gate deliberately uses:
- the user's actual IITJ corpus via the active backend retriever,
- the active vector store / BM25 stack,
- the active IITJ semantic registry at runtime,
- the new core Node 1 orchestration.

It does NOT use answer generation so the test remains deterministic enough
to diagnose retrieval/evidence wiring independently of the answer LLM.

It must fail rather than silently fall back to synthetic fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.nodes import CoreNodes, default_dependencies
from backend.core.query.understanding import (
    understand_query,
    understand_query_frame,
)


CORPUS = (
    ROOT
    / "data"
    / "data_iitj"
    / "iitj_rag_v1_docs_production"
)


QUERY = "What are the M.Tech eligibility requirements?"


def _source(document) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return str(metadata.get("source", "") or "").replace("\\", "/")
    return ""


def _text(document) -> str:
    return str(getattr(document, "page_content", "") or "").strip()


def _assert_real_corpus_loaded() -> None:
    assert CORPUS.is_dir(), f"REAL IITJ CORPUS NOT FOUND: {CORPUS}"

    real_files = list(CORPUS.rglob("*.docx"))
    assert len(real_files) >= 10, (
        "REAL IITJ CORPUS LOOKS INVALID: "
        f"found only {len(real_files)} DOCX files at {CORPUS}"
    )


def main() -> int:
    _assert_real_corpus_loaded()

    deps = default_dependencies()
    nodes = CoreNodes(deps)

    # ---------------------------------------------------------
    # 1. Real semantic query understanding
    # ---------------------------------------------------------
    query = understand_query(QUERY)
    frame = understand_query_frame(QUERY)

    assert query.original_query, "Canonical query is empty."
    assert frame.original_query, "Semantic query frame is empty."

    # ---------------------------------------------------------
    # 2. Real dense + BM25 retrieval from the active backend
    # ---------------------------------------------------------
    retrieval = nodes.hybrid_retrieve_node(
        {
            "resolved_question": QUERY,
            "retrieval_queries": (QUERY,),
        }
    )

    retrieval_results = tuple(retrieval["retrieval_results"])
    assert len(retrieval_results) == 2, retrieval_results
    dense_docs = tuple(retrieval_results[0])
    bm25_docs = tuple(retrieval_results[1])

    assert dense_docs, "REAL IITJ dense retrieval returned zero documents."
    assert bm25_docs, "REAL IITJ BM25 retrieval returned zero documents."

    # ---------------------------------------------------------
    # 3. Real RRF -> canonical retrieval candidates
    # ---------------------------------------------------------
    fused = nodes.fuse_retrieved_documents_node(
        {
            "question": QUERY,
            "resolved_question": QUERY,
            "retrieval_results": retrieval_results,
            "retrieval_weights": retrieval["retrieval_weights"],
            "retrieval_queries": retrieval["retrieval_queries"],
        }
    )

    fused_candidates = tuple(fused["fused_candidates"])
    assert fused_candidates, "RRF produced zero candidates."

    # ---------------------------------------------------------
    # 4. Real candidate verification + new ranking
    # ---------------------------------------------------------
    ranked = nodes.verify_and_rank_node(
        {
            "question": QUERY,
            "resolved_question": QUERY,
            "query": query,
            "query_frame": frame,
            "fused_candidates": fused_candidates,
        }
    )

    verified = tuple(ranked["verified_candidates"])
    ranked_candidates = tuple(ranked["ranked_candidates"])

    assert verified, (
        "Candidate verification rejected all REAL IITJ candidates."
    )
    assert ranked_candidates, (
        "New core ranking returned zero REAL IITJ candidates."
    )

    # ---------------------------------------------------------
    # 5. Evidence groups + local context + scope
    # ---------------------------------------------------------
    context = nodes.evidence_context_node(
        {
            "question": QUERY,
            "resolved_question": QUERY,
            "query": query,
            "ranked_candidates": ranked_candidates,
        }
    )

    evidence_docs = tuple(context["evidence_documents"])
    evidence_candidates = tuple(context["evidence_candidates"])

    assert evidence_docs, (
        "Evidence context became empty after grouping/local-context/scope."
    )
    assert evidence_candidates, (
        "Evidence candidate reconstruction produced zero candidates."
    )

    # ---------------------------------------------------------
    # 6. Evidence sufficiency + coverage
    # ---------------------------------------------------------
    assessment = nodes.assess_evidence_node(
        {
            "query": query,
            "ranked_candidates": evidence_candidates,
        }
    )

    coverage = nodes.assess_coverage_node(
        {
            "query": query,
            "ranked_candidates": evidence_candidates,
        }
    )

    evidence_status = str(
        assessment["evidence_status"]
    ).casefold()

    coverage_status = str(
        coverage["evidence_coverage_status"]
    ).casefold()

    assert evidence_status in {"supported", "partial"}, (
        f"REAL IITJ evidence unexpectedly {evidence_status}: "
        f"{assessment['evidence_assessment']}"
    )
    assert coverage_status in {"supported", "partial"}, (
        f"REAL IITJ coverage unexpectedly {coverage_status}: "
        f"{coverage['coverage_assessment']}"
    )

    # ---------------------------------------------------------
    # 7. Claims + E6 packaging must receive real evidence
    # ---------------------------------------------------------
    audit = nodes.audit_claims_node(
        {
            "query": query,
            "ranked_candidates": evidence_candidates,
        }
    )

    package = nodes.package_evidence_node(
        {
            "evidence_units": audit["evidence_units"],
            "claim_audit": audit["claim_audit"],
        }
    )

    assert package["ready_for_generation"] is True, (
        f"E6 package is not ready: {package['evidence_package']}"
    )
    final_context = str(
        package["final_evidence_context"]
    ).strip()
    assert final_context, "Final E6 evidence context is empty."

    # ---------------------------------------------------------
    # 8. Prove the path touched actual IITJ admission evidence.
    # ---------------------------------------------------------
    all_sources = {
        _source(candidate.document if hasattr(candidate, "document") else candidate)
        for candidate in ranked_candidates
    }

    all_text = "\n".join(
        _text(
            candidate.document
            if hasattr(candidate, "document")
            else candidate
        )
        for candidate in ranked_candidates
    ).casefold()

    assert any(
        source.casefold().endswith("/admissions/mtech_admissions.docx")
        for source in all_sources
    ), (
        "REAL IITJ Node 1 path did not surface the actual "
        "M.Tech admissions source among ranked candidates. "
        f"Sources={sorted(all_sources)[:20]}"
    )

    assert "eligibility" in all_text or "degree" in all_text, (
        "REAL IITJ ranked evidence lacks the expected eligibility/degree "
        "content. This gate refuses to pass on source-name-only matching."
    )

    print(
        "NODE 1 IITJ REAL INTEGRATION GATE: PASS "
        f"(dense={len(dense_docs)}; "
        f"bm25={len(bm25_docs)}; "
        f"fused={len(fused_candidates)}; "
        f"verified={len(verified)}; "
        f"ranked={len(ranked_candidates)}; "
        f"evidence={len(evidence_docs)}; "
        f"package_ready={package['ready_for_generation']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())