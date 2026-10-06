from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ai_platform.runtime.retrieval import (
    corpus_document_count,
    dense_retrieve,
    keyword_retrieve,
)

from ai_platform.core.retrieval.rrf import fuse_ranked_lists
from ai_platform.core.retrieval.semantic_alignment import (
    align_query_to_document,
)
from ai_platform.core.retrieval.ranking import rank_candidates
from ai_platform.core.retrieval.verification import verify_candidates
from ai_platform.core.retrieval.final_context_builder import (
    build_final_context_package,
)
from ai_platform.core.retrieval.contracts import RetrievalCandidate
from ai_platform.core.query.models import (
    QueryFacet,
    RetrievalRequirement,
    SemanticQueryFrame,
)


QUESTIONS = (
    {
        "name": "MTECH_ELIGIBILITY",
        "question": (
            "What are the eligibility requirements for regular M.Tech admission?"
        ),
        "target": "M.Tech",
        "request_type": "eligibility",
        "facets": (
            "eligibility",
            "requirements",
        ),
    },
    {
        "name": "REGISTRATION_DOCUMENTS",
        "question": "What documents are required for registration?",
        "target": "registration",
        "request_type": "registration",
        "facets": (
            "documents",
            "registration",
        ),
    },
)


def make_frame(case: dict) -> SemanticQueryFrame:
    return SemanticQueryFrame(
        original_query=case["question"],
        normalized_query=case["question"],
        semantic_query=case["question"],
        target=case["target"],
        request_type=case["request_type"],
        facets=tuple(
            QueryFacet(
                name="requested_attribute",
                value=value,
            )
            for value in case["facets"]
        ),
        requirement=RetrievalRequirement(
            mode="standard",
            require_target_alignment=True,
            require_attribute_alignment=True,
            allow_partial_evidence=True,
            reject_explicit_conflict=True,
        ),
    )


def source(candidate: RetrievalCandidate) -> str:
    if candidate.source:
        return str(candidate.source)

    document = candidate.document
    metadata = getattr(document, "metadata", {}) or {}
    return str(metadata.get("source", "") or "")


def preview(candidate: RetrievalCandidate, limit: int = 500) -> str:
    text = str(
        getattr(candidate.document, "page_content", "") or ""
    ).strip()

    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def build_verified_ranked_context(case: dict):
    question = case["question"]
    frame = make_frame(case)

    dense = tuple(dense_retrieve(question))
    bm25 = tuple(keyword_retrieve(question))

    fused = fuse_ranked_lists(
        [dense, bm25],
        weights=(0.70, 0.30),
        primary_query=question,
        retrieval_queries=(question,),
    )

    registry_module = __import__(
        "ai_platform.institutions.iitj.semantic_registry",
        fromlist=["SEMANTIC_REGISTRY"],
    )

    registry = getattr(
        registry_module,
        "SEMANTIC_REGISTRY",
    )

    aligned = []

    for candidate in fused:
        (
            _query_meaning,
            document_meaning,
            alignment,
        ) = align_query_to_document(
            question,
            candidate.document,
            registry,
        )

        aligned.append(
            replace(
                candidate,
                meaning=document_meaning,
                alignment=alignment,
            )
        )

    aligned = tuple(aligned)

    verification = verify_candidates(
        frame,
        aligned,
    )

    ranked = tuple(
        rank_candidates(
            question,
            verification.verified,
            top_k=10,
            query_frame=frame,
        )
    )

    package = build_final_context_package(
        question,
        ranked,
        max_candidates=8,
        max_units=18,
        max_context_chars=12000,
    )

    return {
        "dense": dense,
        "bm25": bm25,
        "fused": fused,
        "verification": verification,
        "ranked": ranked,
        "package": package,
    }


def run_case(case: dict) -> None:
    result = build_verified_ranked_context(case)

    dense = result["dense"]
    bm25 = result["bm25"]
    fused = result["fused"]
    verification = result["verification"]
    ranked = result["ranked"]
    package = result["package"]

    print()
    print("=" * 100)
    print(case["name"])
    print("=" * 100)

    print("QUESTION")
    print(case["question"])

    print()
    print("CORPUS")
    print("Canonical chunks:", corpus_document_count())

    print()
    print("RETRIEVAL")
    print("Dense hits:", len(dense))
    print("BM25 hits:", len(bm25))
    print("RRF candidates:", len(fused))

    print()
    print("VERIFICATION")
    print("Verified:", len(verification.verified))
    print("Uncertain:", len(verification.uncertain))
    print("Rejected:", len(verification.rejected))

    print()
    print("TOP VERIFIED/RANKED CANDIDATES")
    for index, candidate in enumerate(ranked, start=1):
        decision = next(
            (
                item
                for item in verification.decisions
                if item.candidate.document_id == candidate.document_id
            ),
            None,
        )

        print()
        print(f"[{index}] candidate_id={candidate.document_id}")
        print(f"    score={candidate.final_score:.4f}")
        print(f"    source={source(candidate)}")
        if decision is not None:
            print(f"    verification={decision.status}")
            print(
                "    reasons="
                + ", ".join(decision.reasons)
            )
        print(f"    text={preview(candidate)}")

    print()
    print("FINAL CONTEXT -> THIS IS WHAT THE ANSWER LLM WOULD RECEIVE")
    print("-" * 100)
    print(package.context)
    print("-" * 100)

    print()
    print("CONTEXT SUMMARY")
    print("Ready:", package.ready_for_generation)
    print("Sources:", package.source_count)
    print("Selected items:", package.selected_count)
    print("Context chars:", len(package.context))
    print("Context token estimate:",
          getattr(package, "context_tokens_estimate", "n/a"))

    assert ranked, "No verified/ranked candidates survived."
    assert package.ready_for_generation, "Final context package is not ready."
    assert package.context.strip(), "Final LLM context is empty."


def main() -> None:
    print("LIVE IITJ PRE-LLM CONTEXT CHECK")
    print("No answer LLM is called.")
    print("Path: runtime retrieval -> RRF -> verification -> ranking -> context")

    for case in QUESTIONS:
        run_case(case)

    print()
    print("=" * 100)
    print("PASS: BOTH REAL QUESTIONS REACHED A NON-EMPTY FINAL CONTEXT")
    print("=" * 100)


if __name__ == "__main__":
    main()
