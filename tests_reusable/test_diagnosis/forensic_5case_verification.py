from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

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


OUTPUT_DIR = Path("tests_reusable/test_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "forensic_5case_report.json"


CASES = (
    {
        "name": "MTECH_ELIGIBILITY",
        "question": "What are the eligibility requirements for regular M.Tech admission?",
        "target": "M.Tech",
        "request_type": "eligibility",
        "facets": (
            "eligibility",
            "requirements",
        ),
    },
    {
        "name": "MTECH_WRITTEN_TEST",
        "question": "Is there a written test for regular M.Tech admission?",
        "target": "M.Tech",
        "request_type": "written_test",
        "facets": (
            "written test",
            "admission",
        ),
    },
    {
        "name": "MSC_ELIGIBILITY",
        "question": "What are the eligibility requirements for M.Sc admission?",
        "target": "M.Sc",
        "request_type": "eligibility",
        "facets": (
            "eligibility",
            "requirements",
        ),
    },
    {
        "name": "PHD_ELIGIBILITY",
        "question": "What are the eligibility requirements for Ph.D admission?",
        "target": "Ph.D",
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


def make_frame(case: dict[str, Any]) -> SemanticQueryFrame:
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


def get_source(candidate: RetrievalCandidate) -> str:
    source = getattr(candidate, "source", None)
    if source:
        return str(source)

    metadata = getattr(
        getattr(candidate, "document", None),
        "metadata",
        {},
    ) or {}

    return str(
        metadata.get("source", "")
        or metadata.get("file_path", "")
        or ""
    )


def compact_text(text: str, limit: int = 350) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def get_text(candidate: RetrievalCandidate) -> str:
    document = getattr(candidate, "document", None)
    return str(
        getattr(document, "page_content", "")
        or ""
    )


def parse_relation(reasons: tuple[str, ...]) -> str:
    if "target_attribute_relation_supported" in reasons:
        return "supported"

    if "target_attribute_relation_weak_or_missing" in reasons:
        return "weak_or_missing"

    if "target_attribute_relation_not_grounded" in reasons:
        return "not_grounded"

    return "not_reported"


def parse_semantic_signal(reasons: tuple[str, ...]) -> str:
    if "semantic_compatibility_present" in reasons:
        return "present"

    if "semantic_compatibility_weak" in reasons:
        return "weak"

    return "not_reported"


def alignment_value(
    candidate: RetrievalCandidate,
    name: str,
) -> Any:
    alignment = getattr(
        candidate,
        "alignment",
        None,
    )

    if alignment is None:
        return None

    if isinstance(alignment, dict):
        return alignment.get(name)

    return getattr(
        alignment,
        name,
        None,
    )


def candidate_row(
    *,
    fused_rank: int,
    candidate: RetrievalCandidate,
    decision: Any,
    final_rank: int | None,
) -> dict[str, Any]:

    reasons = tuple(
        getattr(decision, "reasons", ())
        or ()
    )

    status = str(
        getattr(decision, "status", "")
        or ""
    )

    target_grounded = bool(
        getattr(
            decision,
            "target_grounded",
            False,
        )
    )

    attribute_grounded = bool(
        getattr(
            decision,
            "attribute_grounded",
            False,
        )
    )

    semantic_compatible = bool(
        getattr(
            decision,
            "semantic_compatible",
            False,
        )
    )

    coverage = float(
        getattr(
            decision,
            "coverage",
            0.0,
        )
        or 0.0
    )

    verifier_score = float(
        getattr(
            decision,
            "score",
            0.0,
        )
        or 0.0
    )

    relation = parse_relation(reasons)
    semantic_signal = parse_semantic_signal(reasons)

    weak_verified = (
        status == "verified"
        and (
            not target_grounded
            or not attribute_grounded
            or relation != "supported"
            or not semantic_compatible
            or coverage < 0.50
        )
    )

    return {
        "fused_rank": fused_rank,
        "final_rank": final_rank,
        "candidate_id": str(
            getattr(candidate, "document_id", "")
            or ""
        ),
        "source": get_source(candidate),
        "status": status,
        "target_grounded": target_grounded,
        "attribute_grounded": attribute_grounded,
        "target_attribute_relation": relation,
        "semantic_compatibility": (
            semantic_signal
        ),
        "semantic_compatible_bool": (
            semantic_compatible
        ),
        "coverage": round(coverage, 4),
        "verifier_score": round(
            verifier_score,
            4,
        ),
        "reasons": list(reasons),
        "weak_verified_flag": weak_verified,
        "candidate_final_score": round(
            float(
                getattr(
                    candidate,
                    "final_score",
                    0.0,
                )
                or 0.0
            ),
            4,
        ),
        "candidate_rrf_score": round(
            float(
                getattr(
                    candidate,
                    "rrf_score",
                    0.0,
                )
                or 0.0
            ),
            4,
        ),
        "alignment": {
            "program_match": alignment_value(
                candidate,
                "program_match",
            ),
            "entity_match": alignment_value(
                candidate,
                "entity_match",
            ),
            "topic_match": alignment_value(
                candidate,
                "topic_match",
            ),
            "attribute_match": alignment_value(
                candidate,
                "attribute_match",
            ),
            "intent_match": alignment_value(
                candidate,
                "intent_match",
            ),
            "scope_match": alignment_value(
                candidate,
                "scope_match",
            ),
            "coverage": alignment_value(
                candidate,
                "coverage",
            ),
            "semantic_match": alignment_value(
                candidate,
                "semantic_match",
            ),
        },
        "text": compact_text(
            get_text(candidate)
        ),
    }


def prepare_candidates(
    question: str,
    fused: tuple[RetrievalCandidate, ...],
) -> tuple[RetrievalCandidate, ...]:

    registry_module = __import__(
        "ai_platform.institutions.iitj.semantic_registry",
        fromlist=["SEMANTIC_REGISTRY"],
    )

    registry = getattr(
        registry_module,
        "SEMANTIC_REGISTRY",
    )

    aligned: list[RetrievalCandidate] = []

    for candidate in fused:
        (
            _,
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

    return tuple(aligned)


def run_case(
    case: dict[str, Any],
) -> dict[str, Any]:

    question = case["question"]

    frame = make_frame(case)

    dense = tuple(
        dense_retrieve(question)
    )

    bm25 = tuple(
        keyword_retrieve(question)
    )

    fused = tuple(
        fuse_ranked_lists(
            [dense, bm25],
            weights=(0.70, 0.30),
            primary_query=question,
            retrieval_queries=(question,),
        )
    )

    aligned = prepare_candidates(
        question,
        fused,
    )

    verification = verify_candidates(
        frame,
        aligned,
    )

    ranked = tuple(
        rank_candidates(
            question,
            verification.verified,
            top_k=len(verification.verified),
            query_frame=frame,
        )
    )

    final_rank_map = {
        candidate.document_id: index
        for index, candidate in enumerate(
            ranked,
            start=1,
        )
    }

    decision_map = {
        decision.candidate.document_id: decision
        for decision in verification.decisions
    }

    rows: list[dict[str, Any]] = []

    for fused_rank, candidate in enumerate(
        aligned,
        start=1,
    ):
        decision = decision_map.get(
            candidate.document_id
        )

        if decision is None:
            continue

        rows.append(
            candidate_row(
                fused_rank=fused_rank,
                candidate=candidate,
                decision=decision,
                final_rank=final_rank_map.get(
                    candidate.document_id
                ),
            )
        )

    context_package = build_final_context_package(
        question,
        ranked,
        max_candidates=8,
        max_units=18,
        max_context_chars=12000,
    )

    weak_verified = [
        row
        for row in rows
        if row["weak_verified_flag"]
    ]

    verified_target_missing = [
        row
        for row in rows
        if (
            row["status"] == "verified"
            and not row["target_grounded"]
        )
    ]

    verified_attribute_missing = [
        row
        for row in rows
        if (
            row["status"] == "verified"
            and not row["attribute_grounded"]
        )
    ]

    verified_relation_weak = [
        row
        for row in rows
        if (
            row["status"] == "verified"
            and row["target_attribute_relation"]
            != "supported"
        )
    ]

    verified_semantic_weak = [
        row
        for row in rows
        if (
            row["status"] == "verified"
            and row["semantic_compatibility"]
            == "weak"
        )
    ]

    verified_low_coverage = [
        row
        for row in rows
        if (
            row["status"] == "verified"
            and row["coverage"] < 0.50
        )
    ]

    return {
        "case": case,
        "corpus_chunks": corpus_document_count(),
        "dense_hits": len(dense),
        "bm25_hits": len(bm25),
        "rrf_candidates": len(fused),
        "verification_counts": {
            "verified": len(
                verification.verified
            ),
            "uncertain": len(
                verification.uncertain
            ),
            "rejected": len(
                verification.rejected
            ),
        },
        "ranked_candidates": len(
            ranked
        ),
        "final_context": {
            "ready": bool(
                context_package.ready_for_generation
            ),
            "sources": int(
                getattr(
                    context_package,
                    "source_count",
                    0,
                )
                or 0
            ),
            "selected_items": int(
                getattr(
                    context_package,
                    "selected_count",
                    0,
                )
                or 0
            ),
            "context_chars": len(
                context_package.context
            ),
            "context": context_package.context,
        },
        "forensic_counts": {
            "weak_verified": len(
                weak_verified
            ),
            "verified_target_missing": len(
                verified_target_missing
            ),
            "verified_attribute_missing": len(
                verified_attribute_missing
            ),
            "verified_relation_weak": len(
                verified_relation_weak
            ),
            "verified_semantic_weak": len(
                verified_semantic_weak
            ),
            "verified_low_coverage": len(
                verified_low_coverage
            ),
        },
        "candidates": rows,
    }


def print_candidate(
    row: dict[str, Any],
) -> None:

    warning = (
        " <<< WEAK_VERIFIED"
        if row["weak_verified_flag"]
        else ""
    )

    print(
        f"\n"
        f"[FUSED #{row['fused_rank']:02d}] "
        f"[FINAL #{str(row['final_rank'] or '-'):>2}]"
        f"  {row['status'].upper()}"
        f"{warning}"
    )

    print(
        f"  candidate_id       : "
        f"{row['candidate_id']}"
    )

    print(
        f"  source             : "
        f"{row['source']}"
    )

    print(
        f"  candidate_score    : "
        f"{row['candidate_final_score']}"
    )

    print(
        f"  candidate_rrf      : "
        f"{row['candidate_rrf_score']}"
    )

    print(
        f"  target_grounded    : "
        f"{row['target_grounded']}"
    )

    print(
        f"  attribute_grounded : "
        f"{row['attribute_grounded']}"
    )

    print(
        f"  target_attr_rel    : "
        f"{row['target_attribute_relation']}"
    )

    print(
        f"  semantic           : "
        f"{row['semantic_compatibility']} "
        f"(bool={row['semantic_compatible_bool']})"
    )

    print(
        f"  coverage           : "
        f"{row['coverage']}"
    )

    print(
        f"  verifier_score     : "
        f"{row['verifier_score']}"
    )

    print(
        f"  reasons            : "
        f"{', '.join(row['reasons'])}"
    )

    print(
        f"  alignment          : "
        f"{json.dumps(row['alignment'], sort_keys=True)}"
    )

    print(
        f"  text               : "
        f"{row['text']}"
    )


def print_case_report(
    result: dict[str, Any],
) -> None:

    case = result["case"]

    print()
    print("=" * 120)
    print(case["name"])
    print("=" * 120)

    print(
        f"Question           : "
        f"{case['question']}"
    )

    print(
        f"Target             : "
        f"{case['target']}"
    )

    print(
        f"Corpus chunks      : "
        f"{result['corpus_chunks']}"
    )

    print()
    print("RETRIEVAL")
    print(
        f"  Dense             : "
        f"{result['dense_hits']}"
    )
    print(
        f"  BM25              : "
        f"{result['bm25_hits']}"
    )
    print(
        f"  RRF               : "
        f"{result['rrf_candidates']}"
    )

    print()
    print("VERIFICATION")
    print(
        f"  Verified          : "
        f"{result['verification_counts']['verified']}"
    )
    print(
        f"  Uncertain         : "
        f"{result['verification_counts']['uncertain']}"
    )
    print(
        f"  Rejected          : "
        f"{result['verification_counts']['rejected']}"
    )

    print()
    print("FORENSIC WARNING COUNTS")

    for key, value in result[
        "forensic_counts"
    ].items():
        print(
            f"  {key:<28} : {value}"
        )

    print()
    print(
        "ALL FUSED CANDIDATES"
    )
    print(
        "Every candidate below is shown "
        "before the final top-k context selection."
    )

    for row in result["candidates"]:
        print_candidate(row)

    print()
    print(
        "FINAL CONTEXT -> ANSWER LLM INPUT"
    )
    print("-" * 120)
    print(
        result["final_context"]["context"]
    )
    print("-" * 120)

    print(
        f"Ready              : "
        f"{result['final_context']['ready']}"
    )

    print(
        f"Sources             : "
        f"{result['final_context']['sources']}"
    )

    print(
        f"Selected items      : "
        f"{result['final_context']['selected_items']}"
    )

    print(
        f"Context chars       : "
        f"{result['final_context']['context_chars']}"
    )


def main() -> None:

    print(
        "IITJ 5-CASE VERIFICATION FORENSICS"
    )
    print(
        "No answer LLM is called."
    )
    print(
        "Path: "
        "runtime retrieval -> RRF -> "
        "semantic alignment -> verification -> "
        "ranking -> final context"
    )

    all_results: list[dict[str, Any]] = []

    for case in CASES:
        result = run_case(case)

        print_case_report(
            result
        )

        all_results.append(
            result
        )

    payload = {
        "test": (
            "IITJ 5-case verification "
            "forensic diagnosis"
        ),
        "cases": len(CASES),
        "results": all_results,
    }

    REPORT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("AGGREGATE FORENSIC SUMMARY")
    print("=" * 120)

    for result in all_results:
        print(
            f"\n{result['case']['name']}"
        )

        print(
            f"  RRF={result['rrf_candidates']} "
            f"Verified={result['verification_counts']['verified']} "
            f"Uncertain={result['verification_counts']['uncertain']} "
            f"Rejected={result['verification_counts']['rejected']}"
        )

        for key, value in result[
            "forensic_counts"
        ].items():
            print(
                f"  {key:<28} : {value}"
            )

    print()
    print("=" * 120)
    print(
        f"JSON REPORT: {REPORT_JSON}"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
