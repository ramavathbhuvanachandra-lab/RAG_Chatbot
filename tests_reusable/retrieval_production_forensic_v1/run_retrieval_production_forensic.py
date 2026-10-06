#!/usr/bin/env python3
"""
Production RAG retrieval/evidence forensic gate.

Purpose
-------
Run a small adversarial set through the live CoreNodes pipeline and expose the
first meaningful failure boundary without changing production code.

Pipeline inspected:
    query understanding
      -> dense + BM25 + lexical recall
      -> weighted RRF
      -> verification
      -> deterministic ranking
      -> evidence/context construction
      -> sufficiency + coverage
      -> claim audit
      -> final context package
      -> optional answer generation + grounding/guard

The checks are intentionally generic. Test cases contain only evaluation
fixtures; no institution-specific knowledge is embedded in production logic.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


CASES: dict[str, dict[str, Any]] = {
    "R01": {
        "level": "exact-targeted",
        "question": "What are the M.Tech eligibility requirements?",
        "targets": ("M.Tech",),
        "facets": ("eligibility", "requirements"),
        "forbid": (),
        "purpose": "Exact target + attribute; tests target/attribute/relation integrity.",
    },
    "R02": {
        "level": "exact-numeric",
        "question": "What is the Ph.D. application processing fee?",
        "targets": ("Ph.D.",),
        "facets": ("fee", "application"),
        "forbid": (),
        "purpose": "Numeric/fee request; tests precise target and monetary evidence selection.",
    },
    "R03": {
        "level": "targetless-action",
        "question": "What documents do I need to bring for registration?",
        "targets": (),
        "facets": ("documents", "registration", "bring"),
        "forbid": (),
        "purpose": "Targetless request; tests actionable document evidence rather than generic mentions.",
    },
    "R04": {
        "level": "comparison",
        "question": "What is the difference between M.Tech and M.Sc admission eligibility at IIT Jodhpur?",
        "targets": ("M.Tech", "M.Sc"),
        "facets": ("admission", "eligibility"),
        "forbid": (),
        "purpose": "Two-target comparison; tests evidence mixing and coverage of both sides.",
    },
    "R05": {
        "level": "adversarial-scope",
        "question": "Is JAM required for M.Sc admission?",
        "targets": ("M.Sc", "JAM"),
        "facets": ("admission", "eligibility", "JAM"),
        "forbid": ("GATE",),
        "purpose": "Near-neighbor adversarial query; tests wrong-exam contamination and local scope.",
    },
}


STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "please", "the", "this", "to", "what", "when", "which", "who",
    "with", "you", "your", "iit", "jodhpur", "between", "difference", "than",
}


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def norm(value: Any) -> str:
    return clean(value).casefold()


def tokens(value: Any) -> set[str]:
    out = set()
    for token in re.findall(r"[a-z0-9]+", norm(value)):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        out.add(token)
    return out


def doc_text(document: Any) -> str:
    if isinstance(document, Mapping):
        return clean(document.get("page_content"))
    return clean(getattr(document, "page_content", ""))


def doc_source(document: Any) -> str:
    if isinstance(document, Mapping):
        md = document.get("metadata") or {}
    else:
        md = getattr(document, "metadata", {}) or {}
    if isinstance(md, Mapping):
        for key in ("source", "source_path", "path", "url"):
            value = clean(md.get(key))
            if value:
                return value
    return ""


def candidate_document(candidate: Any) -> Any:
    return getattr(candidate, "document", candidate)


def candidate_source(candidate: Any) -> str:
    source = clean(getattr(candidate, "source", ""))
    return source or doc_source(candidate_document(candidate))


def candidate_id(candidate: Any) -> str:
    for name in ("document_id", "chunk_id"):
        value = clean(getattr(candidate, name, ""))
        if value:
            return value
    return f"{candidate_source(candidate)}|{doc_text(candidate_document(candidate))[:160]}"


def alignment_value(candidate: Any, name: str) -> float | None:
    alignment = getattr(candidate, "alignment", None)
    if alignment is None:
        return None
    value = getattr(alignment, name, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def provenance_value(candidate: Any, name: str) -> Any:
    p = getattr(candidate, "provenance", None)
    return getattr(p, name, None) if p is not None else None


def verification_status(candidate: Any) -> str:
    for name in ("verification_status", "status"):
        value = clean(getattr(candidate, name, ""))
        if value:
            return value.casefold()
    return "verified"


def target_from_frame(query_frame: Any) -> str:
    target = getattr(query_frame, "target", None)
    return clean(getattr(target, "text", target))


def frame_fields(query_frame: Any) -> dict[str, Any]:
    req = getattr(query_frame, "requirement", None)
    return {
        "target": target_from_frame(query_frame),
        "request_type": clean(getattr(query_frame, "request_type", "")),
        "requirement_mode": clean(getattr(req, "mode", "")),
        "require_target": bool(getattr(req, "require_target_alignment", False)) if req else False,
        "require_attribute": bool(getattr(req, "require_attribute_alignment", False)) if req else False,
        "require_scope": bool(getattr(req, "require_scope_alignment", False)) if req else False,
        "facets": [clean(x) for x in (getattr(query_frame, "facets", ()) or ()) if clean(x)],
        "qualifiers": [clean(getattr(x, "value", x)) for x in (getattr(query_frame, "qualifiers", ()) or ()) if clean(getattr(x, "value", x))],
    }


def summarize_candidate(candidate: Any, rank: int | None = None) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": candidate_id(candidate),
        "source": candidate_source(candidate),
        "score": getattr(candidate, "final_score", None),
        "rrf_score": provenance_value(candidate, "rrf_score"),
        "rrf_rank": provenance_value(candidate, "rrf_rank"),
        "semantic_match": alignment_value(candidate, "semantic_match"),
        "coverage": alignment_value(candidate, "coverage"),
        "target_match": alignment_value(candidate, "target_match"),
        "attribute_match": alignment_value(candidate, "attribute_match"),
        "scope_match": alignment_value(candidate, "scope_match"),
        "verification": verification_status(candidate),
        "preview": doc_text(candidate_document(candidate))[:700],
    }


def summarize_documents(items: Sequence[Any], top_n: int = 8) -> list[dict[str, Any]]:
    out = []
    for index, item in enumerate(items[:top_n], start=1):
        if hasattr(item, "document"):
            out.append(summarize_candidate(item, index))
        else:
            out.append({
                "rank": index,
                "source": doc_source(item),
                "preview": doc_text(item)[:700],
            })
    return out


def context_text(package: Any, state: Mapping[str, Any]) -> str:
    for obj in (package, state):
        if obj is None:
            continue
        if isinstance(obj, Mapping):
            value = clean(obj.get("context") or obj.get("final_evidence_context"))
        else:
            value = clean(getattr(obj, "context", ""))
        if value:
            return value
    return ""


def package_items(package: Any) -> Sequence[Any]:
    if package is None:
        return ()
    if isinstance(package, Mapping):
        return tuple(package.get("items") or ())
    return tuple(getattr(package, "items", ()) or ())


def context_fit(case: Mapping[str, Any], text: str, query_frame: Any) -> dict[str, Any]:
    lower = norm(text)
    q_tokens = tokens(case["question"])
    observed_tokens = set(re.findall(r"[a-z0-9]+", lower))
    q_overlap = (len(q_tokens & observed_tokens) / len(q_tokens)) if q_tokens else 0.0

    def target_present(target: str) -> bool:
        compact = re.sub(r"[^a-z0-9]", "", norm(target))
        return bool(compact) and compact in re.sub(r"[^a-z0-9]", "", lower)

    targets = tuple(case.get("targets") or ())
    target_hits = {target: target_present(target) for target in targets}
    facet_hits = {}
    for facet in case.get("facets") or ():
        facet_hits[facet] = norm(facet) in lower
    forbid_hits = {term: norm(term) in lower for term in (case.get("forbid") or ())}

    leakage = any(forbid_hits.values())
    target_rate = (sum(target_hits.values()) / len(target_hits)) if target_hits else None
    facet_rate = (sum(facet_hits.values()) / len(facet_hits)) if facet_hits else None

    if targets and any(not hit for hit in target_hits.values()):
        status = "TARGET_COVERAGE_REVIEW"
    elif leakage:
        status = "FORBIDDEN_SCOPE_LEAK"
    elif facet_hits and (facet_rate or 0.0) < 0.50:
        status = "FACET_COVERAGE_REVIEW"
    elif q_overlap < 0.35:
        status = "LOW_QUERY_FIT"
    else:
        status = "FIT"

    return {
        "status": status,
        "query_token_overlap": round(q_overlap, 3),
        "target_hits": target_hits,
        "facet_hits": facet_hits,
        "forbidden_term_hits": forbid_hits,
        "query_frame": frame_fields(query_frame),
    }


def compact_package_items(package: Any, max_items: int = 8) -> list[dict[str, Any]]:
    result = []
    for index, item in enumerate(package_items(package)[:max_items], start=1):
        result.append({
            "rank": index,
            "evidence_id": clean(getattr(item, "evidence_id", "")),
            "source": clean(getattr(item, "source", "")),
            "heading": clean(getattr(item, "heading", "")),
            "text_preview": clean(getattr(item, "text", ""))[:700],
        })
    return result


def first_failure(case: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    fit = result.get("context_fit") or {}
    if fit.get("status") == "FORBIDDEN_SCOPE_LEAK":
        return "FINAL_CONTEXT_SCOPE_LEAK"
    if fit.get("status") in {"TARGET_COVERAGE_REVIEW", "FACET_COVERAGE_REVIEW", "LOW_QUERY_FIT"}:
        return "FINAL_CONTEXT_QUALITY"

    verification = result.get("verification") or {}
    if verification.get("verified_count", 0) == 0:
        return "VERIFICATION_DROP"

    retrieval = result.get("retrieval") or {}
    if retrieval.get("fused_count", 0) == 0:
        return "RETRIEVAL_RECALL"

    if result.get("package", {}).get("selected_count", 0) == 0:
        return "PACKAGING_DROP"

    coverage = result.get("coverage") or {}
    if str(coverage.get("status", "")).casefold() in {"insufficient", "unsupported"}:
        return "EVIDENCE_COVERAGE"

    return "NONE"


def run_case(case_id: str, nodes: Any, with_answer: bool) -> dict[str, Any]:
    case = CASES[case_id]
    started = time.perf_counter()
    state: dict[str, Any] = {
        "question": case["question"],
        "resolved_question": case["question"],
    }
    result: dict[str, Any] = {"id": case_id, "case": case, "stages": {}}

    def step(name: str, fn, current: Mapping[str, Any]) -> dict[str, Any]:
        value = fn(dict(current))
        if value:
            state.update(value)
            result["stages"][name] = {"ok": True}
        else:
            state.update({})
            result["stages"][name] = {"ok": True}
        return state

    try:
        t = time.perf_counter()
        step("understand_query", nodes.understand_query_node, state)
        q = state.get("query")
        frame = state.get("query_frame")
        result["plan"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "semantic_query": clean(state.get("semantic_query")),
            "retrieval_queries": list(state.get("retrieval_queries") or ()),
            "query_confidence": state.get("query_confidence"),
            "query_resolution_state": clean(state.get("query_resolution_state")),
            "frame": frame_fields(frame) if frame is not None else {},
        }

        t = time.perf_counter()
        step("hybrid_retrieve", nodes.hybrid_retrieve_node, state)
        result["retrieval"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "queries": list(state.get("retrieval_queries") or ()),
            "dense_counts": [len(x or ()) for x in (state.get("retrieval_results") or ())[0::2]],
            "bm25_counts": [len(x or ()) for x in (state.get("retrieval_results") or ())[1::2]],
            "lexical_recall_count": state.get("lexical_recall_count", 0),
            "fused_count": 0,
        }

        t = time.perf_counter()
        step("fuse", nodes.fuse_retrieved_documents_node, state)
        fused = tuple(state.get("fused_candidates") or ())
        result["retrieval"]["fused_count"] = len(fused)
        result["retrieval"]["fused_top"] = summarize_documents(fused, 6)
        result["retrieval"]["fuse_elapsed_ms"] = round((time.perf_counter() - t) * 1000, 1)

        t = time.perf_counter()
        step("verify_and_rank", nodes.verify_and_rank_node, state)
        verified = tuple(state.get("verified_candidates") or ())
        ranked = tuple(state.get("ranked_candidates") or ())
        rejected = tuple(state.get("rejected_candidates") or ())
        result["verification"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "verified_count": len(verified),
            "rejected_count": len(rejected),
            "uncertain_count": len(tuple(state.get("uncertain_candidates") or ())),
            "ranked_count": len(ranked),
            "top_ranked": [summarize_candidate(c, i) for i, c in enumerate(ranked[:8], start=1)],
            "lexical_fallback_used": bool(state.get("lexical_fallback_used", False)),
            "verification_trace": list(state.get("verification_trace") or ())[:8],
        }

        t = time.perf_counter()
        step("assess_evidence", nodes.assess_evidence_node, state)
        assessment = state.get("evidence_assessment")
        result["evidence_assessment"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "status": clean(state.get("evidence_status")),
            "score": state.get("evidence_score"),
            "relevant_documents": state.get("relevant_evidence_documents"),
        }

        t = time.perf_counter()
        step("assess_coverage", nodes.assess_coverage_node, state)
        coverage = state.get("coverage_assessment")
        result["coverage"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "status": clean(state.get("evidence_coverage_status")),
            "question_type": clean(state.get("evidence_question_type")),
        }

        t = time.perf_counter()
        step("evidence_context", nodes.evidence_context_node, state)
        package_like = state.get("final_context_package") or state.get("evidence_package")
        final_context = context_text(package_like, state)
        evidence_candidates = tuple(state.get("evidence_candidates") or ())
        result["context"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "evidence_candidate_count": len(evidence_candidates),
            "evidence_document_count": len(tuple(state.get("evidence_documents") or ())),
            "final_context_chars": len(final_context),
            "preview": final_context[:1800],
        }
        result["context_fit"] = context_fit(case, final_context, frame)
        result["context_package_items"] = compact_package_items(package_like)

        t = time.perf_counter()
        step("audit_claims", nodes.audit_claims_node, state)
        audit = state.get("claim_audit")
        result["claim_audit"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "status": clean(getattr(audit, "status", "")),
            "claims": len(tuple(getattr(audit, "claims", ()) or ())),
            "matches": len(tuple(getattr(audit, "matches", ()) or ())),
            "supported": list(getattr(audit, "supported_claim_ids", ()) or ()),
            "partial": list(getattr(audit, "partial_claim_ids", ()) or ()),
            "unsupported": list(getattr(audit, "unsupported_claim_ids", ()) or ()),
            "conflicting": list(getattr(audit, "conflicting_claim_ids", ()) or ()),
        }

        t = time.perf_counter()
        step("package", nodes.package_evidence_node, state)
        package = state.get("evidence_package") or state.get("final_context_package")
        result["package"] = {
            "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
            "status": clean(getattr(package, "status", state.get("evidence_package_status", ""))),
            "ready_for_generation": bool(state.get("ready_for_generation", getattr(package, "ready_for_generation", False))),
            "selected_count": int(getattr(package, "selected_count", len(package_items(package))) or 0),
            "source_count": int(getattr(package, "source_count", 0) or 0),
            "context_chars": len(context_text(package, state)),
            "items": compact_package_items(package),
        }

        if with_answer:
            t = time.perf_counter()
            step("answer", nodes.answer_node, state)
            generated = state.get("answer_generation")
            result["answer"] = {
                "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
                "answer": clean(state.get("answer")),
                "grounding_status": clean(state.get("answer_grounding_status")),
                "guard_status": clean(state.get("answer_guard_status")),
                "guard_fallback_used": bool(state.get("answer_guard_fallback_used", False)),
                "model_calls": state.get("model_calls", getattr(generated, "model_calls", 0)),
            }

        result["diagnosis"] = {"first_failure": first_failure(case, result), "status": "PASS"}
        if result["diagnosis"]["first_failure"] != "NONE":
            result["diagnosis"]["status"] = "REVIEW"

    except Exception as exc:
        result["diagnosis"] = {
            "first_failure": "RUNTIME_ERROR",
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    result["elapsed_ms_total"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="Case id; repeatable.")
    parser.add_argument("--with-answer", action="store_true", help="Also run the final answer/grounding/guard path.")
    parser.add_argument("--out", default="reports/retrieval_production_forensic", help="Output directory.")
    args = parser.parse_args()

    case_ids = args.case or list(CASES)

    from ai_platform.core.graph.nodes import CoreNodes

    nodes = CoreNodes()

    print("=" * 118)
    print("PRODUCTION RAG RETRIEVAL / FINAL-CONTEXT FORENSIC GATE")
    print(f"CASES={len(case_ids)} | ANSWER={'ON' if args.with_answer else 'OFF'} | NO PRODUCTION MODIFICATIONS")
    print("=" * 118)

    reports = []
    for index, case_id in enumerate(case_ids, start=1):
        case = CASES[case_id]
        print(f"[{index:02d}/{len(case_ids):02d}] {case_id} {case['level']}: {case['question']}")
        item = run_case(case_id, nodes, args.with_answer)
        reports.append(item)

        plan = item.get("plan", {})
        retrieval = item.get("retrieval", {})
        verification = item.get("verification", {})
        fit = item.get("context_fit", {})
        package = item.get("package", {})
        diagnosis = item.get("diagnosis", {})

        print(
            f"  PLAN      target={plan.get('frame', {}).get('target','')} "
            f"request={plan.get('frame', {}).get('request_type','')} "
            f"mode={plan.get('frame', {}).get('requirement_mode','')}"
        )
        print(
            f"  RETRIEVAL dense={retrieval.get('dense_counts',[])} "
            f"bm25={retrieval.get('bm25_counts',[])} lexical={retrieval.get('lexical_recall_count',0)} "
            f"fused={retrieval.get('fused_count',0)}"
        )
        print(
            f"  VERIFY    verified={verification.get('verified_count',0)} "
            f"rejected={verification.get('rejected_count',0)} ranked={verification.get('ranked_count',0)} "
            f"fallback={verification.get('lexical_fallback_used',False)}"
        )
        print(
            f"  CONTEXT   items={package.get('selected_count',0)} chars={package.get('context_chars',0)} "
            f"fit={fit.get('status','UNKNOWN')} query_fit={fit.get('query_token_overlap','?')}"
        )
        print(
            f"  DIAGNOSIS first_failure={diagnosis.get('first_failure')} "
            f"status={diagnosis.get('status')} total={item.get('elapsed_ms_total',0)} ms"
        )
        if args.with_answer:
            answer = item.get("answer", {})
            print(
                f"  ANSWER    grounding={answer.get('grounding_status')} "
                f"guard={answer.get('guard_status')} model_calls={answer.get('model_calls')}"
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"retrieval_production_forensic_{stamp}.json"
    summary_path = out_dir / f"retrieval_production_forensic_{stamp}_summary.txt"

    summary_lines = [
        "PRODUCTION RAG RETRIEVAL / FINAL-CONTEXT FORENSIC GATE",
        f"CASES={len(reports)} ANSWER={'ON' if args.with_answer else 'OFF'}",
        "",
    ]
    counts: dict[str, int] = {}
    for item in reports:
        diag = item.get("diagnosis", {}).get("first_failure", "ERROR")
        counts[diag] = counts.get(diag, 0) + 1
        summary_lines.append(
            f"{item.get('id')}: {diag} | context_fit={item.get('context_fit',{}).get('status','UNKNOWN')} "
            f"| verified={item.get('verification',{}).get('verified_count',0)} "
            f"| context_chars={item.get('package',{}).get('context_chars',0)}"
        )
    summary_lines += ["", f"DIAGNOSIS_COUNTS={counts}"]

    json_path.write_text(json.dumps({"cases": reports, "diagnosis_counts": counts}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("=" * 118)
    print(f"JSON REPORT: {json_path}")
    print(f"SUMMARY TXT: {summary_path}")
    print(f"DIAGNOSIS COUNTS: {counts}")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
