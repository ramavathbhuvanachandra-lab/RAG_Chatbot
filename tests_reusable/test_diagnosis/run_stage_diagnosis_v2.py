#!/usr/bin/env python3
"""
V1 Stage Diagnosis V2
---------------------

Purpose:
    Measure the ACTUAL output of each ai_platform/core stage without using
    backend code and without calling the answer LLM.

This is a diagnostic harness only. It does not modify production code.

What it measures:
    - Dense found relevant evidence?
    - BM25 found relevant evidence?
    - RRF retained it?
    - Dedup retained it?
    - Verification: verified / uncertain / rejected?
    - Ranking preserved it?
    - Evidence grouping changed the candidate set?
    - Local context expansion added candidates?
    - Scope filtering removed candidates?
    - Claim audit status and supported/unsupported/conflicting claims?
    - Evidence packaging status?
    - Exact final evidence context and traceability?

Important:
    The benchmark's positive/negative markers are only a diagnostic oracle.
    "Possible competitor" is NOT automatically "wrong evidence".
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import re
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "tests_reusable" / "test_diagnosis" / "reports" / "stage_diagnosis_v2"
FINAL_CONTEXT_ROOT = REPORT_ROOT / "final_context"

NODE_SEQUENCE = (
    "resolve_conversation_node",
    "understand_query_node",
    "plan_multi_intent_node",
    "hybrid_retrieve_node",
    "fuse_retrieved_documents_node",
    "verify_and_rank_node",
    "evidence_context_node",
    "assess_evidence_node",
    "assess_coverage_node",
    "audit_claims_node",
    "package_evidence_node",
)


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize(text: Any) -> str:
    value = clean(text).casefold()
    value = value.replace("\\", "/")
    return re.sub(r"\s+", " ", value).strip()


def unwrap_document(obj: Any) -> Any:
    document = getattr(obj, "document", None)
    return document if document is not None else obj


def source_of(obj: Any) -> str:
    doc = unwrap_document(obj)
    metadata = getattr(doc, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("source") or metadata.get("path") or metadata.get("url")
        if value:
            return clean(value)
    return ""


def text_of(obj: Any) -> str:
    doc = unwrap_document(obj)
    return clean(getattr(doc, "page_content", ""))


def document_id_of(obj: Any) -> str:
    candidates = (
        getattr(obj, "document_id", None),
        getattr(obj, "id", None),
    )

    metadata = getattr(unwrap_document(obj), "metadata", None)
    if isinstance(metadata, Mapping):
        candidates += (
            metadata.get("document_id"),
            metadata.get("id"),
            metadata.get("chunk_id"),
        )

    for value in candidates:
        if clean(value):
            return clean(value)

    source = source_of(obj)
    content = text_of(obj)
    digest = hashlib.sha1(
        f"{source}\0{content}".encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    return f"anon:{digest}"


def candidate_key(obj: Any) -> str:
    return document_id_of(obj)


def preview(obj: Any, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", text_of(obj))
    return text[:limit]


def marker_hits(text: str, markers: Sequence[str]) -> list[str]:
    normalized = normalize(text)
    return [marker for marker in markers if normalize(marker) in normalized]


def relevance_label(obj: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    source = source_of(obj)
    text = text_of(obj)

    hints = tuple(case.get("expected_source_hints", ()))
    positives = tuple(case.get("positive_markers", ()))
    negatives = tuple(case.get("negative_markers", ()))

    source_hits = [
        hint for hint in hints if normalize(hint) in normalize(source)
    ]
    positive_hits = marker_hits(text, positives)
    negative_hits = marker_hits(text, negatives)

    expected = bool(source_hits) or len(positive_hits) >= 2
    possible_competitor = (
        bool(negative_hits)
        and not expected
    )

    if expected:
        label = "expected"
    elif possible_competitor:
        label = "possible_competitor"
    else:
        label = "other"

    return {
        "label": label,
        "key": candidate_key(obj),
        "source": source,
        "source_hint_hits": source_hits,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
    }


def describe(obj: Any, case: Mapping[str, Any], rank: int | None = None) -> dict[str, Any]:
    record = relevance_label(obj, case)
    record.update(
        {
            "rank": rank,
            "text_preview": preview(obj),
        }
    )

    # Preserve useful candidate-level fields when available.
    for attr in (
        "score",
        "rrf_score",
        "dense_score",
        "bm25_score",
        "original_rank",
        "decision",
        "reason",
        "status",
        "confidence",
    ):
        value = getattr(obj, attr, None)
        if value is not None and isinstance(value, (str, int, float, bool)):
            record[attr] = value

    return record


def safe_json_value(obj: Any, depth: int = 0) -> Any:
    if depth > 3:
        return repr(obj)[:500]

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Mapping):
        return {
            str(k): safe_json_value(v, depth + 1)
            for k, v in list(obj.items())[:80]
        }

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [
            safe_json_value(v, depth + 1)
            for v in list(obj)[:80]
        ]

    if dataclasses.is_dataclass(obj):
        return safe_json_value(dataclasses.asdict(obj), depth + 1)

    if hasattr(obj, "to_dict"):
        try:
            return safe_json_value(obj.to_dict(), depth + 1)
        except Exception:
            pass

    result: dict[str, Any] = {}
    for name in (
        "status",
        "score",
        "question_type",
        "strong_documents",
        "partial_documents",
        "relevant_documents",
        "unique_sources",
        "reasons",
        "supported_claim_ids",
        "unsupported_claim_ids",
        "conflicting_claim_ids",
        "ready_for_generation",
        "context",
    ):
        if hasattr(obj, name):
            try:
                result[name] = safe_json_value(getattr(obj, name), depth + 1)
            except Exception:
                pass

    if result:
        return result

    return repr(obj)[:1000]


def unique_keys(items: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = candidate_key(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def describe_collection(
    name: str,
    items: Sequence[Any],
    case: Mapping[str, Any],
    top_n: int = 10,
) -> dict[str, Any]:
    values = list(items or ())
    keys = unique_keys(values)

    return {
        "name": name,
        "count": len(values),
        "unique_count": len(keys),
        "items": [
            describe(item, case, rank=i)
            for i, item in enumerate(values[:top_n], start=1)
        ],
        "all_keys": keys,
    }


def find_relevant(items: Sequence[Any], case: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = list(items or ())
    findings = []
    for rank, item in enumerate(values, start=1):
        info = relevance_label(item, case)
        if info["label"] == "expected":
            findings.append(
                {
                    "rank": rank,
                    **info,
                    "text_preview": preview(item),
                }
            )
    return findings


def first_relevant(items: Sequence[Any], case: Mapping[str, Any]) -> dict[str, Any] | None:
    matches = find_relevant(items, case)
    return matches[0] if matches else None


def candidate_map(items: Sequence[Any]) -> dict[str, Any]:
    return {candidate_key(item): item for item in items or ()}


def keys_added(before: Sequence[Any], after: Sequence[Any]) -> list[str]:
    before_keys = set(unique_keys(before))
    return [key for key in unique_keys(after) if key not in before_keys]


def keys_removed(before: Sequence[Any], after: Sequence[Any]) -> list[str]:
    after_keys = set(unique_keys(after))
    return [key for key in unique_keys(before) if key not in after_keys]


def resolve_nodes_module():
    errors = []
    for module_name in (
        "ai_platform.core.graph.nodes",
        "ai_platform.core.nodes",
    ):
        try:
            module = importlib.import_module(module_name)
            if not module.__name__.startswith("ai_platform.core"):
                continue
            return module
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not import the standalone ai_platform/core nodes.\n"
        + "\n".join(errors)
    )


def resolve_node_callable(module: Any, name: str):
    fn = getattr(module, name, None)
    if callable(fn):
        return fn
    raise AttributeError(f"{module.__name__}.{name} is unavailable")


def resolve_core_instance(module: Any):
    lazy = getattr(module, "_nodes", None)
    if callable(lazy):
        try:
            return lazy()
        except Exception:
            pass

    cls = getattr(module, "CoreNodes", None)
    if cls is not None:
        try:
            return cls()
        except Exception:
            pass

    return None


def summarize_query(state: Mapping[str, Any]) -> dict[str, Any]:
    query = state.get("query")
    result = {
        "question": clean(state.get("question")),
        "resolved_question": clean(state.get("resolved_question")),
        "semantic_query": clean(state.get("semantic_query")),
        "retrieval_queries": list(state.get("retrieval_queries") or ()),
        "query_confidence": state.get("query_confidence"),
        "query_resolution_state": clean(state.get("query_resolution_state")),
    }

    if query is not None:
        result["query_contract"] = safe_json_value(query)

    frame = state.get("query_frame")
    if frame is not None:
        result["query_frame"] = safe_json_value(frame)

    return result


def trace_hybrid(state: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    queries = list(state.get("retrieval_queries") or ())
    lanes = list(state.get("retrieval_results") or ())
    lane_records = []

    if queries and len(lanes) == len(queries) * 3:
        pattern = ("dense", "bm25", "lexical")
    elif queries and len(lanes) == len(queries) * 2:
        pattern = ("dense", "bm25")
    else:
        pattern = None

    if pattern:
        for q_index, query in enumerate(queries):
            for offset, lane_type in enumerate(pattern):
                index = q_index * len(pattern) + offset
                docs = list(lanes[index] or ())
                lane_records.append(
                    {
                        "query_index": q_index,
                        "query": query,
                        "lane": lane_type,
                        "count": len(docs),
                        "relevant": first_relevant(docs, case),
                        "items": [
                            describe(doc, case, rank=i)
                            for i, doc in enumerate(docs[:10], start=1)
                        ],
                    }
                )
    else:
        for index, docs in enumerate(lanes):
            values = list(docs or ())
            lane_records.append(
                {
                    "lane_index": index,
                    "count": len(values),
                    "relevant": first_relevant(values, case),
                    "items": [
                        describe(doc, case, rank=i)
                        for i, doc in enumerate(values[:10], start=1)
                    ],
                }
            )

    return {
        "query_count": len(queries),
        "retrieval_stream_count": len(lanes),
        "retrieval_weights": list(state.get("retrieval_weights") or ()),
        "lexical_recall_count": state.get("lexical_recall_count"),
        "lanes": lane_records,
    }


def trace_verification(state: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    fused = list(state.get("fused_candidates") or ())
    verified = list(state.get("verified_candidates") or ())
    uncertain = list(state.get("uncertain_candidates") or ())
    rejected = list(state.get("rejected_candidates") or ())
    ranked = list(state.get("ranked_candidates") or ())

    fused_keys = set(unique_keys(fused))

    def status_for_key(key: str) -> str:
        if key in set(unique_keys(verified)):
            return "verified"
        if key in set(unique_keys(uncertain)):
            return "uncertain"
        if key in set(unique_keys(rejected)):
            return "rejected"
        return "missing"

    relevant_fused = [
        item
        for item in fused
        if relevance_label(item, case)["label"] == "expected"
    ]

    relevant_decisions = []
    for item in relevant_fused:
        key = candidate_key(item)
        relevant_decisions.append(
            {
                "key": key,
                "source": source_of(item),
                "fused_rank": unique_keys(fused).index(key) + 1 if key in unique_keys(fused) else None,
                "verification_status": status_for_key(key),
                "ranked_rank": (
                    unique_keys(ranked).index(key) + 1
                    if key in unique_keys(ranked)
                    else None
                ),
                "text_preview": preview(item),
            }
        )

    return {
        "fused_count": len(fused),
        "verified_count": len(verified),
        "uncertain_count": len(uncertain),
        "rejected_count": len(rejected),
        "ranked_count": len(ranked),
        "relevant_candidate_decisions": relevant_decisions,
        "verification_trace": safe_json_value(
            state.get("verification_trace") or ()
        ),
        "lexical_fallback_used": bool(state.get("lexical_fallback_used")),
        "lexical_fallback_candidates": state.get("lexical_fallback_candidates"),
    }


def internal_evidence_breakdown(
    core: Any,
    state: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Split evidence_context_node into its actual internal operations using
    the same dependency methods that the production node invokes.

    This is diagnostic-only and does not modify the node.
    """
    if core is None or not hasattr(core, "deps"):
        return {"status": "unavailable", "reason": "CoreNodes deps unavailable"}

    deps = core.deps

    required = (
        "build_evidence_groups",
        "expand_group_context",
        "flatten_evidence_groups",
        "filter_scope_conflicts",
        "canonical_chunks_provider",
    )
    missing = [name for name in required if not callable(getattr(deps, name, None))]
    if missing:
        return {
            "status": "unavailable",
            "reason": f"Missing dependency methods: {missing}",
        }

    ranked_candidates = tuple(state.get("ranked_candidates") or ())
    ranked_docs = tuple(
        getattr(candidate, "document", candidate)
        for candidate in ranked_candidates
    )

    try:
        groups = list(deps.build_evidence_groups(ranked_docs))
        groups_before = len(groups)

        canonical_chunks = tuple(deps.canonical_chunks_provider(state) or ())

        if canonical_chunks:
            expanded_groups = list(
                deps.expand_group_context(
                    groups,
                    canonical_chunks,
                )
            )
        else:
            expanded_groups = groups

        grouped_docs = tuple(
            deps.flatten_evidence_groups(
                expanded_groups,
                max_documents=40,
            )
        )

        scope_filtered = tuple(
            deps.filter_scope_conflicts(
                state["query"],
                grouped_docs,
                query_semantics=state["query"],
            )
        )

        grouped_keys = unique_keys(grouped_docs)
        scoped_keys = unique_keys(scope_filtered)

        return {
            "status": "ok",
            "input_ranked_docs": len(ranked_docs),
            "group_count": groups_before,
            "canonical_chunk_count": len(canonical_chunks),
            "grouped_document_count": len(grouped_docs),
            "scope_filtered_document_count": len(scope_filtered),
            "grouped_added_keys_vs_ranked": [
                key for key in grouped_keys
                if key not in set(unique_keys(ranked_docs))
            ],
            "scope_removed_keys": [
                key for key in grouped_keys
                if key not in set(scoped_keys)
            ],
            "grouped_relevant": first_relevant(grouped_docs, case),
            "scope_filtered_relevant": first_relevant(scope_filtered, case),
            "grouped_documents": [
                describe(doc, case, rank=i)
                for i, doc in enumerate(grouped_docs[:20], start=1)
            ],
            "scope_filtered_documents": [
                describe(doc, case, rank=i)
                for i, doc in enumerate(scope_filtered[:20], start=1)
            ],
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


def trace_audit(state: Mapping[str, Any]) -> dict[str, Any]:
    audit = state.get("claim_audit")
    claims = list(state.get("claims") or ())
    units = list(state.get("evidence_units") or ())

    result = {
        "claim_count": len(claims),
        "evidence_unit_count": len(units),
        "claim_audit": safe_json_value(audit),
        "claim_audit_status": clean(state.get("claim_audit_status")),
    }

    if audit is not None:
        for name in (
            "supported_claim_ids",
            "unsupported_claim_ids",
            "conflicting_claim_ids",
        ):
            value = getattr(audit, name, None)
            if value is not None:
                result[name] = list(value)

    return result


def trace_package(
    state: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    package = state.get("evidence_package")
    context = clean(state.get("final_evidence_context"))

    evidence_units = list(state.get("evidence_units") or ())

    traceable_units = 0
    unit_sources = []
    for unit in evidence_units:
        unit_text = text_of(unit)
        if not unit_text:
            continue

        normalized_context = normalize(context)
        normalized_unit = normalize(unit_text)

        # Packaging may sanitize/shorten text. Use a bounded prefix probe.
        probe = normalized_unit[:140]
        matched = bool(probe and probe in normalized_context)

        if matched:
            traceable_units += 1

        unit_sources.append(
            {
                "source": source_of(unit),
                "traceable_in_final_context": matched,
                "text_preview": preview(unit),
            }
        )

    return {
        "package_status": clean(state.get("evidence_package_status")),
        "ready_for_generation": bool(state.get("ready_for_generation")),
        "final_context_characters": len(context),
        "final_context_present": bool(context),
        "evidence_unit_count": len(evidence_units),
        "traceable_evidence_units": traceable_units,
        "traceability_rate": (
            traceable_units / len(evidence_units)
            if evidence_units
            else None
        ),
        "package_object": safe_json_value(package),
        "evidence_units": unit_sources,
    }


def run_case(
    case: Mapping[str, Any],
    nodes_module: Any,
    core: Any,
) -> dict[str, Any]:
    question = clean(case["question"])
    state: dict[str, Any] = {
        "question": question,
        "resolved_question": question,
        "chat_history": [],
        "messages": [],
    }

    result: dict[str, Any] = {
        "case_id": case["id"],
        "question": question,
        "stages": {},
        "diagnosis": {},
    }

    for node_name in NODE_SEQUENCE:
        try:
            node = resolve_node_callable(nodes_module, node_name)
            delta = node(state)
            if delta is None:
                delta = {}
            if not isinstance(delta, Mapping):
                raise TypeError(
                    f"{node_name} returned {type(delta).__name__}, expected mapping"
                )

            before = dict(state)
            state.update(dict(delta))

            stage: dict[str, Any] = {
                "status": "ok",
                "state_keys_added": sorted(set(state) - set(before)),
            }

            if node_name in {
                "resolve_conversation_node",
                "understand_query_node",
                "plan_multi_intent_node",
            }:
                stage["query"] = summarize_query(state)

            elif node_name == "hybrid_retrieve_node":
                stage["hybrid"] = trace_hybrid(state, case)

            elif node_name == "fuse_retrieved_documents_node":
                fused = list(state.get("fused_candidates") or ())
                stage["output"] = describe_collection(
                    "fused_candidates",
                    fused,
                    case,
                    top_n=15,
                )
                stage["relevant"] = first_relevant(fused, case)

            elif node_name == "verify_and_rank_node":
                stage["verification"] = trace_verification(state, case)

            elif node_name == "evidence_context_node":
                groups = list(state.get("evidence_groups") or ())
                evidence_docs = list(state.get("evidence_documents") or ())
                evidence_candidates = list(state.get("evidence_candidates") or ())

                stage["evidence_groups_count"] = len(groups)
                stage["evidence_documents"] = describe_collection(
                    "evidence_documents",
                    evidence_docs,
                    case,
                    top_n=20,
                )
                stage["evidence_candidates"] = describe_collection(
                    "evidence_candidates",
                    evidence_candidates,
                    case,
                    top_n=20,
                )
                stage["internal_breakdown"] = internal_evidence_breakdown(
                    core,
                    state,
                    case,
                )

            elif node_name == "assess_evidence_node":
                assessment = state.get("evidence_assessment")
                stage["assessment"] = safe_json_value(assessment)
                stage["evidence_status"] = clean(state.get("evidence_status"))
                stage["evidence_score"] = state.get("evidence_score")
                stage["relevant_evidence_documents"] = state.get(
                    "relevant_evidence_documents"
                )

            elif node_name == "assess_coverage_node":
                coverage = state.get("coverage_assessment")
                stage["assessment"] = safe_json_value(coverage)
                stage["coverage_status"] = clean(
                    state.get("evidence_coverage_status")
                )
                stage["question_type"] = clean(
                    state.get("evidence_question_type")
                )

            elif node_name == "audit_claims_node":
                stage["audit"] = trace_audit(state)

            elif node_name == "package_evidence_node":
                stage["package"] = trace_package(state, case)

                FINAL_CONTEXT_ROOT.mkdir(parents=True, exist_ok=True)
                final_path = FINAL_CONTEXT_ROOT / f"{case['id']}.txt"
                final_path.write_text(
                    clean(state.get("final_evidence_context")),
                    encoding="utf-8",
                )
                stage["package"]["final_context_file"] = str(final_path)

            result["stages"][node_name] = stage

        except Exception as exc:
            result["stages"][node_name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
            result["diagnosis"]["first_failure_stage"] = node_name
            break

    # ---------------------------------------------------------
    # Exact candidate flow summary
    # ---------------------------------------------------------
    stage_map = result["stages"]

    hybrid = stage_map.get("hybrid_retrieve_node", {}).get("hybrid", {})
    dense_found = False
    bm25_found = False

    for lane in hybrid.get("lanes", []):
        if lane.get("lane") == "dense" and lane.get("relevant"):
            dense_found = True
        if lane.get("lane") == "bm25" and lane.get("relevant"):
            bm25_found = True

    fused = list(state.get("fused_candidates") or ())
    ranked = list(state.get("ranked_candidates") or ())
    verified = list(state.get("verified_candidates") or ())
    uncertain = list(state.get("uncertain_candidates") or ())
    rejected = list(state.get("rejected_candidates") or ())
    evidence_candidates = list(state.get("evidence_candidates") or ())
    evidence_documents = list(state.get("evidence_documents") or ())

    relevant_fused = first_relevant(fused, case)
    relevant_ranked = first_relevant(ranked, case)
    relevant_evidence = first_relevant(evidence_candidates, case)
    relevant_evidence_doc = first_relevant(evidence_documents, case)

    relevant_key = None
    if relevant_fused:
        relevant_key = relevant_fused["key"]

    verification_status = None
    if relevant_key:
        if relevant_key in set(unique_keys(verified)):
            verification_status = "verified"
        elif relevant_key in set(unique_keys(uncertain)):
            verification_status = "uncertain"
        elif relevant_key in set(unique_keys(rejected)):
            verification_status = "rejected"
        else:
            verification_status = "missing"

    package_info = stage_map.get("package_evidence_node", {}).get("package", {})
    internal = stage_map.get("evidence_context_node", {}).get(
        "internal_breakdown", {}
    )

    stage3_removed = internal.get("scope_removed_keys", [])
    group_added = internal.get("grouped_added_keys_vs_ranked", [])

    result["diagnosis"].update(
        {
            "first_failure_stage": result["diagnosis"].get("first_failure_stage"),
            "dense_found_relevant": dense_found,
            "bm25_found_relevant": bm25_found,
            "rrf_retained_relevant": bool(relevant_fused),
            "verification_status_of_relevant": verification_status,
            "verification_counts": {
                "verified": len(verified),
                "uncertain": len(uncertain),
                "rejected": len(rejected),
            },
            "ranking_preserved_relevant": bool(relevant_ranked),
            "relevant_rank_after_rrf": (
                relevant_fused.get("rank") if relevant_fused else None
            ),
            "relevant_rank_after_ranking": (
                relevant_ranked.get("rank") if relevant_ranked else None
            ),
            "evidence_boundary_preserved_relevant": bool(relevant_evidence),
            "evidence_document_preserved_relevant": bool(relevant_evidence_doc),
            "evidence_group_added_candidate_count": len(group_added),
            "scope_filter_removed_candidate_count": len(stage3_removed),
            "claim_audit_status": clean(
                stage_map.get("audit_claims_node", {})
                .get("audit", {})
                .get("claim_audit_status")
            ),
            "package_ready_for_generation": bool(
                package_info.get("ready_for_generation")
            ),
            "final_context_non_empty": bool(
                package_info.get("final_context_present")
            ),
            "final_context_traceability_rate": package_info.get(
                "traceability_rate"
            ),
        }
    )

    return result


def build_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {
        "cases": len(results),
        "dense_found_relevant": sum(
            bool(r["diagnosis"].get("dense_found_relevant"))
            for r in results
        ),
        "bm25_found_relevant": sum(
            bool(r["diagnosis"].get("bm25_found_relevant"))
            for r in results
        ),
        "rrf_retained_relevant": sum(
            bool(r["diagnosis"].get("rrf_retained_relevant"))
            for r in results
        ),
        "verification_verified": sum(
            r["diagnosis"].get("verification_status_of_relevant") == "verified"
            for r in results
        ),
        "verification_uncertain": sum(
            r["diagnosis"].get("verification_status_of_relevant") == "uncertain"
            for r in results
        ),
        "verification_rejected": sum(
            r["diagnosis"].get("verification_status_of_relevant") == "rejected"
            for r in results
        ),
        "verification_missing": sum(
            r["diagnosis"].get("verification_status_of_relevant") == "missing"
            for r in results
        ),
        "ranking_preserved_relevant": sum(
            bool(r["diagnosis"].get("ranking_preserved_relevant"))
            for r in results
        ),
        "evidence_boundary_preserved": sum(
            bool(r["diagnosis"].get("evidence_boundary_preserved_relevant"))
            for r in results
        ),
        "package_ready": sum(
            bool(r["diagnosis"].get("package_ready_for_generation"))
            for r in results
        ),
        "final_context_non_empty": sum(
            bool(r["diagnosis"].get("final_context_non_empty"))
            for r in results
        ),
        "first_failure_counts": {},
    }

    for result in results:
        key = result["diagnosis"].get("first_failure_stage") or "none"
        summary["first_failure_counts"][key] = (
            summary["first_failure_counts"].get(key, 0) + 1
        )

    return summary


def write_reports(results: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    payload = {
        "scope": "ai_platform/core",
        "backend_imports": [],
        "with_llm": False,
        "metadata": dict(meta),
        "summary": build_summary(results),
        "results": results,
        "note": (
            "Diagnostic-only output. The relevance oracle uses benchmark "
            "markers and expected source hints. Possible competitors are not "
            "automatically classified as wrong."
        ),
    }

    json_path = REPORT_ROOT / "diagnosis_v2.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=repr),
        encoding="utf-8",
    )

    md: list[str] = []
    md += [
        "# V1 Stage Diagnosis V2",
        "",
        f"Cases: **{len(results)}**",
        "",
        "## Core questions",
        "",
        "| Measurement | Result |",
        "|---|---:|",
    ]

    summary = payload["summary"]
    total = len(results) or 1

    md += [
        f"| Dense found relevant | {summary['dense_found_relevant']}/{total} |",
        f"| BM25 found relevant | {summary['bm25_found_relevant']}/{total} |",
        f"| RRF retained relevant | {summary['rrf_retained_relevant']}/{total} |",
        f"| Relevant candidate verified | {summary['verification_verified']}/{total} |",
        f"| Relevant candidate uncertain | {summary['verification_uncertain']}/{total} |",
        f"| Relevant candidate rejected | {summary['verification_rejected']}/{total} |",
        f"| Relevant candidate missing from verification buckets | {summary['verification_missing']}/{total} |",
        f"| Ranking preserved relevant | {summary['ranking_preserved_relevant']}/{total} |",
        f"| Evidence boundary preserved relevant | {summary['evidence_boundary_preserved']}/{total} |",
        f"| Package ready | {summary['package_ready']}/{total} |",
        f"| Final context non-empty | {summary['final_context_non_empty']}/{total} |",
        "",
        "## Case table",
        "",
        "| Case | Dense | BM25 | RRF | Verification | Rank | Evidence | Scope removed | Package | Context |",
        "|---|---|---|---|---|---:|---|---:|---|---|",
    ]

    for result in results:
        d = result["diagnosis"]
        md.append(
            "| "
            + " | ".join(
                [
                    result["case_id"],
                    "✅" if d.get("dense_found_relevant") else "❌",
                    "✅" if d.get("bm25_found_relevant") else "❌",
                    "✅" if d.get("rrf_retained_relevant") else "❌",
                    clean(d.get("verification_status_of_relevant")) or "—",
                    str(d.get("relevant_rank_after_ranking") or "—"),
                    "✅" if d.get("evidence_boundary_preserved_relevant") else "❌",
                    str(d.get("scope_filter_removed_candidate_count", 0)),
                    "✅" if d.get("package_ready_for_generation") else "❌",
                    "✅" if d.get("final_context_non_empty") else "❌",
                ]
            )
            + " |"
        )

    md += [
        "",
        "## First failures",
        "",
    ]

    for key, count in sorted(
        summary["first_failure_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        md.append(f"- `{key}`: {count}")

    md += [
        "",
        "## Interpretation rules",
        "",
        "- Dense/BM25 answer whether each raw lane surfaced benchmark-relevant evidence.",
        "- RRF answer whether that evidence remained in `fused_candidates`.",
        "- Verification reports the actual bucket containing the relevant candidate.",
        "- Ranking reports whether the relevant candidate remains in `ranked_candidates`.",
        "- Evidence grouping and expansion are measured separately from scope filtering.",
        "- Claim auditing is measured as an audit stage; the current graph does not expose a separate claim-filter node.",
        "- Final context is saved separately per case under `final_context/`.",
        "",
        f"JSON report: `{json_path}`",
    ]

    md_path = REPORT_ROOT / "diagnosis_v2.md"
    md_path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    before_backend_modules = {
        name
        for name in __import__("sys").modules
        if name == "backend" or name.startswith("backend.")
    }

    nodes_module = resolve_nodes_module()

    after_backend_modules = {
        name
        for name in __import__("sys").modules
        if name == "backend" or name.startswith("backend.")
    }
    imported_backend = sorted(after_backend_modules - before_backend_modules)
    if imported_backend:
        raise RuntimeError(
            "Standalone diagnosis imported backend modules: "
            + ", ".join(imported_backend)
        )

    core = resolve_core_instance(nodes_module)

    benchmark_module = importlib.import_module(
        "tests_reusable.test_diagnosis.cases.benchmark"
    )
    get_cases = getattr(benchmark_module, "get_cases")

    cases = list(get_cases())
    if args.case:
        requested = set(args.case)
        cases = [
            case
            for case in cases
            if case["id"] in requested
        ]

    cases = cases[: max(1, args.limit)]

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_CONTEXT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("AI PLATFORM V1 STAGE DIAGNOSIS V2")
    print("=" * 90)
    print(f"Cases: {len(cases)} | Answer LLM: OFF")
    print(f"Nodes module: {nodes_module.__name__}")
    print("Backend imports in diagnostic path: 0")
    print()

    results = []

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] "
            f"{case['id']} :: {case['question']}"
        )

        try:
            result = run_case(
                case,
                nodes_module,
                core,
            )
            results.append(result)

            diagnosis = result["diagnosis"]
            print(
                "    "
                f"dense={'Y' if diagnosis.get('dense_found_relevant') else 'N'} "
                f"bm25={'Y' if diagnosis.get('bm25_found_relevant') else 'N'} "
                f"rrf={'Y' if diagnosis.get('rrf_retained_relevant') else 'N'} "
                f"verify={diagnosis.get('verification_status_of_relevant') or '-'} "
                f"rank={diagnosis.get('relevant_rank_after_ranking') or '-'} "
                f"evidence={'Y' if diagnosis.get('evidence_boundary_preserved_relevant') else 'N'} "
                f"package={'Y' if diagnosis.get('package_ready_for_generation') else 'N'}"
            )
        except Exception as exc:
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            results.append(
                {
                    "case_id": case["id"],
                    "question": case["question"],
                    "stages": {},
                    "diagnosis": {
                        "first_failure_stage": "runner_error",
                        "runner_error": f"{type(exc).__name__}: {exc}",
                    },
                }
            )

    meta = {
        "nodes_module": nodes_module.__name__,
        "case_ids": [case["id"] for case in cases],
        "node_sequence": list(NODE_SEQUENCE),
    }

    write_reports(results, meta)

    print()
    print("=" * 90)
    print("DIAGNOSIS V2 COMPLETE")
    print("=" * 90)
    print(json.dumps(build_summary(results), indent=2))
    print(f"\nReport: {REPORT_ROOT / 'diagnosis_v2.md'}")
    print(f"JSON:   {REPORT_ROOT / 'diagnosis_v2.json'}")
    print(f"Final contexts: {FINAL_CONTEXT_ROOT}")


if __name__ == "__main__":
    main()
