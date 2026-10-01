"""
IIT Jodhpur V1 — LangGraph Nodes

Production flow:

    Conversation Resolver
        ↓
    Multi-Intent Planner
        ↓
    Single Intent → Dense + BM25
    Multi Intent  → Independent Evidence Pipelines
        ↓
    Dense + BM25
        ↓
    Weighted RRF
        ↓
    Deduplication
        ↓
    Initial Reranking
        ↓
    Local Context Expansion
        ↓
    Evidence Grouping
        ↓
    Group Ranking
        ↓
    Evidence Sufficiency
        ↓
    Evidence Coverage
        ↓
    Final Context
        ↓
    One Answer LLM Call
        ↓
    Answer Guard

Important invariants
--------------------
- No additional LLM call is introduced by retrieval enhancements.
- Initial retrieval budget is question-type aware.
- Broad list questions may use more anchors because their answers
  naturally span multiple documents.
- Local context remains deterministic.
- Evidence groups remain the final retrieval units.
- Multi-intent decomposition is deterministic and conservative.
- Multi-intent processing reuses the existing evidence pipeline without
  introducing another LLM call.
- Existing single-intent retrieval and answer behavior remain unchanged.
"""

from backend.state import GraphState

from backend.broad_institutional_retrieval import (
    assemble_broad_candidates,
    build_broad_retrieval_queries,
    is_broad_institutional_question,
)

from backend.institutional_coverage import (
    assess_institutional_coverage,
)


from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    rerank_documents,
    detect_programs,
    detect_entities,
    FINAL_CONTEXT_DOCUMENTS,
    MAX_RETRIEVAL_QUERIES,
    format_context,
)
from backend.evidence_scope_gate import (
    select_answer_evidence,
)

from backend.local_context import (
    expand_local_context,
)

from backend.evidence_groups import (
    build_evidence_groups,
    attach_local_context,
    rank_evidence_groups,
    flatten_evidence_groups,
)

from backend.final_evidence_scope import (
    filter_final_evidence_scope,
)

from backend.claim_context_filter import (
    filter_claim_context,
)


from backend.evidence import (
    assess_evidence_sufficiency,
)

from backend.evidence_coverage import (
    assess_evidence_coverage,
    detect_question_type,
)

from backend.llm import answer_llm
from backend.prompts import answer_prompt
from backend.answer_guard import guard_answer
from backend.conversation_resolver import resolve_conversation
from backend.multi_intent import decompose_multi_intent
from backend.multi_intent_answer import build_multi_intent_answer_package


# =========================================================
# Retrieval Budget
# =========================================================

# Broad enumeration questions naturally require wider evidence
# aggregation than focused factual questions.
BROAD_LIST_INITIAL_DOCUMENTS = 10


def _initial_rerank_budget(
    question: str,
) -> int:
    """
    Determine the number of initial evidence anchors to keep.

    The policy is generic:

        list         -> broader recall budget
        everything   -> conservative focused budget

    This affects only the pre-expansion anchor set. Final evidence
    remains governed by evidence-group ranking.
    """

    question_type = detect_question_type(
        question
    )

    if question_type == "list":
        return max(
            FINAL_CONTEXT_DOCUMENTS,
            BROAD_LIST_INITIAL_DOCUMENTS,
        )

    return FINAL_CONTEXT_DOCUMENTS


# =========================================================
# Answer Chain
# =========================================================

answer_chain = (
    answer_prompt
    | answer_llm
)


# =========================================================
# Conversation Resolution
# =========================================================

def resolve_conversation_node(
    state: GraphState,
) -> GraphState:

    result = resolve_conversation(
        question=state["question"],
        chat_history=state.get(
            "chat_history",
            [],
        ),
    )

    return {
        "resolved_question": (
            result["resolved_question"]
        ),
        "conversation_mode": (
            result["mode"]
        ),
        "active_topic": (
            result["active_topic"]
        ),
        "active_entity": (
            result["active_entity"]
        ),
    }


# =========================================================
# Multi-Intent Planning
# =========================================================

def plan_multi_intent_node(
    state: GraphState,
) -> GraphState:
    """
    Detect independently answerable requests before retrieval.

    Single-intent questions remain on the existing production path.
    Multi-intent questions receive an explicit intent plan so each request
    can be evaluated independently.
    """

    question = state.get(
        "resolved_question",
        state["question"],
    )

    units = decompose_multi_intent(
        question
    )

    intent_units = [
        {
            "question": unit.question,
            "topics": list(unit.topics),
            "entities": list(unit.entities),
        }
        for unit in units
    ]

    return {
        "is_multi_intent": len(units) > 1,
        "intent_count": len(units),
        "intent_units": intent_units,
        "intent_questions": [
            unit.question
            for unit in units
        ],
    }


# =========================================================
# Multi-Intent Evidence Pipeline
# =========================================================

def _run_intent_evidence_pipeline(
    question: str,
) -> dict:
    """
    Execute the existing deterministic evidence pipeline for one intent.

    This helper intentionally excludes answer generation. All retrieval,
    reranking, evidence, and filtering logic is reused as-is.
    """

    local_state: GraphState = {
        "question": question,
        "resolved_question": question,
        "chat_history": [],
    }

    local_state.update(
        hybrid_retrieve(
            local_state
        )
    )

    local_state.update(
        fuse_retrieved_documents(
            local_state
        )
    )

    local_state.update(
        initial_rerank_documents(
            local_state
        )
    )

    local_state.update(
        expand_retrieved_context(
            local_state
        )
    )

    local_state.update(
        final_rerank_documents(
            local_state
        )
    )

    local_state.update(
        assess_evidence_node(
            local_state
        )
    )

    local_state.update(
        assess_evidence_coverage_node(
            local_state
        )
    )

    local_state.update(
        compress_context(
            local_state
        )
    )

    return local_state


def process_multi_intent_node(
    state: GraphState,
) -> GraphState:
    """
    Evaluate every planned intent independently using the existing
    retrieval/evidence pipeline.

    No answer LLM is called here. A single final answer call is performed
    later after intent-specific evidence has been assembled.
    """

    intent_units = state.get(
        "intent_units",
        [],
    )

    intent_results = []
    combined_docs = []
    combined_groups = []

    for intent_index, intent in enumerate(
        intent_units,
        start=1,
    ):

        question = intent.get(
            "question",
            "",
        )

        local_state = _run_intent_evidence_pipeline(
            question
        )

        intent_result = {
            "question": question,
            "topics": intent.get(
                "topics",
                [],
            ),
            "entities": intent.get(
                "entities",
                [],
            ),
            "evidence_status": local_state.get(
                "evidence_status",
                "insufficient",
            ),
            "evidence_score": local_state.get(
                "evidence_score",
                0.0,
            ),
            "evidence_coverage_status": local_state.get(
                "evidence_coverage_status",
                "insufficient",
            ),
            "evidence_question_type": local_state.get(
                "evidence_question_type",
                "descriptive",
            ),
            "relevant_evidence_documents": local_state.get(
                "relevant_evidence_documents",
                0,
            ),
            "evidence_strong_documents": local_state.get(
                "evidence_strong_documents",
                0,
            ),
            "evidence_partial_documents": local_state.get(
                "evidence_partial_documents",
                0,
            ),
            "evidence_combined_characters": local_state.get(
                "evidence_combined_characters",
                0,
            ),
            "compressed_docs": local_state.get(
                "compressed_docs",
                [],
            ),
        }

        intent_results.append(
            intent_result
        )

        combined_docs.extend(
            intent_result[
                "compressed_docs"
            ]
        )

        for group in local_state.get(
            "evidence_groups",
            [],
        ):
            combined_groups.append(
                {
                    "intent_index": intent_index,
                    "question": question,
                    "group": group,
                }
            )

    supported_count = sum(
        1
        for item in intent_results
        if item["evidence_status"] == "supported"
    )

    insufficient_count = sum(
        1
        for item in intent_results
        if item["evidence_status"] == "insufficient"
    )

    if not intent_results or (
        insufficient_count == len(intent_results)
    ):
        overall_status = "insufficient"
        overall_coverage = "insufficient"

    elif insufficient_count:
        overall_status = "supported"
        overall_coverage = "partially_supported"

    elif supported_count == len(intent_results):
        overall_status = "supported"
        overall_coverage = "supported"

    else:
        overall_status = "supported"
        overall_coverage = "partially_supported"

    answer_questions = []

    for index, item in enumerate(
        intent_results,
        start=1,
    ):
        answer_questions.append(
            f"{index}. {item['question']}"
        )

    answer_question = (
        "Answer each user request separately. "
        "Do not merge the evidence or requirements of different requests. "
        "Use only the evidence belonging to each numbered request. "
        "If one request lacks sufficient information, say so for that "
        "request while still answering any other supported request.\n\n"
        + "\n".join(answer_questions)
    )

    answer_package = build_multi_intent_answer_package(
        intent_results
    )

    return {
        "intent_results": intent_results,
        "evidence_groups": combined_groups,
        "compressed_docs": combined_docs,
        "evidence_status": overall_status,
        "evidence_coverage_status": overall_coverage,
        "evidence_question_type": "multi_intent",
        "evidence_strong_documents": sum(
            item["evidence_strong_documents"]
            for item in intent_results
        ),
        "evidence_partial_documents": sum(
            item["evidence_partial_documents"]
            for item in intent_results
        ),
        "evidence_combined_characters": sum(
            item["evidence_combined_characters"]
            for item in intent_results
        ),
        "relevant_evidence_documents": sum(
            item["relevant_evidence_documents"]
            for item in intent_results
        ),
        "answer_question": answer_question,
        "multi_intent_answer_package": answer_package,
    }


def format_multi_intent_context(
    intent_results,
) -> str:
    """
    Format intent-specific evidence so the answer model can see which
    evidence belongs to which request.
    """

    sections = []

    for index, item in enumerate(
        intent_results,
        start=1,
    ):

        lines = [
            f"Intent {index}",
            f"Question: {item['question']}",
        ]

        docs = item.get(
            "compressed_docs",
            [],
        )

        if docs:
            for document_index, document in enumerate(
                docs,
                start=1,
            ):
                lines.append(
                    f"Evidence {document_index}"
                )
                lines.append(
                    document.page_content
                )
        else:
            lines.append(
                "No sufficient evidence was retrieved for this intent."
            )

        sections.append(
            "\n".join(lines)
        )

    return "\n\n".join(
        sections
    )


# =========================================================
# Hybrid Retrieval
# =========================================================

def hybrid_retrieve(
    state: GraphState,
) -> GraphState:
    """
    Production hybrid retrieval.

    The resolved user question is always query zero and remains authoritative.
    Broad institutional questions receive a small, deterministic recall
    expansion before Dense/BM25 retrieval. Focused questions keep the existing
    bounded variant path.
    """

    question = state.get(
        "resolved_question",
        state["question"],
    )

    is_broad = is_broad_institutional_question(
        question
    )

    queries = [
        question,
    ]

    # ---------------------------------------------------------
    # Planner-produced query variants
    # ---------------------------------------------------------

    planner_limit = (
        1
        if is_broad
        else 2
    )

    for candidate in state.get(
        "generated_queries",
        [],
    ):

        candidate = str(
            candidate or ""
        ).strip()

        if not candidate:
            continue

        candidate_key = candidate.casefold()

        if candidate_key in {
            existing.casefold()
            for existing in queries
        }:
            continue

        queries.append(
            candidate
        )

        if len(queries) - 1 >= planner_limit:
            break

    # ---------------------------------------------------------
    # Phase 8B generic broad recall
    # ---------------------------------------------------------

    if is_broad:

        for candidate in build_broad_retrieval_queries(
            question
        ):

            candidate = str(
                candidate or ""
            ).strip()

            if not candidate:
                continue

            candidate_key = candidate.casefold()

            if candidate_key in {
                existing.casefold()
                for existing in queries
            }:
                continue

            queries.append(
                candidate
            )

            # Keep broad retrieval bounded.
            if len(queries) >= 4:
                break

    # ---------------------------------------------------------
    # Dense + BM25 for every bounded query
    # ---------------------------------------------------------

    retrieval_results = []
    retrieval_weights = []

    # Primary query receives full weight. Recovery variants decay
    # deterministically and cannot override a strong primary result.
    variant_decays = (
        1.00,
        0.72,
        0.50,
        0.36,
    )

    for index, query in enumerate(
        queries
    ):

        dense_documents = dense_retrieve(
            query
        )

        keyword_documents = keyword_retrieve(
            query
        )

        decay = variant_decays[
            min(
                index,
                len(variant_decays) - 1,
            )
        ]

        retrieval_results.extend(
            [
                dense_documents,
                keyword_documents,
            ]
        )

        retrieval_weights.extend(
            [
                0.70 * decay,
                0.30 * decay,
            ]
        )

    return {
        "retrieval_results":
            retrieval_results,
        "retrieval_queries":
            queries,
        "retrieval_weights":
            retrieval_weights,
    }





# =========================================================
# Fuse Documents
# =========================================================

def fuse_retrieved_documents(
    state: GraphState,
) -> GraphState:
    """
    Fuse Dense/BM25 results with weighted RRF, deduplicate them, and apply
    Phase 8B broad candidate selection before initial reranking.
    """

    retrieval_results = state.get(
        "retrieval_results",
        [],
    )

    retrieval_weights = state.get(
        "retrieval_weights",
        [],
    )

    if (
        retrieval_weights
        and len(retrieval_weights)
        == len(retrieval_results)
    ):

        fused_docs = reciprocal_rank_fusion(
            retrieval_results,
            weights=retrieval_weights,
        )

    else:

        fused_docs = reciprocal_rank_fusion(
            retrieval_results
        )

    fused_docs = deduplicate_documents(
        fused_docs
    )

    question = state.get(
        "resolved_question",
        state["question"],
    )

    # Broad candidate assembly is deliberately downstream of RRF so that
    # Dense/BM25 evidence from all bounded variants is considered together.
    if is_broad_institutional_question(
        question
    ):

        fused_docs = assemble_broad_candidates(
            query=question,
            documents=fused_docs,
        )

    return {
        "fused_docs":
            fused_docs,
    }





# =========================================================
# Initial Reranking
# =========================================================

def _has_specific_retrieval_signal(
    query: str,
) -> bool:
    """
    Determine whether a generated query contains a concrete program/entity
    signal strong enough to override a broad-question classification.

    Reuses the existing generic retriever detectors. No institution-specific
    program names, source paths, filenames, or IDs are hardcoded here.
    """
    normalized = str(
        query or ""
    ).strip()

    if not normalized:
        return False

    return bool(
        detect_programs(normalized)
        or detect_entities(normalized)
    )


def _authoritative_retrieval_query(
    state: GraphState,
) -> str:
    """
    Return the query that should drive downstream evidence selection.

    A generated query containing a concrete program/entity signal is allowed
    to override the broad-question classifier. This is necessary for queries
    such as:

        "What are the admission routes for M.Sc.?"

    where the wording is broadly classified as an admission/list request,
    but the actual retrieval target is the specific M.Sc. program.

    Genuinely broad institutional requests continue to use the resolved user
    question as their authoritative evidence query.

    The original user wording is never replaced in state.
    """
    question = state.get(
        "resolved_question",
        state["question"],
    )

    generated_queries = [
        str(query or "").strip()
        for query in state.get(
            "generated_queries",
            [],
        )
        if str(query or "").strip()
    ]

    if not generated_queries:
        return question

    primary_generated_query = generated_queries[0]
    broad_question = is_broad_institutional_question(
        question
    )

    if (
        not broad_question
        or _has_specific_retrieval_signal(
            primary_generated_query
        )
    ):
        return primary_generated_query

    return question


def initial_rerank_documents(
    state: GraphState,
) -> GraphState:
    """
    Select initial evidence anchors.

    The primary resolved question remains the authoritative ranking query.
    Broad recovery variants are supplied only as secondary ranking signals.
    """

    question = state.get(
        "resolved_question",
        state["question"],
    )

    initial_budget = _initial_rerank_budget(
        question
    )

    query_variants = []

    if is_broad_institutional_question(
        question
    ):

        query_variants = [
            query
            for query in state.get(
                "retrieval_queries",
                [],
            )[1:]
            if query
        ]

    initial_docs = rerank_documents(
        query=question,
        documents=state.get(
            "fused_docs",
            [],
        ),
        top_k=initial_budget,
        query_variants=query_variants,
    )

    return {
        "initial_reranked_docs":
            initial_docs,
    }





# =========================================================
# Local Context Expansion
# =========================================================

def expand_retrieved_context(
    state: GraphState,
) -> GraphState:
    """
    Expand only the initial high-relevance anchors.
    """

    expanded_docs = expand_local_context(
        state.get(
            "initial_reranked_docs",
            [],
        )
    )

    return {
        "expanded_docs":
            expanded_docs,
    }


# =========================================================
# Evidence Grouping + Ranking
# =========================================================

def final_rerank_documents(
    state: GraphState,
) -> GraphState:
    """
    Build evidence groups from initial anchors and local context,
    rank the groups as coherent evidence units, then remove documents
    with explicit claim-scope conflicts.

    Invariants
    ----------
    - Evidence-group ordering remains deterministic.
    - Claim-scope filtering happens before evidence sufficiency.
    - Incompatible evidence never reaches the answer-generation stage.
    """

    question = _authoritative_retrieval_query(
        state
    )

    anchors = state.get(
        "initial_reranked_docs",
        [],
    )

    expanded_docs = state.get(
        "expanded_docs",
        [],
    )

    groups = build_evidence_groups(
        anchors
    )

    groups = attach_local_context(
        groups,
        expanded_docs,
    )

    # Rank every evidence group that survived initial retrieval.
    #
    # FINAL_CONTEXT_DOCUMENTS is the answer-context budget, not the
    # verification budget. Truncating here can discard valid evidence
    # before scope filtering, sufficiency, and coverage get to inspect it.
    ranked_groups = rank_evidence_groups(
        query=question,
        groups=groups,
        top_k=len(groups),
    )

    grouped_docs = flatten_evidence_groups(
        ranked_groups
    )

    # -----------------------------------------------------
    # Final claim-scope filter
    # -----------------------------------------------------

    final_docs = filter_final_evidence_scope(
        query=question,
        documents=grouped_docs,
    )

    return {
        "evidence_groups": ranked_groups,
        "reranked_docs": final_docs,
    }


# =========================================================
# Evidence Sufficiency
# =========================================================

def assess_evidence_node(
    state: GraphState,
) -> GraphState:

    question = _authoritative_retrieval_query(
        state
    )

    documents = state.get(
        "reranked_docs",
        [],
    )

    result = assess_evidence_sufficiency(
        query=question,
        documents=documents,
    )

    return {
        "evidence_status": (
            result["status"]
        ),
        "evidence_score": (
            result["score"]
        ),
        "relevant_evidence_documents": (
            result["relevant_documents"]
        ),
    }


# =========================================================
# Evidence Coverage
# =========================================================

def assess_evidence_coverage_node(
    state: GraphState,
) -> GraphState:
    """
    Run baseline evidence coverage plus Phase 8C institutional coverage.

    Broad institutional questions are assessed for breadth/completeness
    independently of the lower-level question-type classifier.
    """

    question = _authoritative_retrieval_query(
        state
    )

    documents = state.get(
        "reranked_docs",
        [],
    )

    result = assess_institutional_coverage(
        query=question,
        documents=documents,
    )

    return {
        "evidence_coverage_status":
            result["status"],
        "evidence_question_type":
            result["question_type"],
        "evidence_strong_documents":
            result["strong_documents"],
        "evidence_partial_documents":
            result["partial_documents"],
        "evidence_combined_characters":
            result["combined_characters"],
    }





# =========================================================
# Final Context
# =========================================================

def compress_context(
    state: GraphState,
) -> GraphState:
    """
    Build the final answer context from selected evidence groups.

    Final evidence processing order:

        1. document-level claim-scope filtering
        2. claim-level context filtering

    The goal is to prevent mixed retrieved chunks from exposing
    unrelated claims to the answer model.
    """

    if (
        state.get(
            "evidence_status"
        )
        == "insufficient"
    ):
        return {
            "compressed_docs": []
        }

    question = _authoritative_retrieval_query(
        state
    )

    evidence_groups = state.get(
        "evidence_groups",
        [],
    )

    grouped_docs = flatten_evidence_groups(
        evidence_groups
    )

    final_scope_docs = filter_final_evidence_scope(
        query=question,
        documents=grouped_docs,
    )

    compressed_docs = filter_claim_context(
        query=question,
        documents=final_scope_docs,
    )

    # -----------------------------------------------------
    # Phase 8E — Answer evidence scope gate
    # -----------------------------------------------------
    #
    # Existing scope filters protect against explicit conflicts. This final
    # gate handles a different problem: for a question about one explicit
    # program/entity, a generic document can still contain enough overlapping
    # words to look relevant while contributing unrelated claims.
    #
    # The gate prefers evidence whose source scope explicitly matches the
    # requested program/entity. It contains no institution-specific filenames,
    # IDs, or folder lists.
    # -----------------------------------------------------
    answer_evidence = select_answer_evidence(
        query=question,
        documents=compressed_docs,
        max_documents=FINAL_CONTEXT_DOCUMENTS,
    )

    return {
        "compressed_docs":
            answer_evidence,
    }


# =========================================================
# Generate Answer
# =========================================================

def generate_answer(
    state: GraphState,
) -> GraphState:
    """
    Generate the final grounded answer with exactly one answer LLM
    call.
    """

    evidence_status = state.get(
        "evidence_status",
        "insufficient",
    )

    coverage_status = state.get(
        "evidence_coverage_status",
        "insufficient",
    )

    if (
        evidence_status == "insufficient"
        or coverage_status == "insufficient"
    ):
        return {
            "answer": (
                "I'm sorry, I don't know "
                "based on the available information."
            ),
            "context": "",
            "answer_guard_status": "safe",
            "answer_guard_reason": (
                "insufficient_evidence"
            ),
        }

    if state.get(
        "is_multi_intent",
        False,
    ):
        answer_package = state.get(
            "multi_intent_answer_package",
            build_multi_intent_answer_package(
                state.get(
                    "intent_results",
                    [],
                )
            ),
        )

        context = answer_package[
            "context"
        ]

        answer_question = (
            answer_package[
                "instruction"
            ]
            + "\n\nRequests to answer:\n"
            + "\n".join(
                f"{index}. {item.get('question', '')}"
                for index, item in enumerate(
                    state.get(
                        "intent_results",
                        [],
                    ),
                    start=1,
                )
            )
        )

    else:
        context = format_context(
            state.get(
                "compressed_docs",
                [],
            )
        )

        answer_question = state.get(
            "resolved_question",
            state["question"],
        )

    question_type = state.get(
        "evidence_question_type",
        "descriptive",
    )

    response = answer_chain.invoke(
        {
            "context": context,
            "question": answer_question,
            "chat_history": state.get(
                "chat_history",
                [],
            ),
            "question_type": question_type,
            "evidence_coverage": coverage_status,
        }
    )

    guard_result = guard_answer(
        response.content
    )

    return {
        "answer": guard_result["answer"],
        "context": context,
        "answer_guard_status": (
            guard_result["status"]
        ),
        "answer_guard_reason": (
            guard_result["reason"]
        ),
    }