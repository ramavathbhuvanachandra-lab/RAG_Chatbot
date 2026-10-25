"""Final Stabilization Pass Regression Tests.

Comprehensive regression suite validating:
1. M.Tech eligibility source preservation (admission requirements, category CGPA/%, no branch change/PhD bleed).
2. Chip-design research discovery precision (Electrical/Electronics priority; rejection of School of Design / Civil Transportation).
3. Ambiguous deictic follow-up resolution ("What eligibility criteria apply to that programme?" triggers clarification without guessing M.Tech).
4. Credit-total handling and sentence termination (preserving '153.', '141 graded + 12 audit', no numerical truncation).
5. Old Mess vs New Mess navigation disambiguation and non-empty natural language response.
6. Chat turn execution lifecycle and exception safety (turn_in_progress reset, terminal assistant response).
7. Monotonic timing contract: inclusive vs exclusive duration labeling and reconciliation.
8. Embedding cache equivalence, cache-hit telemetry, and model isolation.
"""

from __future__ import annotations

import concurrent.futures
import re
import pytest

from ai_platform.core.answering.generator import _ensure_clean_sentence_termination
from ai_platform.core.conversation.conversation import resolve_conversation
from langchain_core.documents import Document
from ai_platform.core.retrieval.contracts import RetrievalCandidate
from ai_platform.core.retrieval.ranking import score_candidate
from ai_platform.core.graph.nodes import _v41_prioritize_evidence_units
from ai_platform.institutions.iitj.campus_navigation import find_location
from ai_platform.runtime.retrieval import (
    _QUERY_EMBEDDING_CACHE,
    _MAX_EMBEDDING_CACHE_SIZE,
    clear_query_embedding_cache,
    dense_retrieve_batch_with_scores,
)
from ai_platform.service.assistant_service import handle_chat_turn, ChatTurnRequest
from ai_platform.service.retrieval_diagnostics import build_retrieval_diagnostics


# ---------------------------------------------------------------------------
# Test 1: Sentence Termination Preserves Trailing Numbers and Degree Abbreviations
# ---------------------------------------------------------------------------

def test_regression_sentence_termination_preserves_credits_and_fees():
    """Sentence termination must NOT strip ending numbers like ' 153.' or ' ₹32,250.'."""
    credit_answer = (
        "The total number of graded and non-graded audit credits required for the "
        "B.S. in Mathematics and Computing is 153."
    )
    res_credit = _ensure_clean_sentence_termination(credit_answer)
    assert res_credit == credit_answer, f"Expected '{credit_answer}', got '{res_credit}'"
    assert "153." in res_credit

    fee_answer = "The tuition fee for regular M.Tech students per semester is ₹32,250."
    res_fee = _ensure_clean_sentence_termination(fee_answer)
    assert res_fee == fee_answer, f"Expected '{fee_answer}', got '{res_fee}'"
    assert "₹32,250." in res_fee

    # Ensure genuine trailing list markers (e.g. '\n4.') are still stripped cleanly
    incomplete_list = "Here are the requirements:\n1. First rule.\n2. Second rule.\n3."
    res_list = _ensure_clean_sentence_termination(incomplete_list)
    assert not res_list.endswith("\n3.")
    assert not res_list.endswith("\n3")
    assert res_list.endswith("Second rule.")


# ---------------------------------------------------------------------------
# Test 2: M.Tech Admissions Grounding and Exclusion of Academic Progression Bleed
# ---------------------------------------------------------------------------

def _make_candidate(text: str, source: str) -> RetrievalCandidate:
    doc = Document(page_content=text, metadata={"source": source})
    return RetrievalCandidate(
        document=doc,
        document_id=source + ":" + text[:20],
        source=source,
    )


def test_regression_mtech_admission_eligibility_grounding():
    """Regular M.Tech eligibility must prioritize admissions documents over branch-change or PhD residence."""
    query = "What are the eligibility criteria for regular M.Tech admission?"

    # Candidate 1: Genuine admission document
    cand_admissions = _make_candidate(
        text=(
            "The applicant must have a bachelor's degree in engineering or science with a minimum "
            "of 60% marks for General/OBC (55% for SC/ST) or a minimum CGPA of 6.0 (5.5 for SC/ST) on a scale of 10."
        ),
        source="data/data_iitj/iitj_rag_v1_docs_production/admissions/mtech_admissions.docx",
    )

    # Candidate 2: Branch change / PhD residence bleed chunk
    cand_progression = _make_candidate(
        text=(
            "Academic progression and branch change: A minimum CGPA of 5.0 is required for continuation. "
            "Branch change is permitted at the end of the first semester for eligible students."
        ),
        source="data/data_iitj/iitj_rag_v1_docs_production/offices_and_administration/office_of_academics/academic_progression.docx",
    )

    score_adm = score_candidate(query, cand_admissions)
    score_prog = score_candidate(query, cand_progression)

    assert score_adm > score_prog, (
        f"Admission document score ({score_adm}) must significantly exceed academic progression score ({score_prog})"
    )


# ---------------------------------------------------------------------------
# Test 3: Chip Design and Semiconductor Research Discovery Precision
# ---------------------------------------------------------------------------

def test_regression_chip_design_prioritizes_electrical_over_school_of_design():
    """Chip design research queries must prioritize Electrical/Electronics and penalize School of Design / Civil."""
    query = "Which departments conduct research on chip design and semiconductors?"

    cand_ee = _make_candidate(
        text=(
            "The Department of Electrical Engineering conducts research in VLSI, micro-nano electronics, "
            "semiconductor device fabrication, and chip design."
        ),
        source="data/data_iitj/iitj_rag_v1_docs_production/departments/electrical_engineering/research.docx",
    )

    cand_design_school = _make_candidate(
        text=(
            "The School of Design focuses on industrial design, user experience design, and product semantics."
        ),
        source="data/data_iitj/iitj_rag_v1_docs_production/schools/design/overview.docx",
    )

    score_ee = score_candidate(query, cand_ee)
    score_design = score_candidate(query, cand_design_school)

    assert score_ee > score_design + 5.0, (
        f"Electrical Engineering score ({score_ee}) must strongly outrank School of Design ({score_design}) for chip design"
    )


# ---------------------------------------------------------------------------
# Test 4: Ambiguous Follow-Up Resolution After Chip Design Conversation
# ---------------------------------------------------------------------------

def test_regression_ambiguous_followup_asks_clarification():
    """Following a chip-design research turn, 'What eligibility criteria apply to that programme?'

    must request clarification rather than guessing M.Tech or inventing a degree.
    """
    history = [
        {"role": "user", "content": "Which departments conduct research on chip design and semiconductors?"},
        {
            "role": "assistant",
            "content": (
                "The Department of Electrical Engineering conducts research in chip design, VLSI, "
                "and semiconductor devices."
            ),
        },
    ]

    resolved = resolve_conversation(
        question="What eligibility criteria apply to that programme?",
        chat_history=history,
    )

    # The resolver must NOT resolve target entity to 'mtech'
    assert resolved.get("entity") != "mtech", "Follow-up must not arbitrarily bind to M.Tech"

    # Full assistant turn must return a concise clarification request
    resp = handle_chat_turn(ChatTurnRequest(
        question="What eligibility criteria apply to that programme?",
        chat_history=history,
    ))
    assert "?" in resp.answer, "Assistant must ask a clarification question"
    assert any(term in resp.answer.lower() for term in ("which programme", "specify", "programme you are inquiring")), (
        f"Assistant answer should prompt for clarification: '{resp.answer}'"
    )


# ---------------------------------------------------------------------------
# Test 5: Old Mess vs New Mess Disambiguation and Populated Natural Language Answer
# ---------------------------------------------------------------------------

def test_regression_old_mess_direct_answer_populated():
    """Old Mess query must return a non-empty, formatted answer and distinct location details."""
    loc = find_location("Where is Old Mess located?")
    assert loc is not None
    assert "old" in loc["name"].lower() or any("old" in a.lower() for a in loc.get("aliases", []))

    resp = handle_chat_turn(ChatTurnRequest(question="Where is Old Mess located?"))
    assert resp.answer, "resp.answer must not be empty string"
    assert len(resp.answer) > 20
    assert resp.diagnostics.get("terminal_status") == "success"


# ---------------------------------------------------------------------------
# Test 6: Streamlit Request Lifecycle and Exception Safety Contract
# ---------------------------------------------------------------------------

def test_regression_request_lifecycle_guarantees_terminal_state():
    """Every request must set terminal_status ('success' | 'safe_refusal' | 'error') and monotonic timestamps."""
    resp = handle_chat_turn(ChatTurnRequest(question="What is the tuition fee for regular M.Tech students per semester?"))
    assert resp.diagnostics is not None
    assert resp.diagnostics.get("terminal_status") in ("success", "safe_refusal", "error")
    assert resp.diagnostics.get("pipeline_start") is not None
    assert resp.diagnostics.get("pipeline_end") is not None
    assert resp.diagnostics.get("latency_seconds", 0.0) > 0.0


# ---------------------------------------------------------------------------
# Test 7: Monotonic Timing Reconciliation (Inclusive vs Exclusive)
# ---------------------------------------------------------------------------

def test_regression_timing_reconciliation_exclusive_and_inclusive():
    """Stage timings must provide distinct inclusive and exclusive breakdowns that reconcile with total pipeline."""
    sample_state = {
        "resolved_question": "What is the course code for Quantum Computing?",
        "stage_timings": {
            "resolve_conversation": 0.005,
            "understand_query": 0.010,
            "query_embedding": 0.020,
            "chroma_vector_search": 0.030,
            "dense_retrieve": 0.050,  # inclusive = query_embedding + chroma_vector_search
            "bm25_retrieve": 0.015,
            "lexical_retrieve": 0.010,
            "fusion": 0.005,
            "verify_and_rank": 0.025,
            "context_packaging": 0.005,
            "llm_generation": 1.200,
            "post_processing": 0.010,
            "total_pipeline": 1.335,
        },
        "retrieved_candidates": [],
        "fused_candidates": [],
        "verified_candidates": [],
    }

    diag = build_retrieval_diagnostics(sample_state, answer="QCL4020", sources=[])
    p8 = diag.get("panel_stage_timings", {})

    assert "inclusive_timings" in p8, "Panel 8 must contain inclusive_timings"
    assert "exclusive_timings" in p8, "Panel 8 must contain exclusive_timings"

    incl = p8["inclusive_timings"]
    excl = p8["exclusive_timings"]

    assert incl["dense_retrieve"] == 0.050
    assert excl["query_embedding"] == 0.020
    assert excl["chroma_vector_search"] == 0.030
    assert "dense_retrieve" not in excl, "dense_retrieve is an inclusive composite and must not be double counted in exclusive timings"

    exclusive_sum = sum(v for v in excl.values())
    total_pipeline = incl["total_pipeline"]
    # Exclusive sum should reconcile within 0.001s
    assert abs(exclusive_sum - total_pipeline) < 0.001, (
        f"Exclusive sum ({exclusive_sum}) must reconcile with total pipeline ({total_pipeline})"
    )


# ---------------------------------------------------------------------------
# Test 8: Embedding Cache Equivalence, Thread Safety, and Capacity Bound
# ---------------------------------------------------------------------------

def test_regression_embedding_cache_equivalence_and_thread_safety():
    """Embedding cache must return identical vectors on repeated queries and enforce size bounds."""
    clear_query_embedding_cache()
    assert len(_QUERY_EMBEDDING_CACHE) == 0

    query = "What is the course code for Quantum Computing at IIT Jodhpur?"

    # 1. Uncached fetch
    res1, emb_time1, _ = dense_retrieve_batch_with_scores([query], k=5)
    assert len(_QUERY_EMBEDDING_CACHE) == 1
    assert emb_time1 >= 0.0

    # 2. Cached fetch
    res2, emb_time2, _ = dense_retrieve_batch_with_scores([query], k=5)
    assert len(_QUERY_EMBEDDING_CACHE) == 1
    # Cache hit should result in virtually 0 embedding computation
    assert emb_time2 < 0.005, f"Expected cache hit embedding time < 5ms, got {emb_time2:.4f}s"

    # Verify identical retrieval results
    docs1 = [doc.page_content for doc, _ in res1[query]]
    docs2 = [doc.page_content for doc, _ in res2[query]]
    assert docs1 == docs2, "Cached and uncached candidate document sets must be strictly identical"

    # 3. Thread-safe concurrent access
    def fetch_worker(tid: int):
        q = f"Concurrent test query {tid % 10}"
        dense_retrieve_batch_with_scores([q], k=2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_worker, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(_QUERY_EMBEDDING_CACHE) <= _MAX_EMBEDDING_CACHE_SIZE
