"""
IIT Jodhpur V1 — Graph State

Purpose
-------
Define the typed state contracts shared across the LangGraph workflow.

State is intentionally permissive (total=False) because different
pipeline branches populate different fields.

Invariants
----------
- GraphState contains the complete production workflow state.
- IntentState contains one independently evaluated user request.
- Multi-intent fields remain optional for single-intent requests.
"""

from typing import (
    Any,
    Dict,
    List,
    TypedDict,
)

from langchain_core.documents import Document


# =========================================================
# Intent State
# =========================================================

class IntentState(TypedDict, total=False):
    """
    State for one independently evaluated user intent.

    An intent owns its own:
    - question
    - semantic metadata
    - evidence status
    - coverage status
    - answer
    - evidence documents
    """

    # -----------------------------------------------------
    # Intent definition
    # -----------------------------------------------------

    question: str
    topics: List[str]
    entities: List[str]

    # -----------------------------------------------------
    # Evidence
    # -----------------------------------------------------

    evidence_status: str
    evidence_score: float
    evidence_coverage_status: str
    evidence_question_type: str

    relevant_evidence_documents: int
    evidence_strong_documents: int
    evidence_partial_documents: int
    evidence_combined_characters: int

    evidence_groups: list
    compressed_docs: List[Document]

    # -----------------------------------------------------
    # Contradiction handling
    # -----------------------------------------------------

    contradiction_status: str
    contradiction_count: int
    contradictions: List[Dict[str, Any]]

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    answer: str

    # -----------------------------------------------------
    # Answer-level auditing
    # -----------------------------------------------------

    answer_guard_status: str
    answer_guard_reason: str

    supported_claim_preservation_status: str
    supported_claim_preservation_missing_claims: List[str]


# =========================================================
# Graph State
# =========================================================

class GraphState(TypedDict, total=False):

    # =====================================================
    # User Input
    # =====================================================

    question: str
    chat_history: List[Dict[str, Any]]

    # =====================================================
    # Query Processing
    # =====================================================

    resolved_question: str
    generated_queries: List[str]

    # =====================================================
    # Phase 5 — Student Situation
    # =====================================================

    student_situation: Any
    decision_context: Any
    retrieval_plan: Any
    retrieval_control: Any

    conversation_mode: str
    active_topic: str
    active_entity: str

    # =====================================================
    # Multi-Intent
    # =====================================================

    is_multi_intent: bool
    intent_count: int

    intent_units: List[Dict[str, Any]]
    intent_questions: List[str]

    intent_results: List[IntentState]

    answer_question: str
    multi_intent_answer_package: Dict[str, Any]

    # =====================================================
    # Retrieval Pipeline
    # =====================================================

    retrieval_results: List[List[Document]]
    retrieval_weights: List[float]

    fused_docs: List[Document]
    initial_reranked_docs: List[Document]
    expanded_docs: List[Document]
    reranked_docs: List[Document]
    compressed_docs: List[Document]

    # =====================================================
    # Evidence
    # =====================================================

    relevant_evidence_documents: int

    evidence_status: str
    evidence_score: float

    evidence_coverage_status: str
    evidence_question_type: str

    evidence_strong_documents: int
    evidence_partial_documents: int
    evidence_combined_characters: int

    evidence_groups: list

    # =====================================================
    # Contradiction Handling
    # =====================================================

    contradiction_status: str
    contradiction_count: int
    contradictions: List[Dict[str, Any]]

    # =====================================================
    # Final Output
    # =====================================================

    answer: str
    context: str

    # =====================================================
    # Evaluation
    # =====================================================

    hallucination_score: float
    answer_score: float

    # =====================================================
    # Answer Guard
    # =====================================================

    answer_guard_status: str
    answer_guard_reason: str

    # =====================================================
    # Supported-Claim Preservation
    # =====================================================

    supported_claim_preservation_status: str
    supported_claim_preservation_missing_claims: List[str]
