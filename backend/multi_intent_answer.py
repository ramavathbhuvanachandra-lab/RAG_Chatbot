"""
Phase 4 — Multi-Intent Answer Composition

Purpose
-------
Prepare intent-isolated evidence and strict answer instructions for one
final answer LLM call.

Important invariants
--------------------
- One final answer LLM call.
- Each intent is evaluated independently.
- Evidence is never borrowed across intents.
- Unsupported intents contribute no factual evidence.
- Partially supported intents are explicitly marked as partial.
- Supported intents retain their evidence.
- Classification is performed exactly once per package build.
- Already-classified intent records remain fully usable.
- Deterministic and institution-agnostic.
"""

from __future__ import annotations

from typing import Any, Iterable


# =========================================================
# Constants
# =========================================================

UNKNOWN_RESPONSE = (
    "I'm sorry, I don't know based on the available information."
)

SUPPORTED = "supported"
PARTIALLY_SUPPORTED = "partially_supported"
INSUFFICIENT = "insufficient"

PARTIAL_STATUS_ALIASES = {
    "partial",
    "partially_supported",
}


# =========================================================
# Status normalization
# =========================================================

def normalize_intent_status(
    evidence_status: Any,
    coverage_status: Any,
) -> str:
    """
    Resolve the effective evidence status for one intent.

    Rules
    -----
    supported + supported
        -> supported

    supported + partial
        -> partially_supported

    partial
        -> partially_supported

    everything else
        -> insufficient
    """

    evidence = str(
        evidence_status
        if evidence_status is not None
        else INSUFFICIENT
    ).strip().lower()

    coverage = str(
        coverage_status
        if coverage_status is not None
        else INSUFFICIENT
    ).strip().lower()

    # -----------------------------------------------------
    # Fully supported
    # -----------------------------------------------------

    if (
        evidence == SUPPORTED
        and
        coverage == SUPPORTED
    ):
        return SUPPORTED

    # -----------------------------------------------------
    # Partial support
    # -----------------------------------------------------

    if (
        evidence in PARTIAL_STATUS_ALIASES
        or
        coverage in PARTIAL_STATUS_ALIASES
    ):
        return PARTIALLY_SUPPORTED

    if (
        evidence == SUPPORTED
        and
        coverage != INSUFFICIENT
    ):
        return PARTIALLY_SUPPORTED

    # -----------------------------------------------------
    # Insufficient
    # -----------------------------------------------------

    return INSUFFICIENT


# =========================================================
# Raw-field access
# =========================================================

def _get_evidence_status(
    intent: dict[str, Any],
) -> str:
    """
    Read the evidence status from an intent record.
    """

    value = intent.get(
        "evidence_status"
    )

    if value is None:
        value = INSUFFICIENT

    return str(
        value
    ).strip().lower()


def _get_coverage_status(
    intent: dict[str, Any],
) -> str:
    """
    Read coverage status from either raw or classified intent state.
    """

    value = intent.get(
        "evidence_coverage_status"
    )

    if value is None:
        value = intent.get(
            "coverage_status"
        )

    if value is None:
        value = INSUFFICIENT

    return str(
        value
    ).strip().lower()


# =========================================================
# Evidence extraction
# =========================================================

def _get_documents(
    intent: dict[str, Any],
) -> list:
    """
    Read evidence documents from either raw or classified intent state.

    Raw intent state:
        compressed_docs

    Classified intent state:
        documents
    """

    documents = intent.get(
        "documents"
    )

    if documents is None:
        documents = intent.get(
            "compressed_docs",
            [],
        )

    if documents is None:
        return []

    if isinstance(
        documents,
        list,
    ):
        return documents

    if isinstance(
        documents,
        tuple,
    ):
        return list(
            documents
        )

    try:
        return list(
            documents
        )
    except TypeError:
        return []


# =========================================================
# Intent classification
# =========================================================

def classify_intents(
    intent_results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize raw intent results into one canonical representation.

    Canonical representation:

        {
            index,
            question,
            status,
            evidence_status,
            coverage_status,
            documents
        }

    Important:
        This function is intentionally safe to call on already
        classified records.
    """

    classified = []

    for index, intent in enumerate(
        intent_results or [],
        start=1,
    ):

        evidence_status = _get_evidence_status(
            intent
        )

        coverage_status = _get_coverage_status(
            intent
        )

        existing_status = intent.get(
            "status"
        )

        if existing_status is not None:

            normalized_existing = str(
                existing_status
            ).strip().lower()

            if normalized_existing in {
                SUPPORTED,
                PARTIALLY_SUPPORTED,
                INSUFFICIENT,
            }:

                status = normalized_existing

            else:

                status = normalize_intent_status(
                    evidence_status=evidence_status,
                    coverage_status=coverage_status,
                )

        else:

            status = normalize_intent_status(
                evidence_status=evidence_status,
                coverage_status=coverage_status,
            )

        documents = _get_documents(
            intent
        )

        # -----------------------------------------------------
        # Unsupported intent must never expose factual evidence.
        # -----------------------------------------------------

        if status == INSUFFICIENT:
            documents = []

        classified.append(
            {
                "index": index,
                "question": str(
                    intent.get(
                        "question",
                        "",
                    )
                ).strip(),
                "status": status,
                "evidence_status": evidence_status,
                "coverage_status": coverage_status,
                "documents": documents,
            }
        )

    return classified


# =========================================================
# Context building
# =========================================================

def build_multi_intent_context(
    classified_intents: Iterable[dict[str, Any]],
) -> str:
    """
    Build isolated evidence sections from canonical classified intents.

    This function expects canonical records from classify_intents().
    It also tolerates raw records defensively.
    """

    # Normalize only if necessary.
    records = list(
        classified_intents or []
    )

    if any(
        "status" not in record
        or "documents" not in record
        for record in records
    ):
        records = classify_intents(
            records
        )

    sections = []

    for intent in records:

        index = intent["index"]
        question = intent["question"]
        status = intent["status"]
        documents = intent.get(
            "documents",
            [],
        )

        lines = [
            f"Intent {index}",
            f"Question: {question}",
            f"Evidence Status: {status}",
        ]

        # -----------------------------------------------------
        # Unsupported intent
        # -----------------------------------------------------

        if status == INSUFFICIENT:

            lines.extend(
                [
                    "Evidence: NONE",
                    (
                        "Do not provide factual claims "
                        "for this intent."
                    ),
                    (
                        "State that the available "
                        "information is insufficient."
                    ),
                ]
            )

        else:

            # -------------------------------------------------
            # Partial evidence
            # -------------------------------------------------

            if status == PARTIALLY_SUPPORTED:

                lines.append(
                    (
                        "Evidence Scope: PARTIAL — "
                        "do not imply completeness."
                    )
                )

            # -------------------------------------------------
            # Evidence documents
            # -------------------------------------------------

            if documents:

                for evidence_index, document in enumerate(
                    documents,
                    start=1,
                ):

                    lines.append(
                        f"Evidence {evidence_index}"
                    )

                    lines.append(
                        str(
                            getattr(
                                document,
                                "page_content",
                                "",
                            )
                            or ""
                        )
                    )

            else:

                lines.extend(
                    [
                        "Evidence: NONE",
                        (
                            "Do not provide factual claims "
                            "for this intent."
                        ),
                    ]
                )

        sections.append(
            "\n".join(
                lines
            )
        )

    return "\n\n".join(
        sections
    )


# =========================================================
# Final answer instructions
# =========================================================

def build_multi_intent_instruction(
    classified_intents: Iterable[dict[str, Any]],
) -> str:
    """
    Build deterministic instructions controlling final answer
    composition.

    The function consumes canonical classified records.
    """

    records = list(
        classified_intents or []
    )

    if any(
        "status" not in record
        or "index" not in record
        for record in records
    ):
        records = classify_intents(
            records
        )

    lines = [
        (
            "Answer each user request independently "
            "and in the same order."
        ),
        (
            "Never merge facts, requirements, fees, dates, "
            "eligibility rules, or conditions between intents."
        ),
        (
            "Use only evidence attached to the corresponding intent."
        ),
        (
            "Do not invent information that is absent from evidence."
        ),
        (
            "Do not turn partial evidence into a complete or exhaustive "
            "answer."
        ),
        (
            "Do not add unrelated institutional information."
        ),
        (
            "Do not mention internal evidence labels, document numbers, "
            "retrieval scores, chunk IDs, or system details."
        ),
    ]

    for intent in records:

        index = intent["index"]
        status = intent["status"]

        if status == SUPPORTED:

            lines.append(
                (
                    f"Intent {index}: answer directly using "
                    "its supplied evidence."
                )
            )

        elif status == PARTIALLY_SUPPORTED:

            lines.append(
                (
                    f"Intent {index}: answer only the supported "
                    "portion and clearly state that the available "
                    "evidence is partial."
                )
            )

        else:

            lines.append(
                (
                    f"Intent {index}: do not provide unsupported "
                    "factual information; state that the available "
                    "information is insufficient."
                )
            )

    return "\n".join(
        lines
    )


# =========================================================
# Package builder
# =========================================================

def build_multi_intent_answer_package(
    intent_results: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the complete deterministic package used by final answer
    generation.

    Classification happens exactly once.
    """

    classified = classify_intents(
        intent_results
    )

    total = len(
        classified
    )

    supported = sum(
        1
        for intent in classified
        if intent["status"] == SUPPORTED
    )

    partial = sum(
        1
        for intent in classified
        if intent["status"] == PARTIALLY_SUPPORTED
    )

    insufficient = sum(
        1
        for intent in classified
        if intent["status"] == INSUFFICIENT
    )

    # -----------------------------------------------------
    # Overall package status
    # -----------------------------------------------------

    if total == 0:

        overall_status = INSUFFICIENT

    elif insufficient == total:

        overall_status = INSUFFICIENT

    elif (
        partial > 0
        or
        insufficient > 0
    ):

        overall_status = PARTIALLY_SUPPORTED

    else:

        overall_status = SUPPORTED

    # -----------------------------------------------------
    # Build context/instructions directly from the same canonical
    # records. This prevents evidence/state loss.
    # -----------------------------------------------------

    return {
        "overall_status": overall_status,
        "supported_intents": supported,
        "partial_intents": partial,
        "insufficient_intents": insufficient,
        "intent_count": total,
        "context": build_multi_intent_context(
            classified
        ),
        "instruction": build_multi_intent_instruction(
            classified
        ),
        "intents": classified,
    }
