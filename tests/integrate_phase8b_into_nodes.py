"""
IITJ V1 — Phase 8B Production Integration

Purpose
-------
Integrate the already-tested broad institutional candidate assembler into
backend/nodes.py without manual editing.

The integration point is:

    hybrid_retrieve
        ↓
    fuse_retrieved_documents
        ↓
    8B broad candidate assembly   <-- added here
        ↓
    initial_rerank_documents

Design invariants
-----------------
- Focused questions bypass 8B.
- Broad institutional questions use 8B.
- Multi-intent processing is unchanged.
- Dense/BM25/RRF remain the retrieval foundation.
- No institution-specific program/department list is introduced.
"""

from pathlib import Path
import shutil
import re


ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "backend" / "nodes.py"
BACKUP = ROOT / "backend" / "nodes.py.before_phase8b_integration"


def fail(message: str) -> None:
    raise SystemExit(
        f"\nPHASE 8B INTEGRATION FAILED:\n{message}\n"
    )


def main() -> None:

    if not NODES.exists():
        fail(
            f"Missing file: {NODES}"
        )

    source = NODES.read_text(
        encoding="utf-8"
    )

    if BACKUP.exists():
        print(
            f"Backup already exists: {BACKUP}"
        )
    else:
        shutil.copy2(
            NODES,
            BACKUP,
        )
        print(
            f"Created backup: {BACKUP}"
        )

    # ---------------------------------------------------------
    # 1. Add the 8B import exactly once.
    # ---------------------------------------------------------

    import_block = """from backend.broad_institutional_retrieval import (
    assemble_broad_candidates,
    is_broad_institutional_question,
)
"""

    if "from backend.broad_institutional_retrieval import" not in source:

        anchor = """from backend.retriever import (
"""

        if anchor not in source:
            fail(
                "Could not find backend.retriever import block."
            )

        source = source.replace(
            anchor,
            import_block + "\n" + anchor,
            1,
        )

        print(
            "Added Phase 8B import."
        )

    else:
        print(
            "Phase 8B import already present."
        )

    # ---------------------------------------------------------
    # 2. Replace fuse_retrieved_documents with final 8B-aware
    #    implementation.
    # ---------------------------------------------------------

    pattern = re.compile(
        r"def fuse_retrieved_documents\(\n"
        r"    state: GraphState,\n"
        r"\) -> GraphState:\n"
        r".*?"
        r"\n\n# =========================================================\n"
        r"# Initial Rerank",
        re.DOTALL,
    )

    replacement = """def fuse_retrieved_documents(
    state: GraphState,
) -> GraphState:
    \"""
    Fuse Dense/BM25 candidates, deduplicate them, then apply the Phase 8B
    broad-institutional candidate assembly when the question is a broad
    institutional landscape request.

    Focused questions retain the established retrieval path unchanged.

    Multi-intent requests do not enter this node from the independent
    intent-processing path; their existing pipeline remains unchanged.
    \"""

    question = state.get(
        "resolved_question",
        state["question"],
    )

    fused_docs = reciprocal_rank_fusion(
        state["retrieval_results"]
    )

    fused_docs = deduplicate_documents(
        fused_docs
    )

    # -----------------------------------------------------
    # Phase 8B — Broad Institutional Retrieval
    # -----------------------------------------------------
    #
    # Only broad institutional questions use this selector.
    # It cannot invent evidence: it only reorders/selects candidates that
    # the Dense/BM25/RRF stage already retrieved.
    #
    if is_broad_institutional_question(
        question
    ):
        fused_docs = assemble_broad_candidates(
            query=question,
            documents=fused_docs,
        )

    return {
        "fused_docs": fused_docs,
    }


# =========================================================
# Initial Rerank"""

    matches = pattern.findall(
        source
    )

    if len(matches) != 1:

        fail(
            "Expected exactly one fuse_retrieved_documents block; "
            f"found {len(matches)}."
        )

    updated = pattern.sub(
        replacement,
        source,
        count=1,
    )

    if updated == source:
        fail(
            "nodes.py was not changed."
        )

    NODES.write_text(
        updated,
        encoding="utf-8",
    )

    print(
        "\nPhase 8B integration completed."
    )

    print(
        f"Updated: {NODES}"
    )

    print(
        f"Backup:  {BACKUP}"
    )

    print(
        "\nRun the Phase 8B integration tests now."
    )


if __name__ == "__main__":
    main()
