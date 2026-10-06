from __future__ import annotations

from types import SimpleNamespace

from ai_platform.core.graph.nodes import CoreNodes
from ai_platform.institutions.iitj.profile import (
    INSTITUTION_DATA_ROOT,
    PROFILE,
    VECTORSTORE_COLLECTION,
    VECTORSTORE_ROOT,
)
from ai_platform.runtime.retrieval import dense_retrieve
from ai_platform.runtime.vectorstore import get_vectorstore


QUESTION = "What are the M.Tech eligibility requirements?"


def _fake_node() -> CoreNodes:
    node = CoreNodes.__new__(CoreNodes)
    node.deps = SimpleNamespace(
        extract_claims=lambda _query: ("dummy claim",),
        extract_evidence_units=lambda candidates: tuple(candidates),
        audit_claims=lambda claims, units: SimpleNamespace(
            supported_claim_ids=("c1",) if units else (),
            partial_claim_ids=(),
            conflicting_claim_ids=(),
            matches=(),
        ),
        build_evidence_package=lambda units, **_kwargs: SimpleNamespace(
            ready_for_generation=bool(units),
            context="verified context" if units else "",
        ),
    )
    return node


def test_iitj_profile_wires_collection_name() -> None:
    assert VECTORSTORE_COLLECTION == "iitj_v1"
    assert PROFILE.vectorstore_collection == "iitj_v1"


def test_active_vectorstore_is_the_existing_iitj_v1_collection() -> None:
    assert INSTITUTION_DATA_ROOT.is_dir(), INSTITUTION_DATA_ROOT
    assert VECTORSTORE_ROOT.is_dir(), VECTORSTORE_ROOT

    store = get_vectorstore()
    collection = getattr(store, "_collection", None)

    assert collection is not None
    assert collection.name == VECTORSTORE_COLLECTION
    assert collection.count() > 0, (
        "Active iitj_v1 collection is empty. "
        "Stop here and restore vector ingestion before the RAG benchmark."
    )


def test_production_dense_retrieval_returns_documents() -> None:
    docs = tuple(dense_retrieve(QUESTION))

    assert docs, (
        "Production dense_retrieve() returned zero documents from "
        "the active IITJ vector collection."
    )
    assert all(
        str(getattr(doc, "page_content", "") or "").strip()
        for doc in docs
    )


def test_claim_audit_never_falls_back_to_ranked_candidates() -> None:
    node = _fake_node()
    captured = {}

    def capture_units(candidates):
        captured["candidates"] = tuple(candidates)
        return tuple(candidates)

    node.deps.extract_evidence_units = capture_units

    node.audit_claims_node(
        {
            "query": QUESTION,
            "evidence_candidates": (),
            "ranked_candidates": ("RANKED-BUT-NOT-ACCEPTED",),
        }
    )

    assert captured["candidates"] == ()


def test_package_is_not_generation_ready_without_accepted_evidence() -> None:
    node = _fake_node()

    result = node.package_evidence_node(
        {
            "evidence_candidates": (),
            "evidence_units": ("STALE-UNIT",),
            "claim_audit": None,
        }
    )

    assert result["evidence_package_status"] == "empty"
    assert result["ready_for_generation"] is False
    assert result["final_evidence_context"] == ""
