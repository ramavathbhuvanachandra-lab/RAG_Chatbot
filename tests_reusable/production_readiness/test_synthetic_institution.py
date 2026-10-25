"""Synthetic Institution and Cross-Tenant Data Isolation Test.

Verifies:
1. Practical institution onboarding workflow:
   - Supply new institution profile and raw source corpus.
   - Build chunks and persistent cache.
   - Run queries without editing shared core retrieval or query-planning code.
2. Cross-tenant evidence isolation:
   - Queries under Synthetic Institution profile retrieve only synthetic documents.
   - Queries for Default Institution facts under Synthetic profile abstain.
   - Queries for Synthetic facts under Default Institution profile abstain.
3. Corpus modification / cache invalidation workflow:
   - Modifying a document triggers automatic signature change and cache refresh.
4. Multi-intent decomposition on unseen institutional domains.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from ai_platform.core.institution.profile import InstitutionProfile
from ai_platform.runtime.corpus import load_and_split, invalidate_chunk_cache, get_corpus_signature
from ai_platform.runtime.engine import inspect_context
from ai_platform.runtime.retrieval import get_canonical_chunks, get_bm25_retriever


@pytest.fixture
def synthetic_apex_institution(tmp_path: Path):
    """Create a synthetic institution with isolated documents and Chroma storage."""
    data_dir = tmp_path / "data_apex"
    data_dir.mkdir(parents=True, exist_ok=True)

    admissions_file = data_dir / "quantum_computing.txt"
    admissions_file.write_text(
        "The B.Tech in Quantum Computing at Apex Institute requires 160 credits for completion. "
        "The annual tuition fee for B.Tech in Quantum Computing is ₹1,20,000. "
        "Continuation of the Apex Excellence Scholarship requires a minimum cumulative GPA of 8.5.",
        encoding="utf-8",
    )

    hostel_file = data_dir / "hostel_rules.txt"
    hostel_file.write_text(
        "Hostel accommodation at Apex Institute is provided in Euler Hall. "
        "All first-year residents must pay a one-time refundable hostel caution deposit of ₹15,000. "
        "Dining mess advance is ₹22,000 per semester.",
        encoding="utf-8",
    )

    chroma_dir = tmp_path / "chroma_apex"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    profile = InstitutionProfile(
        institution_id="apex",
        display_name="Apex Institute of Science and Technology",
        data_path=data_dir,
        vectorstore_path=chroma_dir,
        vectorstore_collection="apex_v1",
        default_language="English",
        supported_languages=("English",),
    )
    profile.validate()
    return profile


def test_synthetic_institution_corpus_building_and_caching(synthetic_apex_institution):
    """Verify that a new institution's corpus is chunked and cached persistently."""
    profile = synthetic_apex_institution
    data_dir = profile.data_path

    # First load builds chunks and writes persistent cache
    chunks = load_and_split(data_dir)
    assert len(chunks) >= 2
    assert any("Quantum Computing" in c.page_content for c in chunks)
    assert any("Euler Hall" in c.page_content for c in chunks)

    # Verify persistent cache file was created
    cache_file = data_dir / ".canonical_chunks_cache.json"
    assert cache_file.exists()

    # Second load reads directly from persistent cache
    cached_chunks = load_and_split(data_dir)
    assert len(cached_chunks) == len(chunks)
    assert cached_chunks[0].page_content == chunks[0].page_content


def test_synthetic_institution_single_intent_retrieval(synthetic_apex_institution):
    """Verify inspect_context retrieves synthetic facts when given the synthetic profile."""
    profile = synthetic_apex_institution

    query = "What is the annual tuition fee for B.Tech in Quantum Computing at Apex Institute?"
    result = inspect_context(query, institution_profile=profile)

    assert result["ready_for_generation"] is True
    assert result["will_abstain"] is False

    context = result.get("final_evidence_context", "")
    assert "1,20,000" in context or "120000" in context
    assert "Quantum Computing" in context


def test_synthetic_institution_multi_intent_decomposition(synthetic_apex_institution):
    """Verify that multi-intent queries on a new institution decompose and package both facts."""
    profile = synthetic_apex_institution

    query = (
        "What is the annual tuition fee for B.Tech in Quantum Computing, "
        "and what is the one-time refundable hostel caution deposit?"
    )
    result = inspect_context(query, institution_profile=profile)

    assert result["is_multi_intent"] is True
    assert len(result["intent_questions"]) == 2

    context = result.get("final_evidence_context", "")
    # Verify tuition fee evidence is retained
    assert "1,20,000" in context or "120000" in context
    # Verify refundable caution deposit evidence is retained
    assert "15,000" in context or "15000" in context


def test_cross_tenant_isolation_abstention(synthetic_apex_institution):
    """Verify zero cross-tenant contamination between synthetic and default institutions."""
    apex_profile = synthetic_apex_institution

    # 1. Asking an IIT Jodhpur-specific policy under Apex profile MUST abstain
    iitj_query = "What is the minimum CGPA required for Ph.D. fellowship continuation at IITJ?"
    res_apex = inspect_context(iitj_query, institution_profile=apex_profile)
    assert res_apex["will_abstain"] is True or res_apex["evidence_status"] == "insufficient", (
        "Apex profile must not answer IITJ-specific fellowship policy"
    )

    # 2. Asking an Apex-specific fact under default IITJ profile MUST abstain
    apex_query = "What is the refundable caution deposit for Euler Hall at Apex Institute?"
    res_iitj = inspect_context(apex_query)
    assert res_iitj["will_abstain"] is True or res_iitj["evidence_status"] == "insufficient", (
        "Default IITJ profile must not answer Euler Hall facts from Apex"
    )


def test_corpus_modification_auto_invalidation(synthetic_apex_institution):
    """Verify that modifying corpus files updates signature and refreshes chunk cache."""
    profile = synthetic_apex_institution
    data_dir = profile.data_path

    # Initial load
    initial_chunks = load_and_split(data_dir)
    initial_sig = get_corpus_signature(data_dir)

    # Append a new policy document
    new_doc = data_dir / "library_policy.txt"
    new_doc.write_text(
        "The Ramanujan Library at Apex Institute is open 24 hours during examinations.",
        encoding="utf-8",
    )

    new_sig = get_corpus_signature(data_dir)
    assert new_sig != initial_sig, "Corpus signature must change when files are added"

    # Reload: cache miss should automatically detect changed signature and re-split
    refreshed_chunks = load_and_split(data_dir)
    assert len(refreshed_chunks) > len(initial_chunks)
    assert any("Ramanujan Library" in c.page_content for c in refreshed_chunks)
