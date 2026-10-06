from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_platform.institutions.iitj.profile import (
    DATA_ROOT,
    DISPLAY_NAME,
    INSTITUTION_ID,
    INSTITUTION_DATA_ROOT,
    PLATFORM_ROOT,
    PROFILE,
    VECTORSTORE_COLLECTION,
)
from ai_platform.institutions.iitj.lexical import (
    canonical_terms,
    expand_query,
    get_lexical_profile,
    normalize_for_institution,
    score_pair,
)
from ai_platform.institutions.iitj.navigation import NAVIGATION_MODEL
from ai_platform.institutions.iitj.scope_policy import SCOPE_POLICY
from ai_platform.institutions.iitj.semantic_registry import SEMANTIC_REGISTRY
from ai_platform.institutions.iitj.source_policy import SOURCE_POLICY


REQUIRED_DATA_FILES = (
    "aliases.json",
    "acronyms.json",
    "phrases.json",
    "semantic_terms.json",
    "source_policy.json",
    "scope_policy.json",
    "navigation.json",
)


@pytest.mark.parametrize("filename", REQUIRED_DATA_FILES)
def test_required_data_file_exists_and_is_json_object(filename: str) -> None:
    path = DATA_ROOT / filename
    assert path.is_file(), path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_profile_points_to_new_platform_root() -> None:
    assert INSTITUTION_ID == "iitj"
    assert DISPLAY_NAME == "IIT Jodhpur"
    assert PLATFORM_ROOT.name == "ai_platform"
    assert INSTITUTION_DATA_ROOT == PLATFORM_ROOT.parent / "data" / "data_iitj"
    assert VECTORSTORE_COLLECTION == "iitj_v1"
    PROFILE.validate()


def test_program_boundary_does_not_match_mtechnique() -> None:
    assert "mtech" in canonical_terms("M.Technique is not M.Tech.")
    assert "mtech" not in canonical_terms("M.Technique is not M.Technique.")


def test_masters_technology_alias_normalizes_to_mtech() -> None:
    normalized = normalize_for_institution("What are the masters tech eligibility requirements?")
    assert "mtech" in normalized
    assert "mtech" in canonical_terms("masters tech eligibility requirements")


def test_ms_by_research_is_distinct_program() -> None:
    assert "ms_by_research" in SEMANTIC_REGISTRY.detect_programs(
        "M.S. by Research admission requirements"
    )
    assert "mtech" not in SEMANTIC_REGISTRY.detect_programs(
        "M.S. by Research admission requirements"
    )


def test_semantic_registry_recovers_question_structure() -> None:
    result = SEMANTIC_REGISTRY.describe(
        "What are the M.Tech eligibility requirements in Electrical Engineering?"
    )
    assert "mtech" in result["programs"]
    assert "eligibility" in result["topics"]
    assert "electrical_engineering" in result["entities"]
    assert "eligibility" in result["attributes"]


def test_semantic_registry_handles_generic_location_intent() -> None:
    result = SEMANTIC_REGISTRY.describe("Where is the institute located?")
    assert "location" in result["attributes"]


def test_lexical_profile_is_loaded_and_valid() -> None:
    profile = get_lexical_profile()
    assert profile.institution_id == INSTITUTION_ID
    assert profile.aliases
    assert profile.acronyms
    assert profile.phrases
    assert profile.canonicalization


def test_lexical_pair_is_bounded_and_phrase_aware() -> None:
    match = score_pair(
        "M.Tech eligibility requirements",
        "M.Tech eligibility requirements are listed here.",
    )
    assert 0.0 <= match.query_overlap <= 1.0
    assert match.matched_phrases


def test_query_expansion_keeps_original_first_and_is_bounded() -> None:
    query = "What are the M.Tech eligibility requirements?"
    expanded = expand_query(query)
    assert expanded
    assert expanded[0] == query
    assert len(expanded) <= 3


def test_source_policy_prefers_admissions_for_eligibility() -> None:
    source = "data/data_iitj/iitj_rag_v1_docs_production/admissions/mtech_admissions.docx"
    assert SOURCE_POLICY.classify_source(source) == ("admissions",)
    assert SOURCE_POLICY.source_alignment("eligibility", source) == 1.0


def test_scope_policy_rejects_conflicting_program_dimension() -> None:
    query = SCOPE_POLICY.detect("M.Tech eligibility")
    evidence = SCOPE_POLICY.detect("B.Tech eligibility")
    assert "program" in SCOPE_POLICY.conflicting_dimensions(query, evidence)


def test_navigation_model_is_conceptual_not_answer_content() -> None:
    assert "directions" in NAVIGATION_MODEL.detect("How do I reach the library?")


def test_semantic_registry_is_institution_scoped() -> None:
    assert SEMANTIC_REGISTRY.institution_id == INSTITUTION_ID
    SEMANTIC_REGISTRY.validate()
