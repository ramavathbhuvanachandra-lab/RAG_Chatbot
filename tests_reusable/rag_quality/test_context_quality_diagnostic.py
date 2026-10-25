"""Targeted tests for zero-inference pre-generation context diagnostics and multi-intent packaging.

Validates:
1. Zero-inference context inspection (inspect_context) returns structured diagnostics
   without invoking Ollama or external model calls.
2. Multi-intent queries retain verified evidence across distinct sub-questions with
   up to 32 evidence units.
3. Fail-closed abstention is strictly enforced on out-of-scope and unsupported temporal queries.
4. Institution-agnostic core behavior: Planner and claims matching operate independently
   of hardcoded institutional names.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from ai_platform.runtime.engine import inspect_context
from ai_platform.core.query.planner import _is_institution_name_or_generic


def test_inspect_context_structure():
    """Verify that inspect_context returns complete pre-generation context without LLM inference."""
    q = "What is the minimum CGPA required for B.Tech branch change?"
    result = inspect_context(q)

    assert result["question"] == q
    assert result["pipeline_stop"] is False
    assert result["ready_for_generation"] is True
    assert result["will_abstain"] is False
    assert result["evidence_unit_count"] > 0
    assert result["context_character_count"] > 0
    assert result["estimated_token_count"] == result["context_character_count"] // 4

    # Verify structured evidence items
    units = result["evidence_units"]
    assert len(units) == result["evidence_unit_count"]
    for u in units:
        assert "evidence_id" in u
        assert "source" in u
        assert "text" in u
        assert len(u["text"]) > 0

    # Verify prompt formatting
    prompts = result["prompt_messages"]
    assert len(prompts) == 2
    assert prompts[0]["role"] == "system"
    assert prompts[1]["role"] == "user"
    assert "VERIFIED FACTS:" in prompts[1]["content"]
    assert result["final_evidence_context"] in prompts[1]["content"]


def test_inspect_context_out_of_scope_abstention():
    """Verify that an out-of-scope question halts at query understanding with clean abstention."""
    q = "What is the recipe for chocolate cake?"
    result = inspect_context(q)

    assert result["pipeline_stop"] is True
    assert result["ready_for_generation"] is False
    assert result["will_abstain"] is True
    assert result["abstention_reason"] in {"out_of_scope", "unrelated_cooking_recipe_request"} or "unrelated" in result["abstention_reason"]
    assert result["final_evidence_context"] == ""
    assert result["context_character_count"] == 0
    assert result["prompt_messages"] == []
    assert len(result["fallback_response"]) > 0


def test_inspect_context_temporal_abstention():
    """Verify that unsupported temporal constraints trigger fail-closed abstention."""
    q = "What will be the B.Tech tuition fee in the year 2045?"
    result = inspect_context(q)

    assert result["pipeline_stop"] is False
    assert result["ready_for_generation"] is False
    assert result["will_abstain"] is True
    assert result["abstention_reason"] == "unsupported_temporal_year"
    assert "2045" in result["unsupported_years"]
    assert result["final_evidence_context"] == ""
    assert result["prompt_messages"] == []


def test_inspect_context_multi_intent_propagation():
    """Verify multi-intent questions retain verified evidence for distinct sub-intents."""
    q = "What is the M.Tech eligibility requirement and what is the application fee?"
    result = inspect_context(q)

    assert result["is_multi_intent"] is True
    assert len(result["intent_questions"]) == 2
    assert result["ready_for_generation"] is True
    assert result["will_abstain"] is False
    assert result["evidence_unit_count"] >= 16

    # Verify that the generated prompt has explicit instructions and numbered items
    prompts = result["prompt_messages"]
    assert len(prompts) == 2
    user_content = prompts[1]["content"]
    assert "1. What is the M.Tech eligibility requirement" in user_content
    assert "2. what is the application fee?" in user_content


def test_institution_agnostic_planner_markers():
    """Verify that institutional tokens are handled dynamically and generically."""
    assert _is_institution_name_or_generic("institute") is True
    assert _is_institution_name_or_generic("the institute") is True
    assert _is_institution_name_or_generic("university") is True
    assert _is_institution_name_or_generic("college") is True
    assert _is_institution_name_or_generic("campus") is True
    assert _is_institution_name_or_generic("iit") is True
    assert _is_institution_name_or_generic("nit") is True

    # Real academic subject specializations must NOT be treated as generic institutional names
    assert _is_institution_name_or_generic("computer science") is False
    assert _is_institution_name_or_generic("mechanical engineering") is False
    assert _is_institution_name_or_generic("bioscience and bioengineering") is False
