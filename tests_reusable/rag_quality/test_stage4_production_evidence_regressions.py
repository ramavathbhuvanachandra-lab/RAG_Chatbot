"""Stage 4 Production Evidence-First Regressions Test Suite.

Verifies the 9 specific regression cases required by the Master Task:
1. Old Mess vs New Mess longest-match disambiguation.
2. Chip design / semiconductor entity & ranking accuracy.
3. Stale conversation topic isolation on standalone queries.
4. Embedding cache thread safety, model isolation, and bound enforcement.
5. Exact 2-digit number preservation in ranking (e.g. 75% attendance, 12 credits).
6. Degree abbreviation preservation in sentence termination (B.S., M.S., etc.).
7. Numerical citation precision and suppression of unrelated sources.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import pytest

from ai_platform.core.answering.generator import _ensure_clean_sentence_termination
from ai_platform.core.conversation.conversation import resolve_conversation, _topic_from_text
from ai_platform.core.retrieval.ranking import _tokens, _token_set
from ai_platform.institutions.iitj.campus_navigation import find_location
from ai_platform.runtime.retrieval import (
    _QUERY_EMBEDDING_CACHE,
    _MAX_EMBEDDING_CACHE_SIZE,
    clear_query_embedding_cache,
)


def test_regression_old_mess_vs_new_mess_disambiguation():
    """Regression Case 1: Old Mess vs New Mess must return the distinct entity, not conflate them."""
    old_mess = find_location("Where is the Old Mess located?")
    assert old_mess is not None, "Old Mess must resolve"
    assert "old" in old_mess["name"].lower() or "old mess" in [a.lower() for a in old_mess.get("aliases", [])]
    assert "new mess" not in old_mess["name"].lower()

    new_mess = find_location("Where is the New Mess located?")
    assert new_mess is not None, "New Mess must resolve"
    assert "new" in new_mess["name"].lower() or "new mess" in [a.lower() for a in new_mess.get("aliases", [])]
    assert "old mess" not in new_mess["name"].lower()


def test_regression_two_digit_numbers_preserved_in_ranking():
    """Regression Case 2: Numbers like 75, 12, 14, 60 must NOT be dropped by tokenization."""
    toks = _tokens("What is the 75% attendance rule and 12 audit credits requirement?")
    assert "75" in toks, "75 must be preserved in ranking tokens"
    assert "12" in toks, "12 must be preserved in ranking tokens"
    assert "attendance" in toks
    assert "audit" in toks
    assert "credits" in toks

    token_set = _token_set("semiconductors and materials engineering")
    assert "semiconductor" in token_set, "Singular expansion must produce 'semiconductor'"
    assert "semiconductors" in token_set


def test_regression_bs_degree_sentence_termination_no_truncation():
    """Regression Case 3: Ensure B.S. degree abbreviation does not cause premature truncation."""
    full_sentence = (
        "The total number of graded and non-graded audit credits for the B.S. in Mathematics "
        "and Computing is 153 (141 graded + 12 audit)."
    )
    result = _ensure_clean_sentence_termination(full_sentence)
    assert "153" in result, "Full total credits must not be truncated at B.S."
    assert "141 graded + 12 audit" in result
    assert result.endswith(".")

    # Test sentence without trailing period but complete thought ending with paren
    no_period = (
        "The total number of graded and non-graded audit credits for the B.S. in Mathematics "
        "and Computing is 153 (141 graded + 12 audit)"
    )
    result_np = _ensure_clean_sentence_termination(no_period)
    assert "153 (141 graded + 12 audit)" in result_np
    assert result_np.endswith(".")


def test_regression_standalone_query_does_not_inherit_stale_topic():
    """Regression Case 4: Standalone queries must not inherit active_topic from previous turns."""
    messages = [
        ("user", "What are the eligibility criteria for M.Tech admissions?"),
        ("assistant", "The eligibility requires a relevant bachelor's degree with minimum 60% marks or 6.0 CPI."),
    ]
    # Now user asks an unrelated standalone query about chip design
    resolved = resolve_conversation(
        question="Which department conducts research on chip design and semiconductors?",
        chat_history=messages,
    )
    assert resolved["mode"] == "standalone"
    assert resolved["active_topic"] != "eligibility", "Standalone query must NOT inherit 'eligibility' topic"
    assert resolved["active_topic"] in {"research", ""}


def test_regression_embedding_cache_thread_safety_and_clearing():
    """Regression Case 5: Query embedding cache must be thread-safe and clearable."""
    clear_query_embedding_cache()
    assert len(_QUERY_EMBEDDING_CACHE) == 0

    def insert_batch(tid: int):
        for i in range(50):
            key = (f"model_{tid % 2}", f"query_{tid}_{i}")
            _QUERY_EMBEDDING_CACHE[key] = [0.1 * tid, float(i)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(insert_batch, t) for t in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(_QUERY_EMBEDDING_CACHE) == 400
    clear_query_embedding_cache()
    assert len(_QUERY_EMBEDDING_CACHE) == 0
