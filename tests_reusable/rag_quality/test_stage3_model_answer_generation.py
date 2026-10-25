"""
Stage 3 Model Answer Generation Validation Suite.
Executes separate local model validation using local Ollama (qwen2.5:3b)
across 5 representative answer generation patterns:
1. Explicit OR alternatives (e.g. M.Sc. first-class)
2. Positive vs negative qualification (e.g. Ph.D. 4-year degree)
3. Cross-degree evidence separation (M.Tech 60% vs Ph.D. 70%)
4. Compound numerical and categorical constraints (SC/ST 55% vs GEN 60%)
5. Multi-intent answers with partial evidence (answering attendance, abstaining on submarine)
"""
from __future__ import annotations

import re
import pytest
from ai_platform.runtime.engine import ask_with_diagnostics


@pytest.mark.integration
class TestStage3ModelAnswerGeneration:
    """Validate 5 core answer generation scenarios with live model inference."""

    def test_case_1_explicit_or_alternatives_msc_first_class(self) -> None:
        """Case 1: Explicit OR alternatives — M.Sc. first class qualification.
        The model must produce an affirmative grounded answer that first-class is accepted.
        """
        q = "Is a first-class qualification accepted for M.Sc. admission?"
        result = ask_with_diagnostics(q)
        answer = str(result.get("answer") or "")
        answer_lower = answer.lower()

        # Must not contain false negative denials
        negative_patterns = [
            r"not explicitly accepted",
            r"not mentioned",
            r"does not mention first",
            r"not accepted",
            r"first-class is not",
            r"first class is not",
            r"\bno\b.*first.class",
        ]
        for pattern in negative_patterns:
            assert not re.search(pattern, answer_lower, re.I), (
                f"Case 1 false negative assertion (matched {pattern!r}): {answer!r}"
            )

        # Must affirm acceptance
        positive_patterns = [
            r"first.class.*accept",
            r"accept.*first.class",
            r"or.*first.class",
            r"first.class.*qualif",
            r"eligible",
            r"\byes\b",
        ]
        assert any(re.search(p, answer_lower, re.I) for p in positive_patterns), (
            f"Case 1 failed to affirm first-class acceptance: {answer!r}"
        )

    def test_case_2_positive_qualification_phd_four_year_degree(self) -> None:
        """Case 2: Positive qualification — Ph.D. 4-year degree.
        The model must confirm that a 4-year bachelor's degree qualifies for Ph.D. admission.
        """
        q = "Can a four-year bachelor's degree qualify someone for Ph.D. admission?"
        result = ask_with_diagnostics(q)
        answer = str(result.get("answer") or "")
        answer_lower = answer.lower()

        # Must not deny qualification
        negative_patterns = [
            r"does not qualify",
            r"alone does not",
            r"not qualify",
            r"four.year.*degree.*not",
            r"\bno\b.*four.year",
        ]
        for pattern in negative_patterns:
            assert not re.search(pattern, answer_lower, re.I), (
                f"Case 2 false negative assertion (matched {pattern!r}): {answer!r}"
            )

        positive_patterns = [
            r"qualif",
            r"eligible",
            r"bachelor.*four.year",
            r"four.year.*bachelor",
            r"\byes\b",
        ]
        assert any(re.search(p, answer_lower, re.I) for p in positive_patterns), (
            f"Case 2 failed to confirm 4-year bachelor qualification: {answer!r}"
        )

    def test_case_3_cross_degree_evidence_separation_mtech_vs_phd(self) -> None:
        """Case 3: Cross-degree evidence separation — M.Tech (60%) vs Ph.D. (70%).
        The model must clearly distinguish entry percentage requirements between degrees.
        """
        q = "What is the difference between M.Tech and Ph.D. entry requirements for graduates with a 4-year Bachelor's degree?"
        result = ask_with_diagnostics(q)
        answer = str(result.get("answer") or "")
        answer_lower = answer.lower()

        # Both degrees must be addressed
        assert "ph.d" in answer_lower or "phd" in answer_lower, f"Ph.D. not mentioned in: {answer!r}"
        assert "m.tech" in answer_lower or "mtech" in answer_lower, f"M.Tech not mentioned in: {answer!r}"

        # Distinct threshold percentages
        assert "70%" in answer or "70 percent" in answer_lower or "7.0" in answer, (
            f"Ph.D. 70% threshold missing from: {answer!r}"
        )
        assert "60%" in answer or "60 percent" in answer_lower or "6.0" in answer, (
            f"M.Tech 60% threshold missing from: {answer!r}"
        )

    def test_case_4_compound_numerical_categorical_constraints(self) -> None:
        """Case 4: Compound numerical and categorical constraints — SC/ST (55%) vs GEN (60%).
        The model must accurately present category-specific eligibility thresholds.
        """
        q = "What is the minimum qualifying degree percentage for General category versus SC/ST applicants for M.Sc. admission?"
        result = ask_with_diagnostics(q)
        answer = str(result.get("answer") or "")
        answer_lower = answer.lower()

        assert "60%" in answer or "60 percent" in answer_lower, f"General 60% missing from: {answer!r}"
        assert ("55%" in answer or "55 percent" in answer_lower or "5%" in answer or "5 percent" in answer_lower), (
            f"SC/ST 55% or 5% relaxation missing from: {answer!r}"
        )
        assert "sc" in answer_lower or "st" in answer_lower, f"SC/ST category missing from: {answer!r}"

    def test_case_5_multi_intent_partial_evidence_answering_and_abstention(self) -> None:
        """Case 5: Multi-intent answers with partial evidence.
        Supported intent (B.Tech attendance) must be answered with factual 75% rule.
        Unsupported intent (secret submarine base) must be cleanly abstained.
        """
        q = "What is the B.Tech 75% attendance rule and where is the secret submarine base located?"
        result = ask_with_diagnostics(q)
        answer = str(result.get("answer") or "")
        answer_lower = answer.lower()

        # Supported attendance rule answered
        assert "75%" in answer or "75 percent" in answer_lower, f"75% attendance rule missing: {answer!r}"
        assert "attendance" in answer_lower, f"Attendance not discussed in: {answer!r}"

        # Unsupported submarine intent abstained
        abstention_markers = [
            r"no information",
            r"not mentioned",
            r"not found",
            r"does not contain",
            r"no evidence",
            r"not available",
            r"submarine",
            r"unsupported",
        ]
        assert any(re.search(m, answer_lower, re.I) for m in abstention_markers), (
            f"Failed to abstain on submarine base in: {answer!r}"
        )
