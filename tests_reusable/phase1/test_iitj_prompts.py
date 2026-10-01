from __future__ import annotations

from backend.prompts import answer_prompt as compatibility_prompt
from backend.institutions.iitj.prompts import (
    answer_prompt as iitj_prompt,
)


def test_iitj_prompt_imports() -> None:
    assert iitj_prompt is not None


def test_compatibility_wrapper_points_to_iitj_prompt() -> None:
    assert compatibility_prompt is iitj_prompt


def test_prompt_has_expected_runtime_inputs() -> None:
    assert set(
        iitj_prompt.input_variables
    ) == {
        "chat_history",
        "context",
        "evidence_coverage",
        "question",
        "question_type",
    }


if __name__ == "__main__":
    print("IITJ prompt isolation tests passed.")
