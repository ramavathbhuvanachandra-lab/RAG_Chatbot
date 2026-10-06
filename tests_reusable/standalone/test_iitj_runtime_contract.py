"""IITJ runtime contract tests for the standalone AI platform."""

from __future__ import annotations


def test_iitj_runtime_config_contract() -> None:
    from ai_platform.runtime import config

    assert config.RUNTIME is not None
    assert config.RUNTIME.institution is not None
    assert config.RUNTIME.institution.institution_id == "iitj"


def test_iitj_runtime_retrieval_exports() -> None:
    from ai_platform.runtime import retrieval

    assert callable(retrieval.dense_retrieve)
    assert callable(retrieval.keyword_retrieve)
    assert retrieval.chunks
