"""
pytest configuration for tests_reusable/rag_quality/.

Registers the `integration` marker so pytest does not emit PytestUnknownMarkWarning.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require Ollama (qwen2.5:3b) and ChromaDB to be running. "
        "Deselect with: -m 'not integration'",
    )
