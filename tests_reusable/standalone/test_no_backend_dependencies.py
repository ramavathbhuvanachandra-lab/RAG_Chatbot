"""Hard gate: production ai_platform code must not depend on legacy backend."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLATFORM = ROOT / "ai_platform"


def _backend_refs(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    refs: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "backend" or alias.name.startswith("backend."):
                    refs.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "backend" or module.startswith("backend."):
                refs.append(f"from {module} import ...")

    # Dynamic import paths are also production dependencies.
    if "backend." in source:
        refs.append("qualified backend reference in source text")

    return refs


def test_ai_platform_has_no_backend_dependency():
    assert PLATFORM.is_dir(), PLATFORM
    failures: dict[str, list[str]] = {}
    for path in sorted(PLATFORM.rglob("*.py")):
        refs = _backend_refs(path)
        if refs:
            failures[str(path.relative_to(ROOT))] = refs
    assert not failures, failures
