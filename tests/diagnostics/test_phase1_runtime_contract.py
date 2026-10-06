"""
Phase 1 runtime-contract checks.

These tests are deliberately structural and import-safe. They verify the
specific V6 failure cannot recur:
- canonical evidence-boundary symbol is actually imported;
- NodeDependencies receives the dependency;
- both pieces are present in the same canonical module;
- no UI file is involved.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
NODES = REPO / "ai_platform" / "core" / "graph" / "nodes.py"


def test_boundary_symbol_is_imported_and_wired_in_same_module():
    source = NODES.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "ai_platform.core.retrieval.evidence_boundary"
    ]
    assert len(imports) == 1
    assert any(
        alias.name == "build_evidence_boundary"
        for alias in imports[0].names
    )

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "default_dependencies"
    ]
    assert len(functions) == 1

    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "NodeDependencies"
    ]
    assert len(calls) == 1
    assert any(
        keyword.arg == "build_evidence_boundary"
        for keyword in calls[0].keywords
    )


def test_future_import_precedes_boundary_import():
    source = NODES.read_text(encoding="utf-8")
    tree = ast.parse(source)

    future_lines = [
        node.end_lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
    ]
    boundary_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "ai_platform.core.retrieval.evidence_boundary"
    ]

    assert boundary_lines
    assert not future_lines or max(future_lines) < min(boundary_lines)


def test_context_preservation_helper_does_not_collapse_whitespace():
    source = NODES.read_text(encoding="utf-8")
    start = source.index("def _clean_context")
    block = source[start : start + 700]
    assert 'str(value or "").strip()' in block
    assert '" ".join' not in block


def test_phase1_target_is_not_ui():
    assert "ui" not in str(NODES).lower()
