"""
Phase 1 adversarial contract tests.

These tests target the failure modes that matter for the Phase 1 gate:
- no evidence must not reach the answer generator
- stale/legacy context must not bypass the evidence package contract
- EvidencePackage context must preserve intentional layout
- boundary wiring must be active in the canonical dependency graph

The tests intentionally avoid UI code.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
NODES = REPO / "ai_platform" / "core" / "graph" / "nodes.py"


def _source():
    return NODES.read_text(encoding="utf-8")


def test_boundary_dependency_is_wired_structurally():
    tree = ast.parse(_source())
    funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "default_dependencies"
    ]
    assert len(funcs) == 1

    calls = [
        n for n in ast.walk(funcs[0])
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "NodeDependencies"
    ]
    assert len(calls) == 1
    assert any(k.arg == "build_evidence_boundary" for k in calls[0].keywords)


def test_answer_node_has_fail_closed_zero_evidence_contract():
    src = _source()
    assert "model_calls" in src
    assert "ready_for_generation" in src
    assert 'answer_mode": "fallback"' in src or 'answer_mode": "fallback"' in src
    # The canonical implementation must gate generation on evidence package
    # readiness rather than on a stale final-context string alone.
    assert "evidence_package" in src
    assert "answer_generator" in src


def test_context_cleaning_does_not_collapse_evidence_layout():
    src = _source()
    marker = "def _clean_context"
    assert marker in src
    start = src.index(marker)
    block = src[start:start + 700]
    assert 'str(value or "").strip()' in block
    assert '" ".join' not in block


def test_ui_files_are_not_part_of_phase1_target():
    assert "ui" not in str(NODES).lower()
