from __future__ import annotations

from pathlib import Path
import re


def node_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    direct = root / "ai_platform" / "core" / "nodes.py"
    graph = root / "ai_platform" / "core" / "graph" / "nodes.py"
    if direct.exists():
        return direct
    if graph.exists():
        return graph
    raise AssertionError("active ai_platform nodes.py not found")


def method_block(source: str, name: str, next_name: str) -> str:
    pattern = re.compile(
        rf"^    def {re.escape(name)}\(self, state: State\) -> State:\n.*?^    def {re.escape(next_name)}\(self, state: State\) -> State:",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(source)
    assert match, f"{name} block not found"
    return match.group(0)


def test_verifier_and_context_builder_are_on_the_active_path():
    source = node_path().read_text(encoding="utf-8")
    assert '"ai_platform.core.retrieval.verification"' in source
    assert "build_final_context_package" in source

    retrieve = method_block(source, "retrieve_and_qualify_node", "process_multi_intent_node")
    assert "assess_evidence_node(merged)" not in retrieve
    assert "assess_coverage_node(merged)" not in retrieve
    assert "audit_claims_node(merged)" not in retrieve
    assert "package_evidence_node(merged)" not in retrieve
    assert "evidence_context_node(merged)" in retrieve


def test_context_node_does_not_reverify_or_scope_filter_again():
    source = node_path().read_text(encoding="utf-8")
    context = method_block(source, "evidence_context_node", "assess_evidence_node")
    assert "verify" not in context.casefold()
    assert "filter_scope_conflicts" not in context
    assert "build_final_context_package" in context


def test_answer_gate_uses_final_context_and_keeps_post_generation_checks():
    source = node_path().read_text(encoding="utf-8")
    answer = method_block(source, "answer_node", "retrieve_and_qualify_node")
    assert "final_context_package" in answer
    assert "answer_generator" in answer
    assert "assess_answer_grounding" in answer
    assert "guard_answer" in answer
