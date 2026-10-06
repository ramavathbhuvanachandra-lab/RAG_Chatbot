from pathlib import Path

NODE = Path("ai_platform/core/graph/nodes.py")


def test_graph_uses_lazy_evidence_boundary_import():
    text = NODE.read_text(encoding="utf-8")
    class_body = text.split("class CoreNodes", 1)[1]
    assert "from ai_platform.core.retrieval.evidence_boundary import" not in text.split("class CoreNodes", 1)[0]
    assert "build_evidence_boundary(" in class_body


def test_graph_evidence_stage_consumes_only_evidence_candidates():
    text = NODE.read_text(encoding="utf-8")
    start = text.index("    def assess_evidence_node")
    end = text.index("    def assess_coverage_node", start)
    block = text[start:end]
    assert 'state.get("evidence_candidates", ())' in block
    assert 'state.get("ranked_candidates", ())' not in block


def test_graph_coverage_consumes_only_evidence_candidates():
    text = NODE.read_text(encoding="utf-8")
    start = text.index("    def assess_coverage_node")
    end = text.index("    def audit_claims_node", start)
    block = text[start:end]
    assert 'state.get("evidence_candidates", ())' in block
    assert 'state.get("ranked_candidates", ())' not in block


def test_audit_has_no_ranked_fallback():
    text = NODE.read_text(encoding="utf-8")
    start = text.index("    def audit_claims_node")
    end = text.index("    def package_evidence_node", start)
    block = text[start:end]
    assert 'state.get("ranked_candidates", ())' not in block


def test_package_fails_closed_without_evidence_candidates():
    text = NODE.read_text(encoding="utf-8")
    start = text.index("    def package_evidence_node")
    end = text.index("    def process_multi_intent_node", start)
    block = text[start:end]
    assert "if not evidence_candidates:" in block
    assert "and units" in block
