"""Regression test for packaging unit capacity expansion (max_units=32).

Ensures that:
1. Verified evidence units beyond index 16 are not dropped when total character
   budget (12,000 chars) is available.
2. Hard boundaries (max_context_chars, max_units_per_source) remain strictly enforced.
3. Complex institutional queries with multi-category constraints (e.g. B.Tech branch
   change CGPA requirements for General vs SC/ST/PwD) preserve their essential criteria.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_platform.core.evidence.claims import EvidenceUnit
from ai_platform.core.evidence.packaging import build_evidence_package
from ai_platform.core.graph.nodes import CoreNodes, default_dependencies


def test_packaging_preserves_verified_units_beyond_16():
    """Verify that packages can include up to 32 units across diverse sources."""
    units = [
        EvidenceUnit(
            evidence_id=f"doc_{i // 5}:unit_{i}",
            text=f"Rule statement number {i} covering institutional regulation detail.",
            source=f"office/division_{i // 5}/regulations.docx",
            position=i,
        )
        for i in range(28)
    ]

    pkg = build_evidence_package(
        units,
        max_units=32,
        max_units_per_source=12,
        verified_evidence=True,
    )

    assert pkg.status in {"ready", "partial"}
    assert len(pkg.items) == 28, f"Expected 28 items preserved, got {len(pkg.items)}"
    assert "Rule statement number 27" in pkg.context


def test_packaging_respects_source_and_character_caps():
    """Verify that source diversity (max 12 per source) and char budget are strictly enforced."""
    single_source_units = [
        EvidenceUnit(
            evidence_id=f"single_src:unit_{i}",
            text=f"Single source item {i} with some descriptive text.",
            source="single_department/rules.docx",
            position=i,
        )
        for i in range(30)
    ]

    pkg = build_evidence_package(
        single_source_units,
        max_units=32,
        max_units_per_source=12,
        max_context_chars=12_000,
        verified_evidence=True,
    )

    assert len(pkg.items) <= 12, f"Expected at most 12 items from single source, got {len(pkg.items)}"
    assert len(pkg.context) <= 12_000


def test_btech_branch_change_cgpa_survives_into_package():
    """End-to-end regression test for gen_dev_003 CGPA facet survival into final evidence package."""
    deps = default_dependencies()
    nodes = CoreNodes(deps)

    q = "What are the CGPA requirements for branch change after Semester II for General category versus SC/ST/PwD students?"
    state = {"question": q}
    state.update(nodes.resolve_conversation_node(state))
    state.update(nodes.understand_query_node(state))
    state.update(nodes.hybrid_retrieve_node(state))
    state.update(nodes.fuse_retrieved_documents_node(state))
    state.update(nodes.verify_and_rank_node(state))
    state.update(nodes.evidence_boundary_node(state))
    state.update(nodes.assess_evidence_node(state))
    state.update(nodes.assess_coverage_node(state))
    state.update(nodes.audit_claims_node(state))
    state.update(nodes.package_evidence_node(state))

    pkg = state.get("evidence_package")
    assert pkg is not None
    assert pkg.ready_for_generation is True

    context_lower = pkg.context.lower()
    assert "8.5" in context_lower, "Expected General category CGPA 8.5 threshold in packaged context"
    assert "7.5" in context_lower, "Expected SC/ST/PwD category CGPA 7.5 threshold in packaged context"
