from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.evidence.claims import (  # noqa: E402
    Claim,
    ClaimAudit,
    ClaimEvidenceMatch,
    EvidenceUnit,
)
from backend.core.evidence.packaging import (  # noqa: E402
    build_evidence_package,
    sanitize_evidence_text,
)


def _unit(evidence_id: str, text: str, source: str = "source-a", heading: str | None = None) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        text=text,
        source=source,
        position=0,
        heading=heading,
    )


def _audit(
    *,
    claim_ids: tuple[str, ...],
    matches: tuple[ClaimEvidenceMatch, ...],
    status_claims: str = "supported",
) -> ClaimAudit:
    claims = tuple(
        Claim(
            claim_id=claim_id,
            text=f"claim {claim_id}",
            kind="request",
            name="process",
        )
        for claim_id in claim_ids
    )
    supported = tuple(claim_ids) if status_claims == "supported" else ()
    partial = tuple(claim_ids) if status_claims == "partial" else ()
    return ClaimAudit(
        claims=claims,
        matches=matches,
        supported_claim_ids=supported,
        partial_claim_ids=partial,
        unsupported_claim_ids=(),
        conflicting_claim_ids=(),
    )


def test_supported_claim_evidence_is_selected_first():
    evidence = [
        _unit("e1", "General unrelated content."),
        _unit("e2", "The application process requires an online form."),
    ]
    audit = _audit(
        claim_ids=("c1",),
        matches=(ClaimEvidenceMatch("c1", "e2", 0.9, "supported"),),
    )
    package = build_evidence_package(evidence, claim_audit=audit)
    assert package.status == "ready"
    assert [item.evidence_id for item in package.items] == ["e2"]
    assert "General unrelated content" not in package.context


def test_partial_evidence_can_be_included_when_enabled():
    evidence = [_unit("e1", "The process has several steps.")]
    audit = _audit(
        claim_ids=("c1",),
        matches=(ClaimEvidenceMatch("c1", "e1", 0.3, "partial"),),
        status_claims="partial",
    )
    package = build_evidence_package(evidence, claim_audit=audit, include_partial=True)
    assert package.status == "partial"
    assert package.partial_claim_ids == ("c1",)


def test_partial_evidence_is_excluded_when_disabled():
    evidence = [_unit("e1", "The process has several steps.")]
    audit = _audit(
        claim_ids=("c1",),
        matches=(ClaimEvidenceMatch("c1", "e1", 0.3, "partial"),),
        status_claims="partial",
    )
    package = build_evidence_package(evidence, claim_audit=audit, include_partial=False)
    assert package.status == "empty"
    assert package.items == ()


def test_unsupported_only_evidence_is_not_packaged_with_audit():
    evidence = [_unit("e1", "Semantically similar but unrelated text.")]
    audit = _audit(
        claim_ids=("c1",),
        matches=(ClaimEvidenceMatch("c1", "e1", 0.0, "unsupported"),),
    )
    package = build_evidence_package(evidence, claim_audit=audit)
    assert package.status == "empty"
    assert package.context == ""


def test_no_claim_audit_preserves_ordered_evidence():
    evidence = [
        _unit("e1", "First evidence."),
        _unit("e2", "Second evidence.", source="source-b"),
    ]
    package = build_evidence_package(evidence)
    assert [item.evidence_id for item in package.items] == ["e1", "e2"]
    assert package.context.startswith("Evidence 1\nFirst evidence.")


def test_exact_duplicate_content_is_removed_per_source():
    evidence = [
        _unit("e1", "Same factual text."),
        _unit("e2", "Same factual text."),
        _unit("e3", "Same factual text.", source="source-b"),
    ]
    package = build_evidence_package(evidence)
    assert [item.evidence_id for item in package.items] == ["e1", "e3"]


def test_source_diversity_limit_is_enforced():
    evidence = [
        _unit("e1", "A1", source="source-a"),
        _unit("e2", "A2", source="source-a"),
        _unit("e3", "A3", source="source-a"),
        _unit("e4", "B1", source="source-b"),
    ]
    package = build_evidence_package(evidence, max_units_per_source=2)
    assert [item.evidence_id for item in package.items] == ["e1", "e2", "e4"]


def test_global_unit_limit_is_enforced():
    evidence = [_unit(f"e{i}", f"Text {i}", source=f"source-{i}") for i in range(1, 6)]
    package = build_evidence_package(evidence, max_units=3)
    assert len(package.items) == 3
    assert package.omitted_count == 2


def test_context_limit_never_truncates_a_unit():
    evidence = [
        _unit("e1", "A short fact.", source="source-a"),
        _unit("e2", "A second fact that would exceed the context budget.", source="source-b"),
    ]
    package = build_evidence_package(evidence, max_context_chars=30)
    assert len(package.items) == 1
    assert package.items[0].text == "A short fact."
    assert package.items[0].text in package.context


def test_internal_wrapper_is_removed_from_rendered_context():
    text = "Facts here.\n\nCommand 5 • Retrieval Representation\nOriginal source urls preserved from the Command 5 knowledge source."
    cleaned = sanitize_evidence_text(text)
    assert cleaned == "Facts here."


def test_source_provenance_is_preserved_outside_rendered_context():
    evidence = [_unit("e1", "A useful fact.", source="/internal/source/file.docx")]
    package = build_evidence_package(evidence)
    assert package.items[0].source == "/internal/source/file.docx"
    assert "/internal/source/file.docx" not in package.context


def test_heading_and_claim_provenance_are_kept():
    evidence = [_unit("e1", "Required form submission.", heading="Application", source="source-a")]
    audit = _audit(
        claim_ids=("c1",),
        matches=(ClaimEvidenceMatch("c1", "e1", 0.9, "supported"),),
    )
    package = build_evidence_package(evidence, claim_audit=audit)
    item = package.items[0]
    assert item.heading == "Application"
    assert item.supported_claim_ids == ("c1",)


def test_conflicted_claim_audit_marks_package_conflicted():
    evidence = [_unit("e1", "Value A."), _unit("e2", "Value B.", source="source-b")]
    audit = ClaimAudit(
        claims=(Claim("c1", "c1", "numeric", name="value", value=1, unit="currency"),),
        matches=(
            ClaimEvidenceMatch("c1", "e1", 0.9, "supported"),
            ClaimEvidenceMatch("c1", "e2", 0.9, "supported"),
        ),
        supported_claim_ids=("c1",),
        partial_claim_ids=(),
        unsupported_claim_ids=(),
        conflicting_claim_ids=("c1",),
    )
    package = build_evidence_package(evidence, claim_audit=audit)
    assert package.status == "conflicted"
    assert package.items


def test_invalid_limits_fail_safe_to_empty_package():
    evidence = [_unit("e1", "Fact.")]
    assert build_evidence_package(evidence, max_units=0).status == "empty"
    assert build_evidence_package(evidence, max_units_per_source=0).status == "empty"
    assert build_evidence_package(evidence, max_context_chars=0).status == "empty"


def test_ready_for_generation_requires_items():
    empty = build_evidence_package([])
    assert not empty.ready_for_generation
    ready = build_evidence_package([_unit("e1", "Fact.")])
    assert ready.ready_for_generation


def test_to_dict_is_json_friendly():
    package = build_evidence_package([_unit("e1", "Fact.")])
    value = package.to_dict()
    assert isinstance(value["items"], list)
    assert isinstance(value["supported_claim_ids"], list)
    assert isinstance(value["ready_for_generation"], bool)


def test_supported_and_partial_ids_are_tracked_separately():
    evidence = [
        _unit("e1", "Supported fact."),
        _unit("e2", "Partial fact.", source="source-b"),
    ]
    audit = ClaimAudit(
        claims=(
            Claim("c1", "c1", "request", name="process"),
            Claim("c2", "c2", "request", name="process"),
        ),
        matches=(
            ClaimEvidenceMatch("c1", "e1", 0.9, "supported"),
            ClaimEvidenceMatch("c2", "e2", 0.3, "partial"),
        ),
        supported_claim_ids=("c1",),
        partial_claim_ids=("c2",),
        unsupported_claim_ids=(),
        conflicting_claim_ids=(),
    )
    package = build_evidence_package(evidence, claim_audit=audit)
    assert package.supported_claim_ids == ("c1",)
    assert package.partial_claim_ids == ("c2",)


def test_items_are_immutable_contracts():
    evidence = [_unit("e1", "Fact.")]
    package = build_evidence_package(evidence)
    try:
        package.items[0].text = "changed"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("PackagedEvidenceItem should be immutable")


if __name__ == "__main__":
    test_functions = [
        value for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for fn in test_functions:
        fn()
    print(f"E6 EVIDENCE PACKAGING HARD TESTS: PASS ({len(test_functions)} tests)")