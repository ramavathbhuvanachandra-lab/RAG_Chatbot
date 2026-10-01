"""Hard deterministic tests for generic evidence grouping/context expansion."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.core.evidence.grouping import (
    build_evidence_groups,
    expand_group_context,
    flatten_evidence_groups,
)


@dataclass
class Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def d(text: str, source: str, index: int | None = None) -> Doc:
    metadata = {"source": source}
    if index is not None:
        metadata["chunk_index"] = index
    return Doc(text, metadata)


def test_one_group_per_unique_anchor():
    anchors = [
        d("Admission requirements include a valid degree.", "a.docx", 0),
        d("Admission requirements include a valid degree.", "a.docx", 0),
        d("Hostel rules apply to residents.", "b.docx", 0),
    ]
    groups = build_evidence_groups(anchors)
    assert len(groups) == 2
    assert groups[0].documents[0] is anchors[0]


def test_previous_and_next_same_source_can_be_attached():
    chunks = [
        d("The program admits qualified students under the published criteria.", "a.docx", 0),
        d("The admission process begins with online application submission.", "a.docx", 1),
        d("Applicants must then submit the required certificates and score cards.", "a.docx", 2),
    ]
    groups = build_evidence_groups([chunks[1]])
    expanded = expand_group_context(groups, chunks, min_context_score=0.0)
    assert len(expanded[0].documents) == 3


def test_cross_source_neighbor_is_never_attached():
    anchor = d("Admission process starts with online application.", "a.docx", 0)
    other = d("Admission process starts with a different form.", "b.docx", 0)
    expanded = expand_group_context(build_evidence_groups([anchor]), [anchor, other], min_context_score=0.0)
    assert len(expanded[0].documents) == 1


def test_structural_heading_stops_neighbor_inclusion():
    anchor = d("The hostel provides accommodation and dining facilities for students.", "a.docx", 0)
    heading = d("20.4 Reservation Policy", "a.docx", 1)
    expanded = expand_group_context(build_evidence_groups([anchor]), [anchor, heading], min_context_score=0.0)
    assert len(expanded[0].documents) == 1


def test_navigation_noise_is_rejected():
    anchor = d("The health centre provides medical support to students and staff.", "a.docx", 0)
    noise = d("Original source URLs http://a.example http://b.example http://c.example retrieval rank 4", "a.docx", 1)
    expanded = expand_group_context(build_evidence_groups([anchor]), [anchor, noise], min_context_score=0.0)
    assert len(expanded[0].documents) == 1


def test_anchor_is_always_preserved():
    anchor = d("Important eligibility requirement for the program.", "a.docx", 0)
    expanded = expand_group_context(build_evidence_groups([anchor]), [anchor])
    assert expanded[0].documents[0] is anchor


def test_group_context_budget_is_bounded():
    chunks = [
        d("The department offers a program and the admission process starts online.", "a.docx", 0),
        d("Applicants submit forms and supporting documents.", "a.docx", 1),
        d("Candidates then complete verification.", "a.docx", 2),
        d("Students receive final confirmation.", "a.docx", 3),
    ]
    expanded = expand_group_context(
        build_evidence_groups([chunks[1]]),
        chunks,
        min_context_score=0.0,
        max_previous=2,
        max_next=2,
        max_context_per_group=2,
    )
    assert len(expanded[0].documents) <= 3


def test_flatten_preserves_group_order_and_deduplicates():
    first = d("First evidence text with enough meaningful words here.", "a.docx", 0)
    duplicate = d("First evidence text with enough meaningful words here.", "a.docx", 1)
    second = d("Second evidence text with enough meaningful words here.", "b.docx", 0)
    groups = build_evidence_groups([first, duplicate, second])
    expanded = expand_group_context(groups, [first, duplicate, second], min_context_score=0.0)
    flat = flatten_evidence_groups(expanded)
    keys = [(x.metadata.get("source"), x.page_content) for x in flat]
    assert len(keys) == len(set(keys))
    assert flat[0] is first
    assert flat[-1] is second


def test_unlocatable_anchor_does_not_crash_or_invent_context():
    anchor = d("Evidence exists but this anchor is not in canonical chunks.", "missing.docx")
    other = d("Unrelated neighboring content from another source.", "other.docx", 0)
    expanded = expand_group_context(build_evidence_groups([anchor]), [other])
    assert len(expanded[0].documents) == 1
    assert expanded[0].documents[0] is anchor


def test_content_is_not_mutated_when_neighbor_is_added():
    anchor = d("Admission applications are submitted online,.", "a.docx", 0)
    neighbor = d("The applicant must upload certificates and proof of identity.", "a.docx", 1)
    original = anchor.page_content
    expanded = expand_group_context(build_evidence_groups([anchor]), [anchor, neighbor], min_context_score=0.0)
    assert anchor.page_content == original
    assert expanded[0].documents[0].page_content == original


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"E2 EVIDENCE GROUPING HARD TESTS: PASS ({len(tests)} tests)")