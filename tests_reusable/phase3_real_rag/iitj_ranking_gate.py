"""Real IIT Jodhpur corpus gate for the production ranking module.

This file is intentionally IITJ-specific. It belongs under tests_reusable/ only;
no IITJ names or filenames are imported by the production ranker.

Run from the repository root after replacing the ranker:

    PYTHONPATH=. python tests_reusable/phase3_real_rag/iitj_ranking_gate.py

No answer LLM and no LangGraph are used. This isolates retrieval + RRF + reranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.retrieval_contracts import RetrievalCandidate
from backend.core.rrf import fuse_ranked_lists
from backend.core.retrieval.ranking import rank_candidates


@dataclass(frozen=True)
class Case:
    question: str
    expected_source: str
    required_markers: tuple[str, ...] = ()
    forbidden_markers: tuple[str, ...] = ()


CASES = (
    Case(
        "What is the admission process for IIT Jodhpur?",
        "admissions/general_admissions.docx",
        ("admission", "process"),
    ),
    Case(
        "What documents are required during admission?",
        "academic_administration/registration.docx",
        ("certificate",),
    ),
    Case(
        "How can I contact the admissions office?",
        "programs/phd/general_information.docx",
        ("contact",),
    ),
    Case(
        "What is the hostel fee for students?",
        "finance/fees_and_finance.docx",
        ("student", "semester fee"),
        ("short-term booking charges",),
    ),
    Case(
        "What minor programs are available at IIT Jodhpur?",
        "schools/management_and_entrepreneurship/programs.docx",
        ("minor programs",),
    ),
    Case(
        "What are the research areas offered by the Electrical Engineering department?",
        "departments/electrical_engineering/research.docx",
        ("research areas",),
    ),
    Case(
        "What food and dining facilities are available on campus?",
        "hostel_accommodation/general_information.docx",
        ("dining",),
    ),
    Case(
        "What emergency medical facilities are available at IIT Jodhpur?",
        "research_and_technology_facilities/facilities.docx",
        ("health centre",),
        ("knowledge_buffer_general.docx",),
    ),
    Case(
        "hostel mein students ko kya rules follow karne hote hain?",
        "hostel_accommodation/general_information.docx",
        ("hostel",),
    ),
    Case(
        "Mtech regstration kaise hota hai?",
        "admissions/mtech_admissions.docx",
        ("m tech",),
    ),
    Case(
        "What are the short-term hostel booking charges?",
        "finance/fees_and_finance.docx",
        ("accommodation charges", "short-term bookings"),
    ),
)


def _source(document) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return str(metadata.get("source", "") or "")
    return ""


def _preview(document, limit: int = 220) -> str:
    text = " ".join(str(getattr(document, "page_content", "") or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _same_expected_source(actual: str, expected: str) -> bool:
    return _normalize_path(expected) in _normalize_path(actual)


def _contains_required(document, markers: tuple[str, ...]) -> bool:
    text = " ".join(str(getattr(document, "page_content", "") or "").split()).casefold()
    return all(marker.casefold() in text for marker in markers)


def _contains_forbidden(document, markers: tuple[str, ...]) -> bool:
    text = " ".join(str(getattr(document, "page_content", "") or "").split()).casefold()
    return any(marker.casefold() in text for marker in markers)


def run_case(case: Case) -> tuple[bool, str]:
    import backend.retriever as legacy

    dense = list(legacy.dense_retrieve(case.question))
    bm25 = list(legacy.keyword_retrieve(case.question))
    fused = list(fuse_ranked_lists((dense, bm25), primary_query=case.question))

    ranked = rank_candidates(
        case.question,
        [
            RetrievalCandidate.from_document(
                item.document,
                source=_source(item.document) or "unknown",
                provenance=item.provenance,
            )
            for item in fused
        ],
        top_k=5,
    )

    if not ranked:
        return False, "no ranked candidates"

    top = ranked[0].document
    actual_source = _source(top)

    failures: list[str] = []
    if not _same_expected_source(actual_source, case.expected_source):
        failures.append(f"top1 source expected *{case.expected_source}*, got {actual_source}")
    if not _contains_required(top, case.required_markers):
        failures.append(f"top1 missing required content markers {case.required_markers}")
    if _contains_forbidden(top, case.forbidden_markers):
        failures.append(f"top1 contains forbidden content marker(s) {case.forbidden_markers}")

    normalized_contents = [
        " ".join(str(getattr(item.document, "page_content", "") or "").split()).casefold()
        for item in ranked
    ]
    if len(normalized_contents) != len(set(normalized_contents)):
        failures.append("exact duplicate content leaked into top-5")

    detail = (
        f"top1={actual_source}\n"
        f"top1_preview={_preview(top)}\n"
        f"top5_sources={[ _source(item.document) for item in ranked ]}"
    )
    if failures:
        return False, " | ".join(failures) + "\n" + detail
    return True, detail


def main() -> int:
    passed = 0
    print("IIT JODHPUR REAL RANKING GATE")
    print("=" * 110)

    for index, case in enumerate(CASES, start=1):
        print(f"\n[{index}/{len(CASES)}] {case.question}")
        ok, detail = run_case(case)
        print("PASS" if ok else "FAIL")
        print(detail)
        if ok:
            passed += 1

    print("\n" + "=" * 110)
    print(f"IITJ REAL RANKING GATE: {passed}/{len(CASES)} PASS")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
