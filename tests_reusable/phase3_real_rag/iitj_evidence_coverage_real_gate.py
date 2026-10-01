"""
E4 — IIT Jodhpur REAL-DATA coverage gate.

IMPORTANT:
- This is NOT a synthetic fixture test.
- Every evidence document is loaded from the active IITJ corpus:
    data/data_iitj/iitj_rag_v1_docs_production/
- The gate FAILS if that real corpus is missing.
- It uses the real DOCX corpus and the active IITJ semantic registry only
  as test-time inputs. No synthetic evidence text is embedded below.
- The production E4 coverage module remains institution-agnostic.

This gate tests evidence coverage/completeness after evidence has been
selected. It is intentionally separate from retrieval/ranking tests.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.evidence.coverage import assess_coverage
from backend.core.query.models import (
    Query,
    ListIntent,
    NumericRequirement,
    TemporalConstraint,
)
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
)
from backend.core.semantic_alignment import align_query_to_document
from backend.core.semantic_registry import SemanticRegistry


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CORPUS = (
    ROOT
    / "data"
    / "data_iitj"
    / "iitj_rag_v1_docs_production"
)

CORPUS_PATH = Path(
    # Explicit override is useful when reproducing against a mounted copy,
    # but no fallback fixture is ever used.
    __import__("os").environ.get(
        "REAL_RAG_CORPUS_PATH",
        str(DEFAULT_CORPUS),
    )
).expanduser().resolve()


def _normal(text: str) -> str:
    text = str(text or "").casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _source(document: Document) -> str:
    value = document.metadata.get("source", "")
    return str(value or "").replace("\\", "/")


def _load_real_chunks() -> list[Document]:
    if not CORPUS_PATH.exists() or not CORPUS_PATH.is_dir():
        raise AssertionError(
            f"REAL IITJ CORPUS NOT FOUND: {CORPUS_PATH}\n"
            "This test refuses to use fixtures or another dataset."
        )

    files = sorted(CORPUS_PATH.rglob("*.docx"))
    if not files:
        raise AssertionError(
            f"REAL IITJ CORPUS HAS NO DOCX FILES: {CORPUS_PATH}"
        )

    documents: list[Document] = []
    for path in files:
        loaded = Docx2txtLoader(str(path)).load()
        documents.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if len(chunks) < 10:
        raise AssertionError(
            f"REAL IITJ CORPUS PRODUCED TOO FEW CHUNKS: {len(chunks)}"
        )

    return chunks


def _load_iitj_registry() -> SemanticRegistry:
    import backend.institutions.iitj.semantic_registry as registry_module

    registries = [
        value
        for value in vars(registry_module).values()
        if isinstance(value, SemanticRegistry)
    ]

    if len(registries) != 1:
        raise AssertionError(
            "Expected exactly one active IITJ SemanticRegistry instance; "
            f"found {len(registries)}."
        )

    return registries[0]


CHUNKS = _load_real_chunks()
REGISTRY = _load_iitj_registry()


def _real_candidates(
    query: str,
    *,
    source_suffixes: Iterable[str] | None = None,
    content_markers: Iterable[str] = (),
    limit: int = 6,
) -> list[RetrievalCandidate]:
    """
    Build E4 candidates from ACTUAL corpus chunks only.

    Source suffixes/content markers select real evidence; they never create
    evidence text.
    """
    suffixes = tuple(
        str(item).replace("\\", "/").casefold()
        for item in (source_suffixes or ())
    )
    markers = tuple(
        _normal(item)
        for item in content_markers
        if _normal(item)
    )

    matches: list[Document] = []
    for chunk in CHUNKS:
        source = _source(chunk).casefold()
        content = _normal(chunk.page_content)

        if suffixes and not any(
            source.endswith(suffix)
            for suffix in suffixes
        ):
            continue

        if markers and not all(
            marker in content
            for marker in markers
        ):
            continue

        matches.append(chunk)

    if not matches:
        raise AssertionError(
            "No REAL IITJ evidence chunk matched the requested source/content "
            f"selector for query: {query!r}"
        )

    candidates: list[RetrievalCandidate] = []
    for rank, document in enumerate(
        matches[:limit],
        start=1,
    ):
        _, meaning, alignment = align_query_to_document(
            query,
            document,
            REGISTRY,
        )

        candidates.append(
            RetrievalCandidate.from_document(
                document,
                meaning=meaning,
                alignment=alignment,
                quality=EvidenceQuality(
                    content_quality=1.0,
                    structural_quality=1.0,
                    noise=0.0,
                ),
                source=_source(document),
                final_score=float(1.0 / rank),
            )
        )

    return candidates


def _lexical_real_candidates(
    query: str,
    *,
    limit: int = 12,
) -> list[RetrievalCandidate]:
    """
    Retrieval-like reference set from the real corpus.

    This is deliberately lightweight: it does not replace ranking. It
    provides a real-corpus stress path for E4 without depending on a vector
    server in the test.
    """
    stop = {
        "what", "which", "where", "when", "how", "who", "why",
        "are", "is", "the", "a", "an", "of", "for", "in", "on",
        "to", "at", "do", "does", "can", "could", "would", "should",
        "tell", "me", "please", "i", "and", "or",
    }
    query_terms = {
        token
        for token in _normal(query).split()
        if len(token) > 2 and token not in stop
    }

    scored: list[tuple[float, Document]] = []
    for chunk in CHUNKS:
        content_terms = {
            token
            for token in _normal(chunk.page_content).split()
            if len(token) > 2
        }
        overlap = len(query_terms & content_terms)
        if overlap <= 0:
            continue

        # Prefer more precise lexical matches, then longer clean evidence.
        score = overlap / max(1, len(query_terms))
        scored.append((score, chunk))

    scored.sort(
        key=lambda item: (-item[0], _source(item[1]), item[1].page_content),
    )

    candidates: list[RetrievalCandidate] = []
    for rank, (_, document) in enumerate(
        scored[:limit],
        start=1,
    ):
        _, meaning, alignment = align_query_to_document(
            query,
            document,
            REGISTRY,
        )
        candidates.append(
            RetrievalCandidate.from_document(
                document,
                meaning=meaning,
                alignment=alignment,
                quality=EvidenceQuality(
                    content_quality=1.0,
                    structural_quality=1.0,
                    noise=0.0,
                ),
                source=_source(document),
                final_score=float(1.0 / rank),
            )
        )

    return candidates


def _assert_status(
    name: str,
    query: Query,
    candidates: list[RetrievalCandidate],
    expected: set[str],
) -> None:
    result = assess_coverage(query, candidates)
    assert result.status in expected, (
        f"{name}: expected {sorted(expected)}, "
        f"got {result.status}. "
        f"sources={result.unique_sources}, "
        f"relevant={result.relevant_documents}, "
        f"strong={result.strong_documents}, "
        f"partial={result.partial_documents}, "
        f"inventory={result.inventory_items}, "
        f"uncovered={result.uncovered_units}, "
        f"reasons={result.reasons}"
    )


def test_real_admission_process():
    query = Query(
        "What is the admission process for IIT Jodhpur?",
        request_type="admission",
    )
    candidates = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/admissions/general_admissions.docx",
        ),
        content_markers=("admission", "application"),
        limit=5,
    )
    _assert_status(
        "real-admission-process",
        query,
        candidates,
        {"supported"},
    )


def test_real_mtech_registration_typo_normalized_question():
    query = Query(
        "Mtech regstration kaise hota hai?",
        request_type="registration",
    )
    candidates = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/admissions/mtech_admissions.docx",
        ),
        content_markers=("mtech", "admission"),
        limit=5,
    )
    _assert_status(
        "real-mtech-registration",
        query,
        candidates,
        {"supported", "partial"},
    )


def test_real_hostel_fee_has_numeric_evidence():
    query = Query(
        "Student hostel accommodation fee kitna hai?",
        request_type="fees",
        numeric_requirements=(
            NumericRequirement(
                "fee",
                1,
                unit="currency",
                confidence=1.0,
            ),
        ),
    )
    candidates = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/finance/fees_and_finance.docx",
        ),
        content_markers=("hostel room rent",),
        limit=5,
    )
    _assert_status(
        "real-hostel-fee",
        query,
        candidates,
        {"supported", "partial"},
    )


def test_real_ee_research_list_has_breadth():
    query = Query(
        "What are the research areas in Electrical Engineering?",
        request_type="research",
        list_intent=ListIntent(
            is_list=True,
            item_type="research",
        ),
    )
    candidates = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/departments/electrical_engineering/research.docx",
        ),
        content_markers=("research themes",),
        limit=6,
    )
    _assert_status(
        "real-ee-research-list",
        query,
        candidates,
        {"supported", "partial"},
    )


def test_real_broad_research_uses_distributed_corpus_evidence():
    query = Query(
        "What research areas are available?",
        request_type="research",
        list_intent=ListIntent(
            is_list=True,
            item_type="research",
        ),
    )
    candidates = _lexical_real_candidates(
        query.original_query,
        limit=12,
    )
    # The test is intentionally permissive on status because coverage
    # depends on what the actual current corpus exposes. It must, however,
    # never claim hard completeness from zero evidence.
    result = assess_coverage(query, candidates)
    assert result.relevant_documents > 0
    assert result.status in {"supported", "partial", "insufficient"}


def test_real_requirements_are_not_satisfied_by_unrelated_admission_fee_chunk():
    query = Query(
        "What are the M.Tech eligibility requirements?",
        request_type="eligibility",
    )
    wrong = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/finance/fees_and_finance.docx",
        ),
        content_markers=("fee structure",),
        limit=4,
    )
    result = assess_coverage(query, wrong)
    assert result.status != "supported", result.to_dict()


def test_real_temporal_fee_requires_requested_year():
    query = Query(
        "What is the fee for 2099?",
        request_type="fees",
        temporal_constraints=(
            TemporalConstraint(
                kind="year",
                value="2099",
                confidence=1.0,
            ),
        ),
    )
    candidates = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/finance/fees_and_finance.docx",
        ),
        content_markers=("fee structure",),
        limit=4,
    )
    result = assess_coverage(query, candidates)
    assert result.status != "supported", result.to_dict()


def test_real_temporal_fee_matches_actual_ay_2026_2027():
    query = Query(
        "What is the fee for AY 2026-2027?",
        request_type="fees",
        temporal_constraints=(
            TemporalConstraint(
                kind="academic_year",
                start="2026",
                end="2027",
                confidence=1.0,
            ),
        ),
    )
    candidates = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/finance/fees_and_finance.docx",
        ),
        content_markers=("fee structure", "2026-2027"),
        limit=4,
    )
    _assert_status(
        "real-ay-2026-2027-fee",
        query,
        candidates,
        {"supported", "partial"},
    )


def test_real_unknown_query_is_not_supported():
    query = Query(
        "What is the zorpulax scholarship process?",
        request_type="scholarship",
    )
    candidates = _lexical_real_candidates(
        query.original_query,
        limit=12,
    )
    result = assess_coverage(query, candidates)
    assert result.status != "supported", result.to_dict()


def test_real_nonsense_query_is_not_supported():
    query = Query(
        "How does the quxvorn moon campus protocol work?",
    )
    candidates = _lexical_real_candidates(
        query.original_query,
        limit=12,
    )
    result = assess_coverage(query, candidates)
    assert result.status != "supported", result.to_dict()


def test_real_multi_part_mtech_query_is_partial_when_fee_is_absent():
    query = Query(
        "Tell me the M.Tech admission eligibility and fee.",
        request_type="admission",
        numeric_requirements=(
            NumericRequirement(
                "fee",
                1,
                unit="currency",
                confidence=1.0,
            ),
        ),
        is_multi_part=True,
    )
    mtech = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/admissions/mtech_admissions.docx",
        ),
        content_markers=("mtech",),
        limit=6,
    )
    result = assess_coverage(query, mtech)
    assert result.status in {"partial", "insufficient"}, result.to_dict()


def test_real_duplicate_chunks_do_not_create_breadth():
    query = Query(
        "What research areas are available?",
        list_intent=ListIntent(
            is_list=True,
            item_type="research",
        ),
    )
    real = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/departments/electrical_engineering/research.docx",
        ),
        content_markers=("research themes",),
        limit=1,
    )
    duplicated = real + real + real
    result = assess_coverage(query, duplicated)
    assert result.inventory_items < 7, result.to_dict()


def test_real_hostel_fee_question_with_registration_fee_evidence_is_not_enough_by_itself():
    query = Query(
        "What is the hostel fee?",
        request_type="fees",
        numeric_requirements=(
            NumericRequirement(
                "fee",
                1,
                unit="currency",
                confidence=1.0,
            ),
        ),
    )
    wrong = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/academic_administration/registration.docx",
        ),
        content_markers=("registration fee",),
        limit=4,
    )
    result = assess_coverage(query, wrong)
    assert result.status != "supported", result.to_dict()


def test_real_hostel_booking_rules_evidence_exists():
    query = Query(
        "How is hostel accommodation booked?",
        request_type="hostel",
    )
    candidates = _real_candidates(
        query.original_query,
        source_suffixes=(
            "/hostel_accommodation/general_information.docx",
        ),
        content_markers=("hostel booking", "online erp"),
        limit=5,
    )
    _assert_status(
        "real-hostel-booking",
        query,
        candidates,
        {"supported", "partial"},
    )


def test_real_corpus_is_substantive():
    sources = {
        _source(chunk)
        for chunk in CHUNKS
        if _source(chunk)
    }
    assert len(sources) >= 10, (
        f"Real IITJ corpus is unexpectedly small: {len(sources)} sources"
    )


if __name__ == "__main__":
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]

    for test in tests:
        test()

    print(
        "E4 IITJ REAL-DATA HARD GATE: PASS "
        f"({len(tests)} tests; {len(CHUNKS)} real chunks; "
        f"{len({_source(c) for c in CHUNKS if _source(c)})} real sources)"
    )
