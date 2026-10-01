"""E5 real-data claim/evidence gate using the active IITJ corpus.

The production claims module remains institution-agnostic. IITJ-specific
source selection exists only in this test gate so real corpus behavior is
verifiable.
"""
from __future__ import annotations

from pathlib import Path
import os
import re

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.evidence.claims import (
    audit_claims,
    extract_claims,
    extract_evidence_units,
    match_claim_to_evidence,
    split_evidence_units,
)
from backend.core.query.models import NumericRequirement, Query, Target, TemporalConstraint
from backend.core.retrieval_contracts import CandidateAlignment, EvidenceQuality, RetrievalCandidate

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(os.environ.get(
    "REAL_RAG_CORPUS_PATH",
    str(ROOT / "data" / "data_iitj" / "iitj_rag_v1_docs_production"),
)).expanduser().resolve()


def source(d: Document) -> str:
    return str(d.metadata.get("source", "")).replace("\\", "/")


def normal(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "").casefold()).strip()


def load_chunks() -> list[Document]:
    if not CORPUS.is_dir():
        raise AssertionError(f"REAL IITJ CORPUS NOT FOUND: {CORPUS}")
    files = sorted(CORPUS.rglob("*.docx"))
    if not files:
        raise AssertionError(f"REAL IITJ CORPUS HAS NO DOCX FILES: {CORPUS}")
    docs: list[Document] = []
    for path in files:
        docs.extend(Docx2txtLoader(str(path)).load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if len(chunks) < 100:
        raise AssertionError(f"REAL IITJ CORPUS TOO SMALL: {len(chunks)} chunks")
    return chunks


CHUNKS = load_chunks()


def real_candidates(*, suffix: str, markers: tuple[str, ...], limit: int = 6) -> list[RetrievalCandidate]:
    """Select real corpus chunks using flexible evidence cues.

    The marker list is a test-time retrieval cue, not a requirement that all
    phrases survive inside one DOCX chunk. Real chunking may separate related
    headings, qualifications, and values. Chunks are therefore ranked by how
    many requested cues they contain.
    """
    normalized_markers = tuple(
        normal(marker) for marker in markers if normal(marker)
    )

    scored: list[tuple[int, int, Document]] = []
    for chunk in CHUNKS:
        src = source(chunk).casefold()
        content = normal(chunk.page_content)
        if not src.endswith(suffix.casefold()):
            continue

        cue_count = sum(
            1 for marker in normalized_markers if marker in content
        )
        if normalized_markers and cue_count == 0:
            continue

        # Prefer more cue-bearing chunks, then richer chunks, deterministically.
        scored.append((cue_count, len(chunk.page_content), chunk))

    if not scored:
        raise AssertionError(
            f"No real IITJ chunks matched {suffix} / {markers}"
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            source(item[2]).casefold(),
            item[2].page_content.casefold(),
        )
    )

    candidates: list[RetrievalCandidate] = []
    for rank, (_, _, chunk) in enumerate(scored[:limit], 1):
        candidates.append(
            RetrievalCandidate.from_document(
                chunk,
                source=source(chunk),
                alignment=CandidateAlignment(),
                quality=EvidenceQuality(
                    content_quality=1.0,
                    structural_quality=1.0,
                    noise=0.0,
                ),
                final_score=1.0 / rank,
            )
        )
    return candidates


def test_real_mtech_evidence_contains_requirement_claim():
    q = Query("What are the M.Tech eligibility requirements?", request_type="eligibility")
    candidates = real_candidates(suffix="/admissions/mtech_admissions.docx", markers=("eligibility", "degree"))
    result = audit_claims(extract_claims(q), extract_evidence_units(candidates))
    assert result.status in {"supported", "partial"}, result.to_dict()


def test_real_mtech_evidence_cannot_satisfy_fee_claim():
    q = Query(
        "What is the M.Tech fee?",
        target=Target("M.Tech", confidence=1.0, resolution_state="resolved"),
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    candidates = real_candidates(suffix="/admissions/mtech_admissions.docx", markers=("mtech",))
    audit = audit_claims(extract_claims(q), extract_evidence_units(candidates))
    numeric = next(c for c in audit.claims if c.kind == "numeric")
    assert numeric.claim_id in audit.unsupported_claim_ids


def test_real_fee_evidence_satisfies_currency_claim():
    q = Query(
        "What is the fee?",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    candidates = real_candidates(suffix="/finance/fees_and_finance.docx", markers=("fee structure",), limit=4)
    audit = audit_claims(extract_claims(q), extract_evidence_units(candidates))
    numeric = next(c for c in audit.claims if c.kind == "numeric")
    assert numeric.claim_id in audit.supported_claim_ids, audit.to_dict()


def test_real_registration_fee_does_not_satisfy_hostel_fee_context():
    q = Query(
        "What is the hostel fee?",
        target=Target("hostel", confidence=1.0, resolution_state="resolved"),
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    candidates = real_candidates(suffix="/academic_administration/registration.docx", markers=("registration fee",))
    audit = audit_claims(extract_claims(q), extract_evidence_units(candidates))
    numeric = next(c for c in audit.claims if c.kind == "numeric")
    assert numeric.claim_id in audit.unsupported_claim_ids, audit.to_dict()


def test_real_2099_temporal_claim_is_unsupported():
    q = Query(
        "What is the fee for 2099?",
        temporal_constraints=(TemporalConstraint(kind="year", value="2099", confidence=1.0),),
    )
    candidates = real_candidates(suffix="/finance/fees_and_finance.docx", markers=("fee structure",))
    audit = audit_claims(extract_claims(q), extract_evidence_units(candidates))
    temporal = next(c for c in audit.claims if c.kind == "temporal")
    assert temporal.claim_id in audit.unsupported_claim_ids, audit.to_dict()


def test_real_ay_2026_2027_temporal_claim_is_supported():
    q = Query(
        "What is the fee for AY 2026-2027?",
        temporal_constraints=(TemporalConstraint(kind="academic_year", start="2026", end="2027", confidence=1.0),),
    )
    candidates = real_candidates(suffix="/finance/fees_and_finance.docx", markers=("2026-2027",))
    audit = audit_claims(extract_claims(q), extract_evidence_units(candidates))
    temporal = next(c for c in audit.claims if c.kind == "temporal")
    assert temporal.claim_id in audit.supported_claim_ids, audit.to_dict()


def test_real_research_units_are_segmented_without_institution_logic():
    candidates = real_candidates(suffix="/departments/electrical_engineering/research.docx", markers=("research themes",), limit=1)
    units = split_evidence_units(candidates[0])
    assert units
    assert any("research" in unit.text.casefold() for unit in units)


def test_real_unknown_claim_does_not_match_unrelated_content():
    q = Query("What is the zorpulax scholarship process?", request_type="scholarship")
    candidates = real_candidates(suffix="/finance/fees_and_finance.docx", markers=("fee structure",), limit=3)
    audit = audit_claims(extract_claims(q), extract_evidence_units(candidates))
    assert audit.status != "supported", audit.to_dict()


def test_real_duplicate_chunks_do_not_create_duplicate_evidence_units():
    candidates = real_candidates(suffix="/finance/fees_and_finance.docx", markers=("fee structure",), limit=1)
    units = extract_evidence_units(candidates * 4)
    assert len(units) == len(extract_evidence_units(candidates))


def test_real_claim_match_preserves_source_traceability():
    q = Query(
        "What is the fee?",
        numeric_requirements=(NumericRequirement("fee", 1, unit="currency", confidence=1.0),),
    )
    candidates = real_candidates(suffix="/finance/fees_and_finance.docx", markers=("fee structure",), limit=1)
    units = extract_evidence_units(candidates)
    claim = next(c for c in extract_claims(q) if c.kind == "numeric")
    matched = [match_claim_to_evidence(claim, unit) for unit in units]
    supported = [m for m in matched if m.status == "supported"]
    assert supported
    evidence_id = supported[0].evidence_id
    assert any(unit.evidence_id == evidence_id and unit.source for unit in units)


if __name__ == "__main__":
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
    sources = {source(c) for c in CHUNKS if source(c)}
    print(f"E5 IITJ REAL-DATA CLAIM GATE: PASS ({len(tests)} tests; {len(CHUNKS)} real chunks; {len(sources)} real sources)")