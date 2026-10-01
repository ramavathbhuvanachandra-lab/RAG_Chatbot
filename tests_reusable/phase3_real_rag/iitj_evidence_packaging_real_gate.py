from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.evidence.claims import (
    Claim,
    ClaimAudit,
    ClaimEvidenceMatch,
    extract_evidence_units,
)
from backend.core.evidence.packaging import build_evidence_package
from backend.core.retrieval_contracts import RetrievalCandidate


CORPUS = Path("data/data_iitj/iitj_rag_v1_docs_production")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _load_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for path in sorted(CORPUS.rglob("*.docx")):
        docs = Docx2txtLoader(str(path)).load()
        chunks.extend(splitter.split_documents(docs))
    return chunks


def _candidate(document):
    return RetrievalCandidate.from_document(
        document,
        source=str(document.metadata.get("source", "")),
    )


def _find(chunks, source_marker: str, *markers: str):
    matches = []
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        content = str(chunk.page_content or "").casefold()
        if source_marker not in source:
            continue
        if markers and not any(marker.casefold() in content for marker in markers):
            continue
        matches.append(chunk)
    if not matches:
        raise AssertionError(f"No real IITJ chunks matched {source_marker} / {markers}")
    return matches


def _audit_for(evidence_id: str, claim_id: str, claim_text: str, *, status: str = "supported"):
    claim = Claim(
        claim_id=claim_id,
        text=claim_text,
        kind="request",
        name="process",
    )
    return ClaimAudit(
        claims=(claim,),
        matches=(ClaimEvidenceMatch(claim_id, evidence_id, 0.9 if status == "supported" else 0.3, status),),
        supported_claim_ids=(claim_id,) if status == "supported" else (),
        partial_claim_ids=(claim_id,) if status == "partial" else (),
        unsupported_claim_ids=(),
        conflicting_claim_ids=(),
    )


def test():
    chunks = _load_chunks()
    sources = {str(c.metadata.get("source", "")) for c in chunks}
    assert len(chunks) > 1000, len(chunks)
    assert len(sources) > 100, len(sources)

    # 1. Real M.Tech eligibility text can be packaged when explicitly audited.
    mtech_chunks = _find(chunks, "/admissions/mtech_admissions.docx", "applicant must have")
    mtech_units = extract_evidence_units([_candidate(mtech_chunks[0])])
    target = next((u for u in mtech_units if "applicant must" in u.text.casefold()), None)
    assert target is not None, [u.text for u in mtech_units]
    mtech_audit = _audit_for(target.evidence_id, "c1", "M.Tech eligibility requirements")
    package = build_evidence_package(mtech_units, claim_audit=mtech_audit)
    assert package.status == "ready", package.to_dict()
    assert "applicant" in package.context.casefold()

    # 2. Real fee evidence survives table/cell-style extraction.
    fee_chunks = _find(chunks, "/finance/fees_and_finance.docx", "tuition fee", "fee structure")
    fee_units = extract_evidence_units([_candidate(fee_chunks[0])])
    money_unit = next((u for u in fee_units if u.factual_markers), None)
    if money_unit is None:
        for chunk in fee_chunks[1:]:
            extra = extract_evidence_units([_candidate(chunk)])
            money_unit = next((u for u in extra if u.factual_markers), None)
            if money_unit is not None:
                fee_units = extra
                break
    assert money_unit is not None, [u.text for u in fee_units]
    fee_audit = _audit_for(money_unit.evidence_id, "c2", "fee amount")
    package = build_evidence_package(fee_units, claim_audit=fee_audit)
    assert package.status == "ready", package.to_dict()
    assert money_unit.text in package.context

    # 3. Unsupported evidence is not allowed through a claim-audited package.
    unrelated = SimpleNamespace(
        page_content="An unrelated admission sentence.",
        metadata={"source": "data/data_iitj/other.docx"},
    )
    unrelated_unit = extract_evidence_units([_candidate(unrelated)])[0]
    mixed = list(mtech_units) + [unrelated_unit]
    package = build_evidence_package(mixed, claim_audit=mtech_audit)
    assert unrelated_unit.evidence_id not in {item.evidence_id for item in package.items}

    # 4. Internal retrieval/debug metadata must not leak into answer context.
    wrapped = SimpleNamespace(
        page_content=(
            "Useful factual sentence.\n"
            "Command 5 • Retrieval Representation\n"
            "Original source urls preserved from the Command 5 knowledge source."
        ),
        metadata={"source": "data/data_iitj/test.docx"},
    )
    wrapped_unit = extract_evidence_units([_candidate(wrapped)])[0]
    package = build_evidence_package([wrapped_unit])
    assert "retrieval representation" not in package.context.casefold()
    assert "command 5" not in package.context.casefold()

    print(f"E6 IITJ REAL-DATA EVIDENCE PACKAGING GATE: PASS (4 tests; {len(chunks)} real chunks; {len(sources)} real sources)")


if __name__ == "__main__":
    test()
