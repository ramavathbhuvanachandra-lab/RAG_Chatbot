"""Real-corpus Stage-2 integration gate.

This test is intentionally institution-specific only at the fixture layer.
The production verifier remains institution-agnostic.
It loads the existing IIT Jodhpur knowledge documents and exercises 15 factual
questions plus adversarial cross-document checks.
"""

from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace

import pytest
from docx import Document

from ai_platform.core.query.models import (
    Entity,
    QueryFacet,
    RetrievalRequirement,
    SemanticQueryFrame,
)
from ai_platform.core.retrieval.contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
)
from ai_platform.core.retrieval.verification import verify_candidate

ROOT = Path(__file__).resolve().parents[3]
CORPUS_ROOT = Path(
    os.environ.get(
        "RAG_CORPUS_ROOT",
        str(ROOT / "data" / "data_iitj" / "iitj_rag_v1_docs_production"),
    )
)


def _resolve_doc(filename: str) -> Path:
    direct = CORPUS_ROOT / filename
    if direct.is_file():
        return direct
    if CORPUS_ROOT.exists():
        matches = tuple(CORPUS_ROOT.rglob(filename))
        if matches:
            return matches[0]
        stem = Path(filename).stem
        prefix_matches = tuple(CORPUS_ROOT.rglob(stem + "*.docx"))
        if prefix_matches:
            return prefix_matches[0]
    # Also permit the four fixture files to live directly under RAG_CORPUS_ROOT
    # in a temporary test environment.
    fallback = Path(filename)
    if fallback.is_file():
        return fallback
    return direct


REAL_DOCS = {
    "mtech": _resolve_doc("mtech_admissions.docx"),
    "msc": _resolve_doc("msc_admissions.docx"),
    "general": _resolve_doc("general_admissions.docx"),
    "registration": _resolve_doc("registration.docx"),
}


def _load_text(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"Real corpus fixture missing: {path}")
    return "\n".join(
        paragraph.text
        for paragraph in Document(path).paragraphs
        if paragraph.text.strip()
    )


def _frame(
    question: str,
    target: str,
    attribute: str,
    request_type: str,
    *,
    aliases: tuple[str, ...] = (),
) -> SemanticQueryFrame:
    entity = Entity(
        name=target,
        entity_type="program",
        entity_id=target.casefold().replace(" ", "_"),
        aliases=aliases,
        confidence=1.0,
        resolution_state="resolved",
    )
    return SemanticQueryFrame(
        original_query=question,
        normalized_query=question,
        semantic_query=f"{target} {attribute}",
        target=target,
        request_type=request_type,
        facets=(QueryFacet("requested_attribute", attribute),),
        entities=(entity,),
        requirement=RetrievalRequirement(
            mode="exact",
            require_target_alignment=True,
            require_attribute_alignment=True,
            reject_explicit_conflict=True,
        ),
    )


def _candidate(
    text: str,
    source: Path,
    *,
    semantic_match: float = 0.95,
    conflicts: tuple[str, ...] = (),
) -> RetrievalCandidate:
    document = SimpleNamespace(
        page_content=text,
        metadata={"source": str(source)},
    )
    return RetrievalCandidate(
        document=document,
        document_id=f"real::{source.name}",
        source=str(source),
        meaning=DocumentMeaning(),
        alignment=CandidateAlignment(
            semantic_match=semantic_match,
            coverage=0.95,
            attribute_match=0.95,
            scope_match=1.0,
            conflicts=conflicts,
        ),
        quality=EvidenceQuality(),
    )


POSITIVE_CASES = [
    (
        "mtech_eligibility",
        _frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",)),
        "mtech",
    ),
    (
        "mtech_percentage",
        _frame("What percentage or CGPA is required for M.Tech admission?", "M.Tech", "percentage or CGPA", "eligibility", aliases=("m tech",)),
        "mtech",
    ),
    (
        "mtech_gate",
        _frame("Is GATE required for M.Tech admission?", "M.Tech", "GATE", "admission", aliases=("m tech",)),
        "mtech",
    ),
    (
        "mtech_qualification",
        _frame("Which bachelor's degree qualifications are accepted for M.Tech admission?", "M.Tech", "bachelor's degree", "eligibility", aliases=("m tech",)),
        "mtech",
    ),
    (
        "mtech_written_test",
        _frame("Is there a written test for M.Tech admission?", "M.Tech", "written test", "admission", aliases=("m tech",)),
        "mtech",
    ),
    (
        "mtech_work_experience",
        _frame("What work experience is required for Executive, Part-time, External or Sponsored M.Tech?", "Executive/Part-time/External/Sponsored M.Tech", "work experience", "eligibility", aliases=("executive m tech", "part time m tech", "m tech")),
        "mtech",
    ),
    (
        "msc_eligibility",
        _frame("What are the eligibility requirements for M.Sc admission?", "M.Sc", "eligibility", "eligibility", aliases=("m sc",)),
        "msc",
    ),
    (
        "msc_percentage",
        _frame("What percentage or CGPA is needed for M.Sc admission?", "M.Sc", "60% or CGPA", "eligibility", aliases=("m sc",)),
        "msc",
    ),
    (
        "msc_jam",
        _frame("Is JAM required for M.Sc admission?", "M.Sc", "JAM", "admission", aliases=("m sc",)),
        "msc",
    ),
    (
        "msc_bachelor_route",
        _frame("Can I apply for M.Sc through a bachelor's degree route with a test or interview?", "M.Sc", "bachelor's degree route with written test or interview", "admission", aliases=("m sc",)),
        "msc",
    ),
    (
        "phd_eligibility",
        _frame("What are the eligibility requirements for Ph.D. admission at the School of AI and Data Science?", "Ph.D.", "eligibility", "eligibility", aliases=("ph d",)),
        "general",
    ),
    (
        "phd_application_fee",
        _frame("What is the Ph.D. application processing fee?", "Ph.D.", "processing fee", "cost", aliases=("ph d",)),
        "general",
    ),
    (
        "mba_steps",
        _frame("What are the steps for MBA admission?", "MBA", "steps", "procedure"),
        "general",
    ),
    (
        "registration_documents",
        _frame("What documents should students bring during registration?", "registration", "documents to bring", "registration"),
        "registration",
    ),
    (
        "mtech_summer_registration",
        _frame("What happens if an M.Tech student wants to register for summer courses?", "M.Tech", "summer course registration", "registration", aliases=("m tech",)),
        "registration",
    ),
]


@pytest.mark.parametrize("case_id, query_frame, doc_key", POSITIVE_CASES)
def test_real_iitj_fifteen_cases_verify(case_id, query_frame, doc_key):
    text = _load_text(REAL_DOCS[doc_key])
    result = verify_candidate(
        query_frame,
        _candidate(text, REAL_DOCS[doc_key]),
    )
    assert result.status == "verified", (case_id, result.to_dict())
    assert result.accepted is True
    assert result.evidence is not None
    assert result.evidence.relation_score >= 0.70


def test_real_mtech_query_rejects_msc_document_with_no_scope_metadata():
    # The verifier should not invent a relation when target evidence is absent.
    # The M.Sc document contains no M.Tech factual target, so this is a clean
    # target-grounding negative without relying on institution-specific scope rules.
    text = _load_text(REAL_DOCS["msc"])
    q = _frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",))
    result = verify_candidate(q, _candidate(text, REAL_DOCS["msc"], semantic_match=0.99))
    assert result.status == "rejected"
    assert "required_target_not_grounded" in result.reasons


def test_real_phd_query_rejects_mtech_document_with_no_phd_target():
    text = _load_text(REAL_DOCS["mtech"])
    q = _frame("What are the Ph.D. eligibility requirements?", "Ph.D.", "eligibility", "eligibility", aliases=("ph d",))
    result = verify_candidate(q, _candidate(text, REAL_DOCS["mtech"], semantic_match=0.99))
    assert result.status == "rejected"
    assert "required_target_not_grounded" in result.reasons


def test_real_scope_conflict_remains_hard_rejection():
    text = _load_text(REAL_DOCS["mtech"])
    q = _frame("What are the M.Tech eligibility requirements?", "M.Tech", "eligibility", "eligibility", aliases=("m tech",))
    result = verify_candidate(
        q,
        _candidate(
            text,
            REAL_DOCS["mtech"],
            semantic_match=0.99,
            conflicts=("program_scope",),
        ),
    )
    assert result.status == "rejected"
    assert result.conflict_detected is True
