"""Real-corpus adversarial gate for E7.2 answer grounding.

The gate loads the same IITJ production DOCX corpus the project uses, chunks it
with the established 800/150 settings for corpus statistics, and then runs all
assertions against text from the real source files.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.answering.grounding import assess_answer_grounding


CORPUS = Path("data/data_iitj/iitj_rag_v1_docs_production")


def load_real_sources() -> tuple[dict[str, str], int, int]:
    if not CORPUS.exists():
        raise AssertionError(f"Missing real IITJ corpus: {CORPUS}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    sources: dict[str, str] = {}
    chunk_count = 0

    for path in CORPUS.rglob("*.docx"):
        try:
            docs = Docx2txtLoader(str(path)).load()
        except Exception:
            continue
        combined = "\n".join(str(doc.page_content or "") for doc in docs).strip()
        if not combined:
            continue
        relative = str(path.relative_to(CORPUS)).replace("\\", "/")
        sources[relative] = combined
        chunk_count += len(splitter.split_text(combined))

    if not sources:
        raise AssertionError("No real IITJ DOCX sources could be loaded.")

    return sources, chunk_count, len(sources)


def source_text(sources: dict[str, str], suffix: str) -> str:
    matches = [text for path, text in sources.items() if path.endswith(suffix)]
    if not matches:
        raise AssertionError(f"Real source not found: {suffix}")
    return max(matches, key=len)


def test() -> None:
    sources, real_chunks, real_sources = load_real_sources()

    mtech = source_text(sources, "admissions/mtech_admissions.docx")
    finance = source_text(sources, "finance/fees_and_finance.docx")

    checks = []

    r = assess_answer_grounding(
        answer="Applicants must have the required qualifying degree.",
        evidence=mtech,
    )
    checks.append(("real_mtech_policy_supported", r.status == "grounded", r.to_dict()))

    r = assess_answer_grounding(
        answer="Based on that, you are eligible for M.Tech admission.",
        evidence=mtech,
    )
    checks.append(("real_mtech_personal_eligibility_rejected", r.status == "review", r.to_dict()))

    r = assess_answer_grounding(
        answer="The fee is ₹50,000.",
        evidence=finance,
    )
    checks.append(("real_finance_existing_amount_supported", r.status == "grounded", r.to_dict()))

    r = assess_answer_grounding(
        answer="The fee is ₹99,999.",
        evidence=finance,
    )
    checks.append(("real_finance_fabricated_amount_rejected", r.status == "review", r.to_dict()))

    r = assess_answer_grounding(
        answer="Your exact total cost is ₹50,000.",
        evidence=finance,
    )
    checks.append(("real_finance_personal_total_rejected", r.status == "review", r.to_dict()))

    r = assess_answer_grounding(
        answer="This fee applies in 2099.",
        evidence=finance,
    )
    checks.append(("real_wrong_year_rejected", r.status == "review", r.to_dict()))

    r = assess_answer_grounding(
        answer="The hostel room rent amount is ₹500 per day.",
        evidence=finance,
    )
    checks.append(("real_hostel_rate_statement_supported", r.status == "grounded", r.to_dict()))

    r = assess_answer_grounding(
        answer="Your exact total hostel cost is ₹500.",
        evidence=finance,
    )
    checks.append(("real_hostel_personal_total_rejected", r.status == "review", r.to_dict()))

    failed = [item for item in checks if not item[1]]
    if failed:
        for name, _, detail in failed:
            print(f"FAIL: {name}: {detail}")
        raise AssertionError(f"{len(failed)} real grounding checks failed")

    print(
        "E7.2 IITJ REAL-DATA GROUNDING RED-TEAM GATE: PASS "
        f"({len(checks)} tests; {real_chunks} real chunks; {real_sources} real sources)"
    )


if __name__ == "__main__":
    test()
