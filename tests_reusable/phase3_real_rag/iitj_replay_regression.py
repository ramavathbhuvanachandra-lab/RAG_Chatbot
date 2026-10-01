"""Replay the actual IITJ evidence snippets observed in the real ranking run.

This is a local regression replay, not a substitute for running against the
full IITJ vector store. The snippets are taken from the supplied real-corpus
probe output so changes can be validated before the Mac-side corpus gate.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.retrieval_contracts import RetrievalCandidate, RetrievalProvenance
from backend.core.retrieval.ranking import rank_candidates


class Doc:
    def __init__(self, content: str, source: str) -> None:
        self.page_content = content
        self.metadata = {"source": source}


def C(content: str, source: str, rrf: float) -> RetrievalCandidate:
    return RetrievalCandidate.from_document(
        Doc(content, source),
        source=source,
        provenance=RetrievalProvenance(rrf_score=rrf, primary_query=""),
    )


BASE = "data/data_iitj/iitj_rag_v1_docs_production/"

CASES = [
    (
        "What is the admission process for IIT Jodhpur?",
        f"{BASE}admissions/general_admissions.docx",
        [
            C("IIT Jodhpur Admissions General Admissions 20.3 Admission of Non-Degree Students", f"{BASE}admissions/general_admissions.docx", .020),
            C("IIT Jodhpur Admissions Msc Admissions. Admission to M.Sc. Programs offered by the Institute will be through avenues.", f"{BASE}admissions/msc_admissions.docx", .030),
            C("International applicants: eligibility and admission process.", f"{BASE}schools/management_and_entrepreneurship/international_relations.docx", .025),
            C("Admission to M Tech Program. Applicants must satisfy eligibility conditions.", f"{BASE}admissions/mtech_admissions.docx", .018),
        ],
    ),
    (
        "What documents are required during admission?",
        f"{BASE}academic_administration/registration.docx",
        [
            C("Registration & Academic Session. What items need to be brought and what are the procedures? Basically, you have to bring originals and copies of all the necessary certificates, JEE Score cards, proof of address.", f"{BASE}academic_administration/registration.docx", .024),
            C("Fulfillment of Admission Requirements. Admission to any undergraduate program requires that the applicant be eligible.", f"{BASE}admissions/general_admissions.docx", .030),
            C("Students admitted provisionally shall submit copies of their mark sheets, provisional certificates, etc. and other documents by the last date.", f"{BASE}admissions/general_admissions.docx", .029),
            C("Applicants are requested to fill and submit the application form online and pay the processing fee.", f"{BASE}admissions/general_admissions.docx", .022),
        ],
    ),
    (
        "How can I contact the admissions office?",
        f"{BASE}programs/phd/general_information.docx",
        [
            C("In case of any query with respect to online application, applicants may contact the Office of Automation (Academics) by email. Other general queries may be directed to the PhD Program Coordinator.", f"{BASE}programs/phd/general_information.docx", .030),
            C("When can I apply for a particular program? Applications follow the academic cycle.", f"{BASE}admissions/general_admissions.docx", .028),
            C("Postgraduate admissions. The process of admission for programs offered by the Institute normally is underway during April-May.", f"{BASE}admissions/general_admissions.docx", .026),
        ],
    ),
    (
        "What is the hostel fee for students?",
        f"{BASE}finance/fees_and_finance.docx",
        [
            C("₹4,000/- D. Semester Fee ₹32,250/- E. Refundable Deposit ₹8,000/-. The semester fee includes the hostel fees.", f"{BASE}finance/fees_and_finance.docx", .030),
            C("For exceeding 11 days but less than a month: Rs. 2175 excluding GST. The above rates are also applicable to visitors and short stays.", f"{BASE}finance/fees_and_finance.docx", .029),
            C("Program fee structure: tuition fee, semester fee, admission fee and convocation fee.", f"{BASE}schools/management_and_entrepreneurship/general_information.docx", .027),
            C("Hostel Booking. Hostel accommodation is managed through an online ERP-based booking and allotment system.", f"{BASE}hostel_accommodation/general_information.docx", .024),
        ],
    ),
    (
        "What minor programs are available at IIT Jodhpur?",
        f"{BASE}schools/management_and_entrepreneurship/programs.docx",
        [
            C("Minor Programs. Undergraduate students are offered minor programs to complement their Majors.", f"{BASE}schools/management_and_entrepreneurship/programs.docx", .030),
            C("Minor Programs. Undergraduate students are offered minor programs to complement their Majors.", f"{BASE}schools/management_and_entrepreneurship/programs.docx", .029),
            C("Undergraduate Programs. IIT Jodhpur currently offers Bachelor of Technology programs and capability linked opportunities.", f"{BASE}offices_and_administration/office_of_academics/programs.docx", .027),
            C("Dining Options. Separate vegetarian and non-vegetarian messes are available on campus.", f"{BASE}hostel_accommodation/general_information.docx", .024),
        ],
    ),
    (
        "What are the research areas offered by the Electrical Engineering department?",
        f"{BASE}departments/electrical_engineering/research.docx",
        [
            C("Sociodigital Reality, High-speed VLSI Systems and EDA tools, Signal Integrity, Neuromorphic Computing. Research areas pursued by faculty members of the Department of Electrical Engineering.", f"{BASE}departments/electrical_engineering/research.docx", .030),
            C("Research overview and research areas mapped to Technology Tracks of the Department.", f"{BASE}departments/electrical_engineering/research.docx", .029),
            C("Electrical Engineering overview with collaborations and research themes.", f"{BASE}departments/electrical_engineering/overview.docx", .026),
            C("Research activities organised around broad thematic areas in Economics.", f"{BASE}departments/economics/research.docx", .028),
        ],
    ),
    (
        "What food and dining facilities are available on campus?",
        f"{BASE}hostel_accommodation/general_information.docx",
        [
            C("Dining Options. Separate dining facilities (vegetarian and non-vegetarian messes) are available on campus. Meal charges are paid directly to the mess vendor.", f"{BASE}hostel_accommodation/general_information.docx", .030),
            C("Dining Options. Separate dining facilities (vegetarian and non-vegetarian messes) are available on campus. Meal charges are paid directly to the mess vendor.", f"{BASE}hostel_accommodation/general_information.docx", .029),
            C("Program fee structure and semester fee information.", f"{BASE}schools/management_and_entrepreneurship/general_information.docx", .028),
            C("The Health Centre provides round-the-clock health care facilities.", f"{BASE}research_and_technology_facilities/facilities.docx", .027),
        ],
    ),
    (
        "What emergency medical facilities are available at IIT Jodhpur?",
        f"{BASE}research_and_technology_facilities/facilities.docx",
        [
            C("The Health Centre provides round-the-clock health care facilities to Students, Faculty/Staff and dependents.", f"{BASE}research_and_technology_facilities/facilities.docx", .030),
            C("Extension of Medical Services. Visitors may access basic healthcare services through the Health Center.", f"{BASE}fallback/knowledge_buffer_general.docx", .029),
            C("Hospital Tie-Ups and Medical Support Network established with several leading hospitals.", f"{BASE}fallback/knowledge_buffer_general.docx", .028),
            C("Hostel facilities and residential campus information.", f"{BASE}hostel_accommodation/general_information.docx", .027),
        ],
    ),
    (
        "hostel mein students ko kya rules follow karne hote hain?",
        f"{BASE}hostel_accommodation/general_information.docx",
        [
            C("Semester fee includes hostel fees and refundable deposit information.", f"{BASE}finance/fees_and_finance.docx", .030),
            C("Hostel Booking. Hostel accommodation is managed by the Office of Students through an online booking and allotment system.", f"{BASE}hostel_accommodation/general_information.docx", .029),
            C("Hostel allocation is governed by institute policies and based on eligibility, program of study, year and availability.", f"{BASE}hostel_accommodation/general_information.docx", .028),
            C("Hostel booking links and Office of Students links.", f"{BASE}hostel_accommodation/general_information.docx", .026),
        ],
    ),
    (
        "Mtech regstration kaise hota hai?",
        f"{BASE}admissions/mtech_admissions.docx",
        [
            C("Admission to M Tech Program (Regular). The applicant must have a qualifying bachelor degree and satisfy the admission process.", f"{BASE}admissions/mtech_admissions.docx", .030),
            C("The department offers M.Tech in Mechanical Engineering with specializations and modes of admission.", f"{BASE}departments/mechanical_engineering/general_information.docx", .029),
            C("M.Tech curriculum and academic credit requirements.", f"{BASE}offices_and_administration/office_of_academics/curriculum.docx", .028),
        ],
    ),
]


def main() -> int:
    passed = 0
    for index, (query, expected, items) in enumerate(CASES, 1):
        ranked = rank_candidates(query, items, top_k=5)
        top_source = ranked[0].source if ranked else ""
        ok = expected.casefold() in top_source.casefold()
        print(f"[{index:02d}] {'PASS' if ok else 'FAIL'} | {query}")
        print(f"     expected: {expected}")
        print(f"     actual:   {top_source}")
        print(f"     top5:     {[item.source for item in ranked]}")
        if ok:
            passed += 1
    print(f"IITJ SNIPPET REPLAY: {passed}/{len(CASES)} PASS")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
