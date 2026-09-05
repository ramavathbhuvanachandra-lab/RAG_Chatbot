from langchain_core.documents import Document
from backend.broad_institutional_retrieval import assemble_broad_candidates, build_broad_retrieval_queries, detect_broad_scope
from backend.institutional_coverage import assess_institutional_coverage

def d(text, source):
    return Document(page_content=text, metadata={"source": source})

def test_broad_queries_are_bounded():
    q = build_broad_retrieval_queries("What departments are there?")
    assert 1 <= len(q) <= 2

def test_department_scope():
    assert detect_broad_scope("What departments are there?").name == "departments"

def test_unrelated_hostel_not_program_candidate():
    program = d("Academic programs include B.Tech, M.Tech and M.Sc.", "/programs/programs_overview.docx")
    hostel = d("Hostel accommodation and room facilities.", "/hostel_accommodation/general_information.docx")
    result = assemble_broad_candidates("What academic programs are available?", [program, hostel])
    assert program in result
    assert hostel not in result

def test_facilities_can_use_hostel():
    hostel = d("Hostel facilities include furnished rooms and common amenities.", "/hostel_accommodation/general_information.docx")
    result = assemble_broad_candidates("What facilities are available?", [hostel])
    assert hostel in result

def test_program_coverage_without_overview_is_partial():
    result = assess_institutional_coverage("What academic programs are available?", [d("M.Tech program details.", "/admissions/mtech.docx"), d("M.Sc program details.", "/admissions/msc.docx")])
    assert result["status"] == "partially_supported"

def test_program_coverage_with_overview_is_supported():
    result = assess_institutional_coverage("What academic programs are available?", [d("Academic programs overview. B.Tech M.Tech M.Sc. Ph.D. MBA.", "/programs/programs_overview.docx")])
    assert result["status"] == "supported"
