"""
FINAL IITJ INSTITUTIONAL CANDIDATE-VERIFICATION REGRESSION SUITE

120 total cases
----------------
96 normal / valid-evidence cases
24 related-but-wrong / negative cases

Coverage
--------
- Institute overview
- Admissions
- Academic registration
- Academics
- M.Tech
- Fees
- Hostel
- Campus facilities
- Library
- Healthcare
- Sports
- Transportation
- Student life
- Clubs / festivals
- Scholarships
- Student support
- Research / laboratories
- Placements
- Policies / safety
- IT / campus services
- English
- Hindi / Hinglish

The suite is intentionally broad rather than "hardcore".
The negative cases are there to verify that related evidence is not
mistaken for the requested evidence.

Core invariant:

    correct target + correct requested attribute
        -> accepted

    related topic but wrong target/context
        -> rejected

    target and attribute appearing in unrelated local contexts
        -> rejected for exact requests
"""

from types import SimpleNamespace

from backend.core.candidate_verification import verify_candidate
from backend.core.query_frame import (
    QueryFacet,
    QueryRequirement,
    SemanticQueryFrame,
)
from backend.core.retrieval_contracts import (
    CandidateAlignment,
    DocumentMeaning,
    EvidenceQuality,
    RetrievalCandidate,
    RetrievalProvenance,
)


# ============================================================================
# HELPERS
# ============================================================================


def make_alignment(
    *,
    topic_match: float = 1.0,
    entity_match: float = 1.0,
    attribute_match: float = 1.0,
    intent_match: float = 1.0,
    scope_match: float = 1.0,
    coverage: float = 1.0,
    semantic_match: float = 1.0,
    conflicts: tuple[str, ...] = (),
) -> CandidateAlignment:
    return CandidateAlignment(
        program_match=0.0,
        entity_match=entity_match,
        topic_match=topic_match,
        attribute_match=attribute_match,
        intent_match=intent_match,
        scope_match=scope_match,
        coverage=coverage,
        semantic_match=semantic_match,
        conflicts=conflicts,
    )


def make_candidate(
    case_name: str,
    evidence: str,
    *,
    request_type: str,
    alignment: CandidateAlignment | None = None,
) -> RetrievalCandidate:
    if alignment is None:
        alignment = make_alignment()

    document = SimpleNamespace(
        page_content=evidence,
        metadata={
            "source": case_name,
        },
    )

    meaning = DocumentMeaning(
        programs=(),
        entities=(),
        topics=(),
        attributes=(),
        intent=request_type,
        scope=(),
        qualifiers=(),
        constraints=(),
    )

    provenance = RetrievalProvenance(
        dense_rank=1,
        dense_score=0.99,
        bm25_rank=1,
        bm25_score=0.99,
        rrf_score=1.0,
        alternate_score_contribution=0.0,
        primary_query="",
        retrieval_queries=(),
        signals=(),
    )

    quality = EvidenceQuality(
        content_quality=1.0,
        structural_quality=1.0,
        noise=0.0,
    )

    return RetrievalCandidate(
        document=document,
        document_id=case_name,
        source=case_name,
        provenance=provenance,
        meaning=meaning,
        alignment=alignment,
        quality=quality,
        final_score=1.0,
    )


def make_frame(
    *,
    question: str,
    target: str,
    attribute: str,
    request_type: str,
    language: str = "en",
    mode: str = "exact",
) -> SemanticQueryFrame:
    facet = QueryFacet(
        name="requested_attribute",
        value=attribute,
        required=True,
        importance=1.0,
    )

    return SemanticQueryFrame(
        original_query=question,
        normalized_query=question.lower(),
        semantic_query=question,
        target=target,
        request_type=request_type,
        facets=(facet,),
        qualifiers=(),
        conditions=(),
        relations=(),
        temporal_context=None,
        comparison_targets=(),
        preserved_terms=(
            target,
            attribute,
        ),
        is_list_question=False,
        is_comparison_question=False,
        is_multi_part=False,
        requirement=QueryRequirement(
            mode=mode,
            require_target_alignment=True,
            require_attribute_alignment=True,
            require_scope_alignment=False,
            reject_explicit_conflict=True,
            allow_partial_evidence=False,
            max_unmatched_required_facets=0,
        ),
        confidence=0.97,
        needs_clarification=False,
        clarification_reason="",
        language=language,
        interpretation_status="trusted",
    )


# ============================================================================
# TEST CASES
# ============================================================================
#
# 01-96  = valid / expected acceptance
# 97-120 = related-but-wrong / expected rejection
# ============================================================================


TEST_CASES = [

    # ========================================================================
    # 01-12 : INSTITUTE OVERVIEW
    # ========================================================================

    {
        "name": "01_institute_identity",
        "question": "What is IIT Jodhpur?",
        "target": "IIT Jodhpur",
        "attribute": "Institute of National Importance",
        "request_type": "definition",
        "evidence": (
            "IIT Jodhpur is an Institute of National Importance established in 2008."
        ),
        "expected": True,
    },

    {
        "name": "02_institute_establishment",
        "question": "When was IIT Jodhpur established?",
        "target": "IIT Jodhpur",
        "attribute": "established in 2008",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur was established in 2008."
        ),
        "expected": True,
    },

    {
        "name": "03_campus_location",
        "question": "Where is the permanent campus located?",
        "target": "permanent campus",
        "attribute": "located near Karwar",
        "request_type": "directions",
        "evidence": (
            "The permanent campus is located near Karwar on the outskirts of Jodhpur."
        ),
        "expected": True,
    },

    {
        "name": "04_residential_campus",
        "question": "Is IIT Jodhpur a residential campus?",
        "target": "IIT Jodhpur",
        "attribute": "residential campus",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur is primarily a residential campus where most undergraduate "
            "students stay in institute hostels."
        ),
        "expected": True,
    },

    {
        "name": "05_reaching_campus",
        "question": "How can I reach the campus?",
        "target": "campus",
        "attribute": "reach by road",
        "request_type": "directions",
        "evidence": (
            "Students usually reach the campus by road from Jodhpur Airport or "
            "the railway station."
        ),
        "expected": True,
    },

    {
        "name": "06_campus_infrastructure",
        "question": "What major facilities are on campus?",
        "target": "campus",
        "attribute": "Central Library",
        "request_type": "information",
        "evidence": (
            "Major campus facilities include the Lecture Hall Complex, Central Library, "
            "research laboratories, hostels, Medical Centre, Sports Complex, and "
            "Student Activity Centre."
        ),
        "expected": True,
    },

    {
        "name": "07_wifi",
        "question": "Is Wi-Fi available on campus?",
        "target": "campus",
        "attribute": "Wi-Fi",
        "request_type": "information",
        "evidence": (
            "The campus provides Wi-Fi connectivity for academic and personal use."
        ),
        "expected": True,
    },

    {
        "name": "08_library_exists",
        "question": "Does IIT Jodhpur have a library?",
        "target": "IIT Jodhpur",
        "attribute": "central library",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur has a Central Library supporting teaching and research."
        ),
        "expected": True,
    },

    {
        "name": "09_medical_support_exists",
        "question": "Is medical support available?",
        "target": "IIT Jodhpur",
        "attribute": "medical support",
        "request_type": "information",
        "evidence": (
            "Students at IIT Jodhpur have access to campus medical support."
        ),
        "expected": True,
    },

    {
        "name": "10_sports_available",
        "question": "Are sports facilities available?",
        "target": "IIT Jodhpur",
        "attribute": "sports facilities",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur provides indoor and outdoor sports facilities on campus."
        ),
        "expected": True,
    },

    {
        "name": "11_campus_transport",
        "question": "How do students usually travel from Jodhpur airport?",
        "target": "Jodhpur airport",
        "attribute": "taxi or cab",
        "request_type": "directions",
        "evidence": (
            "Students usually travel from Jodhpur Airport to the campus by taxi or cab."
        ),
        "expected": True,
    },

    {
        "name": "12_campus_student_ecosystem",
        "question": "Does the campus support both academics and student life?",
        "target": "campus",
        "attribute": "student facilities",
        "request_type": "information",
        "evidence": (
            "The campus integrates teaching, research, innovation, student life, "
            "residential facilities, and community interaction."
        ),
        "expected": True,
    },

    # ========================================================================
    # 13-24 : ADMISSIONS
    # ========================================================================

    {
        "name": "13_btech_admission",
        "question": "How do I get admission to B.Tech?",
        "target": "B.Tech",
        "attribute": "admission",
        "request_type": "admission",
        "evidence": (
            "Admission to B.Tech is through JEE Advanced followed by JoSAA counselling."
        ),
        "expected": True,
    },

    {
        "name": "14_jee_advanced_josaa",
        "question": "What comes after JEE Advanced for B.Tech admission?",
        "target": "B.Tech admission",
        "attribute": "JoSAA counselling",
        "request_type": "admission",
        "evidence": (
            "After JEE Advanced, candidates participate in JoSAA counselling for "
            "undergraduate admission."
        ),
        "expected": True,
    },

    {
        "name": "15_admission_documents",
        "question": "What documents are required during admission?",
        "target": "admission",
        "attribute": "documents",
        "request_type": "admission",
        "evidence": (
            "Important admission documents include identity proof, educational "
            "certificates, category certificate, medical certificate, and admission "
            "related documents."
        ),
        "expected": True,
    },

    {
        "name": "16_branch_change",
        "question": "Can I change my branch?",
        "target": "branch change",
        "attribute": "eligibility",
        "request_type": "eligibility",
        "evidence": (
            "Branch change depends on institute regulations and the applicable eligibility criteria."
        ),
        "expected": True,
    },

    {
        "name": "17_admission_registration",
        "question": "When does academic registration happen after admission?",
        "target": "academic registration",
        "attribute": "academic calendar",
        "request_type": "registration",
        "evidence": (
            "Academic registration is conducted according to the academic calendar."
        ),
        "expected": True,
    },

    {
        "name": "18_admission_reporting",
        "question": "What happens during institute reporting?",
        "target": "institute reporting",
        "attribute": "document verification",
        "request_type": "procedure",
        "evidence": (
            "During institute reporting, students complete document verification and "
            "the required admission formalities."
        ),
        "expected": True,
    },

    {
        "name": "19_category_certificate",
        "question": "Why do I need a category certificate?",
        "target": "category certificate",
        "attribute": "reservation verification",
        "request_type": "information",
        "evidence": (
            "The category certificate is used for reservation verification during admission."
        ),
        "expected": True,
    },

    {
        "name": "20_parent_visit",
        "question": "Can parents visit the campus during admission?",
        "target": "parents",
        "attribute": "campus visit",
        "request_type": "information",
        "evidence": (
            "Visitors are generally allowed according to institute visitor guidelines "
            "and security procedures."
        ),
        "expected": True,
    },

    {
        "name": "21_mtech_admission_modes",
        "question": "What are the admission modes for M.Tech?",
        "target": "M.Tech",
        "attribute": "admission modes",
        "request_type": "admission",
        "evidence": (
            "M.Tech admission modes include regular with fellowship, industry-sponsored, "
            "self-sponsored, and part-time."
        ),
        "expected": True,
    },

    {
        "name": "22_itep_admission",
        "question": "Does IIT Jodhpur offer ITEP?",
        "target": "ITEP",
        "attribute": "academic programme",
        "request_type": "information",
        "evidence": (
            "The Integrated Teacher Education Programme (ITEP) is among the academic "
            "programmes offered by IIT Jodhpur."
        ),
        "expected": True,
    },

    {
        "name": "23_admission_workflow",
        "question": "What is the general undergraduate admission workflow?",
        "target": "undergraduate admission",
        "attribute": "seat allotment",
        "request_type": "procedure",
        "evidence": (
            "The undergraduate admission workflow includes eligibility, entrance "
            "examination, counselling, seat allotment, fee payment, document verification, "
            "and institute reporting."
        ),
        "expected": True,
    },

    {
        "name": "24_admission_documents_identity",
        "question": "Is Aadhaar used during admission verification?",
        "target": "Aadhaar",
        "attribute": "identity verification",
        "request_type": "information",
        "evidence": (
            "Aadhaar is used as an identity-verification document during admission."
        ),
        "expected": True,
    },

    # ========================================================================
    # 25-36 : ACADEMICS / REGISTRATION
    # ========================================================================

    {
        "name": "25_sgpa",
        "question": "What is SGPA?",
        "target": "SGPA",
        "attribute": "Semester Grade Point Average",
        "request_type": "definition",
        "evidence": (
            "SGPA stands for Semester Grade Point Average for one semester."
        ),
        "expected": True,
    },

    {
        "name": "26_cgpa",
        "question": "What is CGPA?",
        "target": "CGPA",
        "attribute": "cumulative academic performance",
        "request_type": "definition",
        "evidence": (
            "CGPA represents the overall cumulative academic performance."
        ),
        "expected": True,
    },

    {
        "name": "27_electives",
        "question": "Can students choose electives?",
        "target": "electives",
        "attribute": "curriculum and prerequisites",
        "request_type": "eligibility",
        "evidence": (
            "Students can choose electives subject to the curriculum and applicable prerequisites."
        ),
        "expected": True,
    },

    {
        "name": "28_faculty_advisor",
        "question": "Who is a faculty advisor?",
        "target": "faculty advisor",
        "attribute": "academic guidance",
        "request_type": "definition",
        "evidence": (
            "A faculty advisor is a faculty member who guides students academically."
        ),
        "expected": True,
    },

    {
        "name": "29_summer_mtech_backlog",
        "question": "How does summer M.Tech registration work?",
        "target": "M.Tech",
        "attribute": "summer registration",
        "request_type": "registration",
        "evidence": (
            "Summer course registration may be offered to M.Tech students to clear "
            "backlog courses and regular credit courses."
        ),
        "expected": True,
    },

    {
        "name": "30_summer_course_approval",
        "question": "Who approves a summer course?",
        "target": "summer course",
        "attribute": "Dean or Associate Dean approval",
        "request_type": "procedure",
        "evidence": (
            "A summer course is offered after approval of the Dean or Associate Dean "
            "and availability of a faculty member to run it."
        ),
        "expected": True,
    },

    {
        "name": "31_semester_registration_process",
        "question": "How do I register for a semester?",
        "target": "semester registration",
        "attribute": "course registration",
        "request_type": "registration",
        "evidence": (
            "Semester registration includes fee confirmation and course registration "
            "within the notified academic schedule."
        ),
        "expected": True,
    },

    {
        "name": "32_course_registration",
        "question": "What is course registration?",
        "target": "course registration",
        "attribute": "academic process",
        "request_type": "definition",
        "evidence": (
            "Course registration is the academic process through which students register "
            "for courses within the semester."
        ),
        "expected": True,
    },

    {
        "name": "33_academic_calendar",
        "question": "Why is the academic calendar important for registration?",
        "target": "academic calendar",
        "attribute": "registration schedule",
        "request_type": "information",
        "evidence": (
            "The academic calendar determines the schedule for registration and other "
            "academic activities."
        ),
        "expected": True,
    },

    {
        "name": "34_academic_integrity",
        "question": "What is academic integrity?",
        "target": "academic integrity",
        "attribute": "honesty in assignments, exams and research",
        "request_type": "definition",
        "evidence": (
            "Academic integrity means maintaining honesty in assignments, examinations, "
            "and research."
        ),
        "expected": True,
    },

    {
        "name": "35_dual_degree",
        "question": "Does IIT Jodhpur offer dual degree programmes?",
        "target": "dual degree programmes",
        "attribute": "academic programme",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur offers dual degree programmes among its academic offerings."
        ),
        "expected": True,
    },

    {
        "name": "36_academic_advising",
        "question": "Where can students get academic advising?",
        "target": "academic advising",
        "attribute": "student support",
        "request_type": "information",
        "evidence": (
            "Academic advising is available as part of student support services at IIT Jodhpur."
        ),
        "expected": True,
    },

    # ========================================================================
    # 37-48 : FEES
    # ========================================================================

    {
        "name": "37_tuition_fee",
        "question": "What is the tuition fee?",
        "target": "tuition fee",
        "attribute": "₹50,000 per semester",
        "request_type": "cost",
        "evidence": (
            "The tuition fee is ₹50,000 per semester under the stated fee structure."
        ),
        "expected": True,
    },

    {
        "name": "38_admission_fee",
        "question": "What is the admission fee?",
        "target": "admission fee",
        "attribute": "₹3,800",
        "request_type": "cost",
        "evidence": (
            "The admission fee is ₹3,800 and is payable one time during admission."
        ),
        "expected": True,
    },

    {
        "name": "39_convocation_fee",
        "question": "What is the convocation fee?",
        "target": "convocation fee",
        "attribute": "₹4,000",
        "request_type": "cost",
        "evidence": (
            "The convocation fee is ₹4,000 and is a one-time fee."
        ),
        "expected": True,
    },

    {
        "name": "40_semester_fee",
        "question": "How much is the semester fee?",
        "target": "semester fee",
        "attribute": "₹32,250",
        "request_type": "cost",
        "evidence": (
            "The semester fee is ₹32,250 per semester under the stated fee structure."
        ),
        "expected": True,
    },

    {
        "name": "41_refundable_deposit",
        "question": "What is the refundable deposit?",
        "target": "refundable deposit",
        "attribute": "₹8,000",
        "request_type": "cost",
        "evidence": (
            "The refundable deposit is ₹8,000 and is payable one time."
        ),
        "expected": True,
    },

    {
        "name": "42_experiential_fee",
        "question": "What is the experiential and professional development fee?",
        "target": "Experiential and Professional Development Fee",
        "attribute": "₹2,000",
        "request_type": "cost",
        "evidence": (
            "The Experiential and Professional Development Fee is ₹2,000."
        ),
        "expected": True,
    },

    {
        "name": "43_hostel_in_semester",
        "question": "Is hostel fee included in the semester fee?",
        "target": "semester fee",
        "attribute": "hostel fees included",
        "request_type": "information",
        "evidence": (
            "The semester fee includes the hostel fees."
        ),
        "expected": True,
    },

    {
        "name": "44_mess_advance",
        "question": "Is there a separate mess advance?",
        "target": "Mess advance",
        "attribute": "separate payment",
        "request_type": "information",
        "evidence": (
            "A separate Mess advance has to be paid in each semester."
        ),
        "expected": True,
    },

    {
        "name": "45_reserved_tuition_exemption",
        "question": "Are SC, ST and PwD students exempt from tuition fees?",
        "target": "SC, ST and PwD students",
        "attribute": "tuition fees exempt",
        "request_type": "eligibility",
        "evidence": (
            "SC, ST and PwD students are exempted from paying the tuition fees."
        ),
        "expected": True,
    },

    {
        "name": "46_mba_program_fee",
        "question": "What is the MBA tuition fee?",
        "target": "MBA program",
        "attribute": "INR 2,25,000 per semester",
        "request_type": "cost",
        "evidence": (
            "For the MBA program, the tuition fee is INR 2,25,000 per semester."
        ),
        "expected": True,
    },

    {
        "name": "47_dual_degree_day_scholar_fee",
        "question": "What fees are paid by international dual degree students?",
        "target": "international dual degree program",
        "attribute": "day-scholar fees at IIT Jodhpur",
        "request_type": "cost",
        "evidence": (
            "Students opting for the international dual degree program in their second "
            "year are required to pay the day-scholar fees at IIT Jodhpur."
        ),
        "expected": True,
    },

    {
        "name": "48_application_fee_categories",
        "question": "What is the application fee for reserved categories?",
        "target": "SC ST PwD categories",
        "attribute": "₹600 application fee",
        "request_type": "cost",
        "evidence": (
            "The non-refundable application fee is ₹600 for SC, ST and PwD categories."
        ),
        "expected": True,
    },

    # ========================================================================
    # 49-60 : HOSTEL
    # ========================================================================

    {
        "name": "49_hostel_general_facilities",
        "question": "What hostel facilities are available?",
        "target": "hostel facilities",
        "attribute": "Wi-Fi and LAN connectivity",
        "request_type": "information",
        "evidence": (
            "Hostel facilities include Wi-Fi and LAN connectivity, reading rooms, "
            "common rooms, indoor games, gymnasium access, and laundry facilities."
        ),
        "expected": True,
    },

    {
        "name": "50_total_hostels",
        "question": "How many student hostels are there?",
        "target": "student hostels",
        "attribute": "17 student hostels",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur has a total of 17 student hostels."
        ),
        "expected": True,
    },

    {
        "name": "51_girls_boys_hostels",
        "question": "How many girls' and boys' hostels are there?",
        "target": "student hostels",
        "attribute": "5 girls' hostels and 12 boys' hostels",
        "request_type": "information",
        "evidence": (
            "The institute has 5 girls' hostels and 12 boys' hostels."
        ),
        "expected": True,
    },

    {
        "name": "52_first_year_allocation",
        "question": "How are first-year students accommodated?",
        "target": "first-year students",
        "attribute": "double-occupancy basis",
        "request_type": "procedure",
        "evidence": (
            "All first-year students are provided rooms on a double-occupancy basis."
        ),
        "expected": True,
    },

    {
        "name": "53_hostel_room_size",
        "question": "How large are hostel rooms?",
        "target": "hostel rooms",
        "attribute": "80 sq ft",
        "request_type": "information",
        "evidence": (
            "Each hostel room is approximately 80 sq ft in size."
        ),
        "expected": True,
    },

    {
        "name": "54_hostel_wifi",
        "question": "Do hostels have Wi-Fi and LAN?",
        "target": "hostels",
        "attribute": "Wi-Fi and LAN connectivity",
        "request_type": "information",
        "evidence": (
            "Wi-Fi and LAN connectivity for academic and personal use are provided in hostels."
        ),
        "expected": True,
    },

    {
        "name": "55_hostel_gym",
        "question": "Do the hostels have gyms?",
        "target": "hostels",
        "attribute": "indoor gym facilities",
        "request_type": "information",
        "evidence": (
            "Indoor gym facilities are available in each hostel."
        ),
        "expected": True,
    },

    {
        "name": "56_hostel_laundry",
        "question": "Is laundry available in the hostels?",
        "target": "hostels",
        "attribute": "laundry facilities",
        "request_type": "information",
        "evidence": (
            "Laundry facilities are available on campus for hostel residents."
        ),
        "expected": True,
    },

    {
        "name": "57_hostel_management",
        "question": "Who manages hostel accommodation?",
        "target": "hostel accommodation",
        "attribute": "Office of Students",
        "request_type": "information",
        "evidence": (
            "Hostel accommodation is managed by the Office of Students (Hostel Affairs)."
        ),
        "expected": True,
    },

    {
        "name": "58_hostel_erp",
        "question": "How is hostel booking handled?",
        "target": "hostel booking",
        "attribute": "online ERP-based system",
        "request_type": "procedure",
        "evidence": (
            "Hostel booking is handled through an online ERP-based booking and "
            "allotment system."
        ),
        "expected": True,
    },

    {
        "name": "59_short_term_visitor_hostel",
        "question": "Can visitors get short-term hostel accommodation?",
        "target": "short-term hostel accommodation",
        "attribute": "visitors",
        "request_type": "eligibility",
        "evidence": (
            "Short-term hostel accommodation is available for visitors, interns, "
            "event participants, and other approved invitees for stays not exceeding ten days."
        ),
        "expected": True,
    },

    {
        "name": "60_long_stay_hostel",
        "question": "What is the long-stay hostel charge?",
        "target": "long bookings",
        "attribute": "₹2,175 per month",
        "request_type": "cost",
        "evidence": (
            "For long bookings, the hostel charge is ₹2,175 per month for the applicable duration."
        ),
        "expected": True,
    },

    # ========================================================================
    # 61-72 : CAMPUS / LIBRARY / HEALTH / SPORTS / TRANSPORT
    # ========================================================================

    {
        "name": "61_library_resources",
        "question": "What resources does the Central Library provide?",
        "target": "Central Library",
        "attribute": "digital resources",
        "request_type": "information",
        "evidence": (
            "The Central Library provides books, reference materials, digital resources, "
            "research journals, reading halls, and online databases."
        ),
        "expected": True,
    },

    {
        "name": "62_library_journals",
        "question": "Does the library provide research journals?",
        "target": "Central Library",
        "attribute": "research journals",
        "request_type": "information",
        "evidence": (
            "The Central Library provides research journals for teaching and research."
        ),
        "expected": True,
    },

    {
        "name": "63_medical_centre",
        "question": "What does the campus medical centre provide?",
        "target": "medical centre",
        "attribute": "medical consultation",
        "request_type": "information",
        "evidence": (
            "The campus medical centre provides medical consultation, first aid, "
            "emergency support, and health awareness."
        ),
        "expected": True,
    },

    {
        "name": "64_health_emergency",
        "question": "Is emergency medical support available?",
        "target": "medical centre",
        "attribute": "emergency support",
        "request_type": "information",
        "evidence": (
            "The campus medical centre provides emergency medical support."
        ),
        "expected": True,
    },

    {
        "name": "65_sports",
        "question": "What sports facilities are available?",
        "target": "sports facilities",
        "attribute": "cricket",
        "request_type": "information",
        "evidence": (
            "Sports facilities include cricket, football, basketball, badminton, "
            "volleyball, athletics, indoor games, and fitness facilities."
        ),
        "expected": True,
    },

    {
        "name": "66_transport",
        "question": "How do students usually reach campus from Jodhpur city?",
        "target": "campus",
        "attribute": "taxi, cab or bus",
        "request_type": "directions",
        "evidence": (
            "Students generally reach campus from Jodhpur city by taxi, cab, or bus."
        ),
        "expected": True,
    },

    {
        "name": "67_airport_transport",
        "question": "How do I reach IIT Jodhpur from the airport?",
        "target": "IIT Jodhpur",
        "attribute": "taxi or cab from airport",
        "request_type": "directions",
        "evidence": (
            "Students usually travel by taxi or cab from Jodhpur Airport to the campus."
        ),
        "expected": True,
    },

    {
        "name": "68_dining",
        "question": "What dining facilities are available?",
        "target": "dining facilities",
        "attribute": "Mess",
        "request_type": "information",
        "evidence": (
            "Dining facilities include messes, food courts, cafeterias, and convenience stores."
        ),
        "expected": True,
    },

    {
        "name": "69_food_court",
        "question": "What is available at the campus food court?",
        "target": "Food Court",
        "attribute": "snacks and beverages",
        "request_type": "information",
        "evidence": (
            "The Food Court provides snacks and beverages."
        ),
        "expected": True,
    },

    {
        "name": "70_student_activity_centre",
        "question": "Does the campus have a Student Activity Centre?",
        "target": "Student Activity Centre",
        "attribute": "student facility",
        "request_type": "information",
        "evidence": (
            "The Student Activity Centre is one of the major facilities on campus."
        ),
        "expected": True,
    },

    {
        "name": "71_campus_wifi_academic",
        "question": "Can students use campus Wi-Fi for academic work?",
        "target": "campus Wi-Fi",
        "attribute": "academic use",
        "request_type": "information",
        "evidence": (
            "Campus Wi-Fi and LAN connectivity are available for academic and personal use."
        ),
        "expected": True,
    },

    {
        "name": "72_campus_residential_facilities",
        "question": "Does the campus have residential facilities?",
        "target": "campus",
        "attribute": "hostels",
        "request_type": "information",
        "evidence": (
            "The campus includes residential facilities and hostels for students."
        ),
        "expected": True,
    },

    # ========================================================================
    # 73-84 : STUDENT LIFE / CLUBS / FESTIVALS / SUPPORT
    # ========================================================================

    {
        "name": "73_clubs",
        "question": "What extracurricular clubs are available?",
        "target": "student clubs",
        "attribute": "technical activities",
        "request_type": "information",
        "evidence": (
            "Student clubs and societies cover technical, cultural, sports, photography, "
            "music, dance, entrepreneurship, robotics, coding, and other activities."
        ),
        "expected": True,
    },

    {
        "name": "74_multiple_clubs",
        "question": "Can I join more than one club?",
        "target": "multiple clubs",
        "attribute": "students may join multiple clubs",
        "request_type": "eligibility",
        "evidence": (
            "Students may join multiple clubs and societies based on their interests."
        ),
        "expected": True,
    },

    {
        "name": "75_inter_iit",
        "question": "Does IIT Jodhpur participate in Inter-IIT competitions?",
        "target": "IIT Jodhpur",
        "attribute": "Inter-IIT competitions",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur participates in Inter-IIT competitions."
        ),
        "expected": True,
    },

    {
        "name": "76_festivals",
        "question": "What major student festivals are held at IIT Jodhpur?",
        "target": "student festivals",
        "attribute": "Varchas",
        "request_type": "information",
        "evidence": (
            "Major student festivals include Varchas, Ignus, Prometeo, and Sandstone Summit."
        ),
        "expected": True,
    },

    {
        "name": "77_varchas",
        "question": "What is Varchas?",
        "target": "Varchas",
        "attribute": "annual sports festival",
        "request_type": "definition",
        "evidence": (
            "Varchas is the annual sports festival of IIT Jodhpur."
        ),
        "expected": True,
    },

    {
        "name": "78_ignus",
        "question": "What is Ignus?",
        "target": "Ignus",
        "attribute": "annual socio-cultural festival",
        "request_type": "definition",
        "evidence": (
            "Ignus is the annual socio-cultural festival of IIT Jodhpur."
        ),
        "expected": True,
    },

    {
        "name": "79_prometeo",
        "question": "What is Prometeo?",
        "target": "Prometeo",
        "attribute": "annual technical and entrepreneurial festival",
        "request_type": "definition",
        "evidence": (
            "Prometeo is the annual technical and entrepreneurial festival of IIT Jodhpur."
        ),
        "expected": True,
    },

    {
        "name": "80_sandstone",
        "question": "What is Sandstone Summit?",
        "target": "Sandstone Summit",
        "attribute": "annual business summit",
        "request_type": "definition",
        "evidence": (
            "Sandstone Summit is the annual business summit of IIT Jodhpur."
        ),
        "expected": True,
    },

    {
        "name": "81_mentorship",
        "question": "Are mentorship programmes available?",
        "target": "mentorship programmes",
        "attribute": "student guidance",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur provides mentorship programmes for student guidance and development."
        ),
        "expected": True,
    },

    {
        "name": "82_student_guide",
        "question": "Who helps new students with campus-related questions?",
        "target": "Student Guide",
        "attribute": "academic, hostel, and campus guidance",
        "request_type": "information",
        "evidence": (
            "Every new student is assigned a Student Guide to help with academic, "
            "hostel, and campus-related queries."
        ),
        "expected": True,
    },

    {
        "name": "83_womens_cell",
        "question": "Is there a Women Cell for student safety and welfare?",
        "target": "Women Cell",
        "attribute": "student safety and welfare",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur has a Women Cell that works to ensure a safe, respectful, "
            "and inclusive environment for female students."
        ),
        "expected": True,
    },

    {
        "name": "84_ragging",
        "question": "Is ragging allowed at IIT Jodhpur?",
        "target": "ragging",
        "attribute": "prohibited",
        "request_type": "eligibility",
        "evidence": (
            "Ragging is strictly prohibited at IIT Jodhpur."
        ),
        "expected": True,
    },

    # ========================================================================
    # 85-96 : RESEARCH / PLACEMENTS / SCHOLARSHIPS / IT / SUPPORT
    # ========================================================================

    {
        "name": "85_ug_research",
        "question": "Can undergraduate students do research?",
        "target": "undergraduate students",
        "attribute": "research participation",
        "request_type": "eligibility",
        "evidence": (
            "Undergraduate students can participate in faculty-led research projects."
        ),
        "expected": True,
    },

    {
        "name": "86_research_labs",
        "question": "Does IIT Jodhpur have research labs?",
        "target": "IIT Jodhpur",
        "attribute": "research laboratories",
        "request_type": "information",
        "evidence": (
            "IIT Jodhpur departments support multiple research laboratories."
        ),
        "expected": True,
    },

    {
        "name": "87_research_internships",
        "question": "Are research internships encouraged?",
        "target": "research internships",
        "attribute": "encouraged",
        "request_type": "information",
        "evidence": (
            "Research and industry internships are encouraged for students."
        ),
        "expected": True,
    },

    {
        "name": "88_ppo",
        "question": "What is a PPO?",
        "target": "PPO",
        "attribute": "Pre-Placement Offer",
        "request_type": "definition",
        "evidence": (
            "A PPO is a Pre-Placement Offer received after a successful internship."
        ),
        "expected": True,
    },

    {
        "name": "89_placement_preparation",
        "question": "How should I prepare for placements?",
        "target": "placements",
        "attribute": "projects, academics and interview practice",
        "request_type": "procedure",
        "evidence": (
            "Students should build projects, strengthen academics, and practice interviews "
            "to prepare for placements."
        ),
        "expected": True,
    },

    {
        "name": "90_recruiters",
        "question": "Which companies recruit students?",
        "target": "recruiters",
        "attribute": "Amazon",
        "request_type": "information",
        "evidence": (
            "Recruiters have included Amazon, Google, Microsoft, NVIDIA, Samsung R&D, "
            "Goldman Sachs, and other organizations."
        ),
        "expected": True,
    },

    {
        "name": "91_higher_studies",
        "question": "Can B.Tech graduates pursue higher studies?",
        "target": "B.Tech graduates",
        "attribute": "higher studies",
        "request_type": "eligibility",
        "evidence": (
            "B.Tech graduates can continue with postgraduate and other higher studies."
        ),
        "expected": True,
    },

    {
        "name": "92_prototyping",
        "question": "Are there facilities for hardware prototyping?",
        "target": "prototyping labs",
        "attribute": "3D printing",
        "request_type": "information",
        "evidence": (
            "Specialized prototyping labs support hardware development, 3D printing, "
            "electronics testing, and technology validation."
        ),
        "expected": True,
    },

    {
        "name": "93_hinglish_hostel",
        "question": "Hostel mein Wi-Fi aur study room milta hai kya?",
        "target": "hostel",
        "attribute": "Wi-Fi and study rooms",
        "request_type": "information",
        "language": "hi",
        "evidence": (
            "Hostel facilities include Wi-Fi and LAN connectivity and study rooms "
            "for group and individual academic work."
        ),
        "expected": True,
    },

    {
        "name": "94_hinglish_library",
        "question": "Library mein online resources available hain kya?",
        "target": "library",
        "attribute": "online databases",
        "request_type": "information",
        "language": "hi",
        "evidence": (
            "The Central Library provides digital resources, research journals, "
            "reading halls, and online databases."
        ),
        "expected": True,
    },

    {
        "name": "95_hinglish_registration",
        "question": "Semester registration ke liye kya karna hota hai?",
        "target": "semester registration",
        "attribute": "fee confirmation",
        "request_type": "procedure",
        "language": "hi",
        "evidence": (
            "Semester registration includes institute reporting, identity verification, "
            "fee confirmation, hostel allocation, and course registration."
        ),
        "expected": True,
    },

    {
        "name": "96_hinglish_medical",
        "question": "Campus mein medical help milti hai kya?",
        "target": "campus medical centre",
        "attribute": "medical support",
        "request_type": "information",
        "language": "hi",
        "evidence": (
            "The campus medical centre provides medical consultation, first aid, "
            "and emergency support."
        ),
        "expected": True,
    },

    # ========================================================================
    # 97-120 : NEGATIVE / RELATED-BUT-WRONG CASES
    # ========================================================================
    #
    # These should REJECT.
    #
    # The candidate often contains some combination of the right topic,
    # target, attribute, or vocabulary, but the requested fact is wrong.
    # ========================================================================

    {
        "name": "97_reject_mtech_btech_cross_context",
        "question": "How does summer M.Tech registration work?",
        "target": "M.Tech",
        "attribute": "registration",
        "request_type": "registration",
        "evidence": (
            "M.Tech students follow the prescribed academic curriculum.\n\n"
            "Summer registration is conducted online for B.Tech students."
        ),
        "expected": False,
    },

    {
        "name": "98_reject_hostel_guest_house",
        "question": "What is the hostel fee?",
        "target": "hostel",
        "attribute": "charges",
        "request_type": "cost",
        "evidence": (
            "Hostel accommodation is available for eligible students.\n\n"
            "Guest house room charges are ₹1,200 per room per day."
        ),
        "expected": False,
    },

    {
        "name": "99_reject_tuition_hostel",
        "question": "What is the tuition fee?",
        "target": "tuition fee",
        "attribute": "amount",
        "request_type": "cost",
        "evidence": (
            "Hostel accommodation charges are payable according to the applicable "
            "hostel fee schedule."
        ),
        "expected": False,
    },

    {
        "name": "100_reject_admission_modes_eligibility",
        "question": "What are the admission modes?",
        "target": "admission modes",
        "attribute": "modes",
        "request_type": "admission",
        "evidence": (
            "Admission eligibility depends on academic qualification, minimum marks, "
            "and other eligibility requirements."
        ),
        "expected": False,
    },

    {
        "name": "101_reject_summer_curriculum_for_registration",
        "question": "How do I register for a summer course?",
        "target": "summer course",
        "attribute": "registration",
        "request_type": "registration",
        "evidence": (
            "The summer term may include additional courses and follows the academic "
            "curriculum and credit requirements."
        ),
        "expected": False,
    },

    {
        "name": "102_reject_library_sports",
        "question": "What resources does the library provide?",
        "target": "library",
        "attribute": "resources",
        "request_type": "information",
        "evidence": (
            "The campus provides cricket, football, basketball, badminton, volleyball, "
            "and athletics facilities."
        ),
        "expected": False,
    },

    {
        "name": "103_reject_visitor_rate_for_student",
        "question": "What is the hostel charge for students?",
        "target": "students",
        "attribute": "hostel charge",
        "request_type": "cost",
        "evidence": (
            "Hostel accommodation charges for visitors are ₹450 per day for the applicable "
            "double-occupancy rate."
        ),
        "expected": False,
    },

    {
        "name": "104_reject_hinglish_mixed_context",
        "question": "M.Tech ka summer mein registration kaise hota hai?",
        "target": "M.Tech",
        "attribute": "registration",
        "request_type": "registration",
        "language": "hi",
        "evidence": (
            "M.Tech students follow the institute's academic regulations.\n\n"
            "Summer registration is available to B.Tech students through the department portal."
        ),
        "expected": False,
    },

    {
        "name": "105_reject_mtech_and_registration_separate",
        "question": "Summer M.Tech registration kaise hota hai?",
        "target": "M.Tech",
        "attribute": "registration",
        "request_type": "registration",
        "language": "hi",
        "evidence": (
            "M.Tech students follow the academic regulations.\n\n"
            "Registration for the summer term applies to another programme."
        ),
        "expected": False,
    },

    {
        "name": "106_reject_hostel_plus_guest_price",
        "question": "Hostel ka fee kitna hai?",
        "target": "hostel",
        "attribute": "fee",
        "request_type": "cost",
        "language": "hi",
        "evidence": (
            "The hostel provides accommodation to students.\n\n"
            "Guest house rooms cost ₹1,200 per day."
        ),
        "expected": False,
    },

    {
        "name": "107_reject_generic_tuition_for_reserved_waiver",
        "question": "SC ST PwD students ko tuition fee deni padti hai kya?",
        "target": "SC ST PwD students",
        "attribute": "tuition fee exemption",
        "request_type": "eligibility",
        "language": "hi",
        "evidence": (
            "The tuition fee for the programme is ₹50,000 per semester."
        ),
        "expected": False,
    },

    {
        "name": "108_reject_scholarship_with_loan",
        "question": "What scholarships are available?",
        "target": "scholarships",
        "attribute": "financial assistance",
        "request_type": "information",
        "evidence": (
            "Students may obtain educational loans from banks according to applicable "
            "banking procedures."
        ),
        "expected": False,
    },

    {
        "name": "109_reject_sports_for_hostel_facilities",
        "question": "What hostel facilities are available?",
        "target": "hostel",
        "attribute": "hostel facilities",
        "request_type": "information",
        "evidence": (
            "The campus sports complex provides cricket, football, volleyball, and athletics."
        ),
        "expected": False,
    },

    {
        "name": "110_reject_medical_for_library",
        "question": "What does the library provide?",
        "target": "library",
        "attribute": "digital resources",
        "request_type": "information",
        "evidence": (
            "The medical centre provides first aid, health awareness, and emergency support."
        ),
        "expected": False,
    },

    {
        "name": "111_reject_research_lab_for_prototyping",
        "question": "Where can I do hardware prototyping?",
        "target": "prototyping labs",
        "attribute": "3D printing",
        "request_type": "information",
        "evidence": (
            "Research laboratories support biological and life-sciences research."
        ),
        "expected": False,
    },

    {
        "name": "112_reject_airport_question_for_campus_location",
        "question": "How do I reach IIT Jodhpur from the airport?",
        "target": "IIT Jodhpur",
        "attribute": "airport route",
        "request_type": "directions",
        "evidence": (
            "The permanent campus is located near Karwar on the outskirts of Jodhpur."
        ),
        "expected": False,
    },

    {
        "name": "113_reject_branch_change_for_admission",
        "question": "How can I change my branch?",
        "target": "branch change",
        "attribute": "procedure",
        "request_type": "procedure",
        "evidence": (
            "Undergraduate admission is conducted through JEE Advanced followed by "
            "JoSAA counselling."
        ),
        "expected": False,
    },

    {
        "name": "114_reject_cgpa_for_sgpa",
        "question": "What is SGPA?",
        "target": "SGPA",
        "attribute": "Semester Grade Point Average",
        "request_type": "definition",
        "evidence": (
            "CGPA represents the overall cumulative academic performance."
        ),
        "expected": False,
    },

    {
        "name": "115_reject_registration_for_hostel_allocation",
        "question": "How do I complete academic registration?",
        "target": "academic registration",
        "attribute": "course registration",
        "request_type": "registration",
        "evidence": (
            "Hostel allocation is based on eligibility, programme of study, year, and availability."
        ),
        "expected": False,
    },

    {
        "name": "116_reject_room_size_for_hostel_facilities",
        "question": "What hostel facilities are available?",
        "target": "hostel",
        "attribute": "facilities",
        "request_type": "information",
        "evidence": (
            "Each hostel room is approximately 80 sq ft in size."
        ),
        "expected": False,
    },

    {
        "name": "117_reject_women_cell_for_medical",
        "question": "What medical support is available?",
        "target": "medical support",
        "attribute": "medical consultation",
        "request_type": "information",
        "evidence": (
            "The Women Cell supports a safe, respectful, and inclusive environment "
            "for female students."
        ),
        "expected": False,
    },

    {
        "name": "118_reject_ppo_for_internship",
        "question": "What is a PPO?",
        "target": "PPO",
        "attribute": "Pre-Placement Offer",
        "request_type": "definition",
        "evidence": (
            "Research and industry internships are encouraged for students."
        ),
        "expected": False,
    },

    {
        "name": "119_reject_scholarship_for_admission_fee",
        "question": "What is the scholarship amount?",
        "target": "scholarship",
        "attribute": "amount",
        "request_type": "cost",
        "evidence": (
            "The admission fee is ₹3,800 and is payable one time during admission."
        ),
        "expected": False,
    },

    {
        "name": "120_reject_unknown_free_hostel_claim",
        "question": "Bhai free wala hostel ka kya scene hai?",
        "target": "free hostel",
        "attribute": "free accommodation",
        "request_type": "information",
        "language": "hi",
        "evidence": (
            "Scholarship schemes provide financial assistance to eligible students."
        ),
        "expected": False,
    },
]


# ============================================================================
# RUNNER
# ============================================================================


def run_all_tests() -> int:
    total = len(TEST_CASES)
    passed = 0
    failed = 0
    errors = 0

    print("=" * 80)
    print("FINAL IITJ INSTITUTIONAL VERIFICATION REGRESSION SUITE")
    print("=" * 80)
    print(f"TOTAL CASES: {total}")
    print("EXPECTED POSITIVE CASES: 96")
    print("EXPECTED NEGATIVE CASES: 24")
    print()

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):
        try:
            frame = make_frame(
                question=case["question"],
                target=case["target"],
                attribute=case["attribute"],
                request_type=case["request_type"],
                language=case.get(
                    "language",
                    "en",
                ),
            )

            candidate = make_candidate(
                case_name=case["name"],
                evidence=case["evidence"],
                request_type=case["request_type"],
            )

            decision = verify_candidate(
                frame,
                candidate,
            )

            expected = case["expected"]
            actual = decision.accepted

            if actual == expected:
                passed += 1

                print(
                    f"{index:03d}. PASS  "
                    f"{case['name']}"
                )

            else:
                failed += 1

                print(
                    f"{index:03d}. FAIL  "
                    f"{case['name']}"
                )

                print(
                    f"      Question : {case['question']}"
                )

                print(
                    f"      Expected : {expected}"
                )

                print(
                    f"      Actual   : {actual}"
                )

                print(
                    f"      Status   : {decision.status}"
                )

                print(
                    f"      Target   : "
                    f"{decision.target_grounded}"
                )

                print(
                    f"      Attribute: "
                    f"{decision.attribute_grounded}"
                )

                print(
                    f"      Coverage : "
                    f"{decision.coverage}"
                )

                print(
                    "      Reasons  : "
                    + ", ".join(
                        decision.reasons
                    )
                )

        except Exception as exc:
            errors += 1

            print(
                f"{index:03d}. ERROR "
                f"{case['name']}"
            )

            print(
                f"      {type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 80)

    print(
        f"RESULT: {passed}/{total} passed | "
        f"{failed} failed | "
        f"{errors} errors"
    )

    print("=" * 80)

    if failed == 0 and errors == 0:
        print(
            "FINAL IITJ 120-CASE VERIFICATION SUITE: PASS"
        )

        return 0

    print(
        "FINAL IITJ 120-CASE VERIFICATION SUITE: FAIL"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        run_all_tests()
    )