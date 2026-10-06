from __future__ import annotations

CASES = [
    {
        "id": "mtech_regular_eligibility",
        "question": "What are the M.Tech eligibility requirements?",
        "positive_markers": ("m.tech", "eligibility", "bachelor", "minimum", "gate", "written test"),
        "negative_markers": ("m.sc", "ph.d", "mba", "jam"),
        "expected_source_hints": ("mtech", "admission"),
    },
    {
        "id": "mtech_percentage",
        "question": "What minimum percentage or CGPA is required for M.Tech admission?",
        "positive_markers": ("m.tech", "minimum", "percentage", "cgpa", "marks"),
        "negative_markers": ("m.sc", "jam", "ph.d"),
        "expected_source_hints": ("mtech",),
    },
    {
        "id": "mtech_gate",
        "question": "Is GATE required for M.Tech admission?",
        "positive_markers": ("m.tech", "gate", "admission", "written test"),
        "negative_markers": ("jam", "ph.d", "mba"),
        "expected_source_hints": ("mtech", "admission"),
    },
    {
        "id": "mtech_qualification",
        "question": "Which bachelor's degree qualifications are accepted for M.Tech admission?",
        "positive_markers": ("m.tech", "bachelor", "degree", "relevant"),
        "negative_markers": ("m.sc", "jam", "ph.d"),
        "expected_source_hints": ("mtech",),
    },
    {
        "id": "mtech_written_test",
        "question": "Is there a written test for M.Tech admission?",
        "positive_markers": ("m.tech", "written test", "admission"),
        "negative_markers": ("m.sc", "jam", "mba"),
        "expected_source_hints": ("mtech", "admission"),
    },
    {
        "id": "mtech_work_experience",
        "question": "How much work experience is required for Executive, Part-time, External or Sponsored M.Tech?",
        "positive_markers": ("m.tech", "executive", "part-time", "external", "sponsored", "work experience", "2 years"),
        "negative_markers": ("m.sc", "jam", "ph.d"),
        "expected_source_hints": ("mtech", "admission"),
    },
    {
        "id": "msc_eligibility",
        "question": "What are the eligibility requirements for M.Sc admission?",
        "positive_markers": ("m.sc", "eligibility", "jam", "bachelor", "60%", "6.0", "55%", "5.5"),
        "negative_markers": ("m.tech", "gate", "ph.d"),
        "expected_source_hints": ("msc", "admission"),
    },
    {
        "id": "msc_percentage",
        "question": "What percentage or CGPA is needed for M.Sc admission?",
        "positive_markers": ("m.sc", "60%", "6.0", "55%", "5.5"),
        "negative_markers": ("m.tech", "gate", "ph.d"),
        "expected_source_hints": ("msc",),
    },
    {
        "id": "msc_jam",
        "question": "Is JAM required for M.Sc admission?",
        "positive_markers": ("m.sc", "jam", "admission"),
        "negative_markers": ("m.tech", "gate", "ph.d"),
        "expected_source_hints": ("msc", "admission"),
    },
    {
        "id": "msc_bachelor_route",
        "question": "Can I apply for M.Sc through a bachelor's degree route with a test or interview?",
        "positive_markers": ("m.sc", "bachelor", "written test", "interview"),
        "negative_markers": ("m.tech", "gate", "ph.d"),
        "expected_source_hints": ("msc", "admission"),
    },
    {
        "id": "phd_eligibility",
        "question": "What are the Ph.D. eligibility requirements?",
        "positive_markers": ("ph.d", "eligibility", "master", "bachelor", "admission"),
        "negative_markers": ("m.tech", "m.sc", "gate", "jam"),
        "expected_source_hints": ("phd", "admission"),
    },
    {
        "id": "phd_application_fee",
        "question": "What is the Ph.D. application processing fee?",
        "positive_markers": ("ph.d", "application", "processing fee", "fee"),
        "negative_markers": ("m.tech", "m.sc", "jam"),
        "expected_source_hints": ("general", "admission"),
    },
    {
        "id": "mba_steps",
        "question": "What are the steps for MBA admission?",
        "positive_markers": ("mba", "admission", "application", "selection"),
        "negative_markers": ("m.tech", "gate", "jam", "ph.d"),
        "expected_source_hints": ("mba", "admission"),
    },
    {
        "id": "registration_documents",
        "question": "What documents do I need to bring for registration?",
        "positive_markers": ("registration", "documents", "bring", "certificate", "identity"),
        "negative_markers": ("gate", "jam", "m.tech eligibility"),
        "expected_source_hints": ("registration",),
    },
    {
        "id": "registration_provisional",
        "question": "What happens during provisional registration?",
        "positive_markers": ("provisional registration", "certificate", "qualifying examination", "registration"),
        "negative_markers": ("jam", "gate", "mba"),
        "expected_source_hints": ("registration",),
    },
    {
        "id": "registration_late",
        "question": "What happens if I register late?",
        "positive_markers": ("late registration", "registration", "late", "academic calendar"),
        "negative_markers": ("jam", "gate", "mba"),
        "expected_source_hints": ("registration",),
    },
    {
        "id": "summer_courses",
        "question": "What are the rules for summer courses for M.Tech students?",
        "positive_markers": ("summer", "course", "m.tech", "registration"),
        "negative_markers": ("m.sc", "jam", "mba"),
        "expected_source_hints": ("registration",),
    },
    {
        "id": "ee_mtech_programs",
        "question": "Which M.Tech programs are offered in Electrical Engineering?",
        "positive_markers": ("m.tech", "power systems", "power electronics", "cyber-physical systems", "electrical engineering"),
        "negative_markers": ("m.sc", "ph.d", "mba"),
        "expected_source_hints": ("overview", "electrical"),
    },
    {
        "id": "ee_btech",
        "question": "Does Electrical Engineering offer a B.Tech program?",
        "positive_markers": ("b.tech", "electrical engineering"),
        "negative_markers": ("m.sc", "mba", "ph.d"),
        "expected_source_hints": ("overview", "electrical"),
    },
    {
        "id": "ee_research",
        "question": "What research areas are available in Electrical Engineering?",
        "positive_markers": ("research", "power systems", "smart grids", "renewable energy", "high-voltage", "robotics"),
        "negative_markers": ("m.sc admission", "mba", "jam"),
        "expected_source_hints": ("overview", "electrical"),
    },
    {
        "id": "ee_smart_grid_lab",
        "question": "Does Electrical Engineering have a Smart Grid Laboratory?",
        "positive_markers": ("smart grid", "laboratory", "electrical engineering"),
        "negative_markers": ("mba", "jam", "gate"),
        "expected_source_hints": ("facilities", "smart"),
    },
    {
        "id": "ee_faculty",
        "question": "Who are some faculty members in Electrical Engineering?",
        "positive_markers": ("electrical engineering", "faculty"),
        "negative_markers": ("m.sc", "mba", "jam"),
        "expected_source_hints": ("faculty", "electrical"),
    },
    {
        "id": "mtech_vs_msc",
        "question": "What is the difference between M.Tech and M.Sc admission eligibility at IIT Jodhpur?",
        "positive_markers": ("m.tech", "m.sc", "gate", "jam", "eligibility"),
        "negative_markers": ("mba", "ph.d"),
        "expected_source_hints": ("mtech", "msc", "admission"),
    },
    {
        "id": "mtech_vs_phd",
        "question": "How does M.Tech eligibility differ from Ph.D. eligibility?",
        "positive_markers": ("m.tech", "ph.d", "eligibility"),
        "negative_markers": ("m.sc", "mba", "jam"),
        "expected_source_hints": ("mtech", "phd", "admission"),
    },
    {
        "id": "mtech_hinglish",
        "question": "M.Tech ke eligibility requirements kya hain?",
        "positive_markers": ("m.tech", "eligibility", "bachelor", "gate"),
        "negative_markers": ("m.sc", "jam", "ph.d"),
        "expected_source_hints": ("mtech", "admission"),
    },
    {
        "id": "msc_hinglish",
        "question": "M.Sc admission ke liye eligibility kya hai?",
        "positive_markers": ("m.sc", "eligibility", "jam", "bachelor"),
        "negative_markers": ("m.tech", "gate", "ph.d"),
        "expected_source_hints": ("msc", "admission"),
    },
    {
        "id": "phd_hinglish",
        "question": "Ph.D. admission ke liye eligibility kya chahiye?",
        "positive_markers": ("ph.d", "eligibility", "admission", "master", "bachelor"),
        "negative_markers": ("m.tech", "m.sc", "jam"),
        "expected_source_hints": ("phd", "admission"),
    },
    {
        "id": "ee_hinglish",
        "question": "Electrical Engineering mein kaunse M.Tech programs hain?",
        "positive_markers": ("electrical engineering", "m.tech", "power systems", "power electronics", "cyber-physical"),
        "negative_markers": ("m.sc", "mba", "jam"),
        "expected_source_hints": ("overview", "electrical"),
    },
    {
        "id": "gate_vs_jam",
        "question": "For postgraduate admission, when is GATE used and when is JAM used?",
        "positive_markers": ("gate", "jam", "m.tech", "m.sc"),
        "negative_markers": ("ph.d", "mba"),
        "expected_source_hints": ("mtech", "msc", "admission"),
    },
    {
        "id": "mtech_gate_hinglish",
        "question": "M.Tech admission ke liye GATE compulsory hai kya?",
        "positive_markers": ("m.tech", "gate", "admission"),
        "negative_markers": ("m.sc", "jam", "ph.d", "mba"),
        "expected_source_hints": ("mtech", "admission"),
    },
]


def get_cases():
    return list(CASES)


def get_case(case_id):
    for case in CASES:
        if case["id"] == case_id:
            return dict(case)
    raise KeyError(case_id)
