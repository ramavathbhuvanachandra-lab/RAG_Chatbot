"""
Production Regression Questions — Phase 5 Baseline

Purpose
-------
A fixed 102-question production benchmark for the IIT Jodhpur chatbot.

This is a diagnostic evaluation set.

It deliberately contains:
- normal factual questions
- paraphrases
- broad-list questions
- quantitative questions
- temporal questions
- eligibility / admission questions
- program questions
- department / school questions
- research questions
- hostel / accommodation questions
- finance questions
- facilities questions
- multi-intent questions
- contextual/anaphoric questions
- unsupported questions
- scope-conflict questions
- adversarial questions
- ambiguity tests

The benchmark is intentionally mixed rather than grouped by topic so
the production graph is tested under realistic question ordering.

Important:
-----------
Do not add expected answers here.

This file measures actual production behavior.
"""

REGRESSION_QUESTIONS = [

    # =========================================================
    # 01-12 — Institution / academic baseline
    # =========================================================

    "What kind of institute is IIT Jodhpur?",

    "What programs are offered at IIT Jodhpur?",

    "Which academic departments are available at IIT Jodhpur?",

    "Which schools are there at IIT Jodhpur?",

    "What academic programs are available at IIT Jodhpur?",

    "What degrees can students pursue at IIT Jodhpur?",

    "What research areas are available at IIT Jodhpur?",

    "What facilities are available for students at IIT Jodhpur?",

    "What hostel facilities are available for students?",

    "What accommodation options are available in the hostel?",

    "What centres are involved in academic or research programs?",

    "What are the main academic areas covered by the institute?",


    # =========================================================
    # 13-24 — Undergraduate / postgraduate programs
    # =========================================================

    "Which B.Tech programs are available at IIT Jodhpur?",

    "Which M.Tech specializations are offered?",

    "What M.Sc. programs are available?",

    "Does IIT Jodhpur offer an M.S. by Research program?",

    "What kinds of Ph.D. programs are available?",

    "Which departments offer undergraduate academic programs?",

    "Which schools offer academic programs?",

    "What programs are available through the School of Artificial Intelligence and Data Science?",

    "What programs are offered by the School of Management and Entrepreneurship?",

    "What undergraduate programs are available in engineering?",

    "What postgraduate programs are available?",

    "Are research-oriented degree programs available?",


    # =========================================================
    # 25-38 — Ph.D. admissions / eligibility
    # =========================================================

    "What are the eligibility requirements for regular Ph.D. admission?",

    "What are the requirements for part-time Ph.D. admission?",

    "Is a master's degree mandatory for every regular Ph.D. route?",

    "What qualifications can be used instead of a conventional master's degree for Ph.D. admission?",

    "Is GATE required for regular Ph.D. admission?",

    "When is GATE exempted for Ph.D. applicants?",

    "What are the eligibility requirements for Ph.D. admission through a four-year bachelor's degree?",

    "Can someone with a four-year bachelor's degree apply for a regular Ph.D.?",

    "What is the minimum CGPA required for regular Ph.D. admission?",

    "What percentage is required for regular Ph.D. admission?",

    "What are the category-wise percentage requirements for Ph.D. admission?",

    "Are there restrictions on employment or concurrent enrollment for regular Ph.D. students?",

    "Can a B.Tech graduate get financial assistance for a Ph.D.?",

    "What are the admission routes for regular Ph.D. candidates?",


    # =========================================================
    # 39-48 — Other admissions / qualification questions
    # =========================================================

    "What are the eligibility requirements for regular M.Tech. admission?",

    "What qualifications are required for M.Tech. admission?",

    "What is the minimum percentage required for M.Tech. admission?",

    "What is the minimum CGPA required for M.Tech. admission?",

    "What are the eligibility requirements for M.Sc. admission?",

    "What qualifications are required for undergraduate admission?",

    "How are B.Tech admissions conducted?",

    "Is JEE used for B.Tech admission?",

    "What is the admission process for postgraduate programs?",

    "What is the application fee for admission?",


    # =========================================================
    # 49-60 — Electrical Engineering research
    # =========================================================

    "What research areas are available in Electrical Engineering?",

    "What research themes are covered by the Electrical Engineering department?",

    "What kind of work is being done in control systems in Electrical Engineering?",

    "What research is being done in power systems and power electronics?",

    "What research areas involve robotics and control systems?",

    "What research areas are related to VLSI?",

    "What research is being done in RF, microwaves, and photonics?",

    "What research is being done in communication systems?",

    "What research areas involve signal processing?",

    "What research is being done in device simulation?",

    "What research is being done in organic and flexible electronics?",

    "What areas of Electrical Engineering research involve visual computing?",


    # =========================================================
    # 61-70 — Other research / department questions
    # =========================================================

    "What research groups are available in the Physics department?",

    "What research areas are covered in Physics?",

    "What academic programs are offered by the Physics department?",

    "What research facilities are available across the institute?",

    "What kinds of research laboratories are available?",

    "Which departments are active in interdisciplinary research?",

    "Which schools are involved in research activities?",

    "What research themes are available outside Electrical Engineering?",

    "What research facilities are available for students?",

    "What is being done in advanced technology and prototyping research?",


    # =========================================================
    # 71-80 — Hostel / accommodation / finance
    # =========================================================

    "What are the hostel fees?",

    "What is the daily hostel accommodation charge?",

    "How much does short-term hostel accommodation cost?",

    "How much does long-term hostel accommodation cost?",

    "What is the difference between single and double hostel accommodation?",

    "What are the hostel charges for students?",

    "What are the hostel charges for visitors or executive students?",

    "What are the hostel charges with bedding?",

    "What are the hostel charges without bedding?",

    "Does the semester fee include hostel fees?",


    # =========================================================
    # 81-86 — Facilities / services / accommodation scope
    # =========================================================

    "What facilities are available in the hostel?",

    "What dining options are available in the hostel?",

    "How does the hostel booking process work?",

    "Who can use the hostel accommodation facilities?",

    "How do hostel charges vary by duration of stay?",

    "Are single rooms available in the hostel?",


    # =========================================================
    # 87-94 — Temporal / numeric / adversarial evidence
    # =========================================================

    "What is the latest hostel fee for 2027?",

    "What is the latest hostel fee for 2028?",

    "What is the daily hostel fee if the published rate is monthly?",

    "What was the hostel fee for AY 2026-2027?",

    "What is the current hostel fee?",

    "Can a monthly hostel fee be converted into a daily fee?",

    "What is the exact salary of every professor at IIT Jodhpur?",

    "Which professor is the best researcher in Electrical Engineering?",


    # =========================================================
    # 95-102 — Multi-intent / scope / unsupported / adversarial
    # =========================================================

    "What are the Ph.D. eligibility requirements and the hostel fees?",

    "What are the hostel facilities and accommodation charges?",

    "What research areas are available in Electrical Engineering and what programs are offered there?",

    "What programs are available through the School of Artificial Intelligence and Data Science and what research is done there?",

    "Is the financial-assistance rule the same thing as Ph.D. admission eligibility?",

    "Does the School of AI and Data Science Ph.D. eligibility automatically apply to every Ph.D. program at IIT Jodhpur?",

    "Which department has the best placements at IIT Jodhpur?",

    "What programs will IIT Jodhpur definitely offer in 2028?",
]
