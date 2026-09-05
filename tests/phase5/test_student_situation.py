"""
Phase 5.1 — Student Situation Understanding

Aggressive behavioral tests.

The tests intentionally cover:
- ordinary student language
- personal situations
- implicit goals
- explicit constraints
- numbers
- academic background
- hostel scenarios
- research scenarios
- decision scenarios
- ambiguous situations
- irrelevant/personal wording
- navigation/emergency situations
- negative cases
- serialization stability

These tests validate the interpretation contract, not college policy.
"""

from backend.student_situation import (
    situation_to_dict,
    understand_student_situation,
)


# =========================================================
# Basic situation understanding
# =========================================================

def test_personal_phd_eligibility_situation():
    result = understand_student_situation(
        "I have a four-year B.Tech with 72% and want to apply for a regular Ph.D."
    )

    assert result.intent == "phd_eligibility"
    assert result.goal == "determine_eligibility"

    assert result.user_facts["degree"] == "bachelors_degree"
    assert result.user_facts["degree_duration_years"] == 4
    assert 72.0 in result.user_facts["percentages"]

    assert "phd" in result.entities
    assert result.confidence > 0.7


def test_master_degree_eligibility_situation():
    result = understand_student_situation(
        "I have a master's degree but I'm unsure whether my marks meet the Ph.D. requirement."
    )

    assert result.intent == "phd_eligibility"
    assert result.goal == "determine_eligibility"
    assert result.user_facts["degree"] == "masters_degree"


def test_category_is_preserved():
    result = understand_student_situation(
        "I'm in the SC category and checking whether I qualify for Ph.D. admission."
    )

    assert result.intent == "phd_eligibility"
    assert result.goal == "determine_eligibility"
    assert result.user_facts["category"] == "SC"


def test_employment_fact_is_preserved():
    result = understand_student_situation(
        "I work full-time and want to apply for a regular Ph.D."
    )

    assert result.intent == "phd_eligibility"
    assert result.user_facts["employment_status"] == "full_time"


# =========================================================
# Hostel situations
# =========================================================

def test_hostel_cost_saving_situation():
    result = understand_student_situation(
        "I'll be staying for 20 days and I'm comfortable sharing a room if it reduces the cost."
    )

    assert result.intent == "hostel"
    assert result.goal == "minimize_cost"

    assert result.constraints["occupancy"] == "double"

    assert result.constraints["stay_duration"] == {
        "value": 20.0,
        "unit": "days",
    }

    assert result.constraints["cost_sensitive"] is True


def test_single_room_hostel_situation():
    result = understand_student_situation(
        "I need a room for myself for 5 days and don't need bedding."
    )

    assert result.intent == "hostel"
    assert result.constraints["occupancy"] == "single"
    assert result.constraints["bedding"] == "without"

    assert result.constraints["stay_duration"] == {
        "value": 5.0,
        "unit": "days",
    }


def test_hostel_bedding_requirement():
    result = understand_student_situation(
        "I need bedding along with the room for my stay."
    )

    assert result.intent == "hostel"
    assert result.constraints["bedding"] == "with"


def test_hostel_budget_is_extracted():
    result = understand_student_situation(
        "I need hostel accommodation and want to keep the cost under ₹3000."
    )

    assert result.intent == "hostel"
    assert result.constraints["budget"] == 3000.0


def test_long_hostel_stay():
    result = understand_student_situation(
        "I'll probably stay for three months, so I need to understand the accommodation cost."
    )

    assert result.intent == "hostel"

    assert result.constraints["stay_duration"] == {
        "value": 3.0,
        "unit": "months",
    }


# =========================================================
# Research situations
# =========================================================

def test_robotics_control_research_situation():
    result = understand_student_situation(
        "My interest is robotics and control systems, and I'm considering Electrical Engineering."
    )

    assert result.intent == "research"
    assert result.goal == "find_relevant_research"

    assert (
        "robotics"
        in result.constraints["research_interests"]
    )

    assert (
        "control systems"
        in result.constraints["research_interests"]
    )

    assert "electrical engineering" in result.entities


def test_vlsi_research_situation():
    result = understand_student_situation(
        "I want to work on VLSI during higher studies and I'm looking for a relevant research direction."
    )

    assert result.intent == "research"
    assert result.goal == "find_relevant_research"
    assert "vlsi" in result.constraints["research_interests"]


def test_visual_computing_interest():
    result = understand_student_situation(
        "My research interest is visual computing, and I'm deciding whether Electrical Engineering fits that goal."
    )

    assert result.intent == "research"
    assert result.goal in {
        "find_relevant_research",
        "choose_option",
    }

    assert (
        "visual computing"
        in result.constraints["research_interests"]
    )


def test_power_renewable_interest():
    result = understand_student_situation(
        "I want to work on power systems and renewable energy during higher studies."
    )

    assert result.intent == "research"

    interests = (
        result.constraints["research_interests"]
    )

    assert "power systems" in interests
    assert "renewable energy" in interests


# =========================================================
# Decision / comparison situations
# =========================================================

def test_mtech_vs_ms_decision_situation():
    result = understand_student_situation(
        "I'm deciding between M.Tech and M.S. by Research because I prefer research over a purely coursework-based path."
    )

    assert result.intent in {
        "programs",
        "research",
    }

    assert result.goal in {
        "compare_options",
        "choose_option",
    }

    assert result.constraints[
        "research_preference"
    ] is True


def test_interdisciplinary_preference():
    result = understand_student_situation(
        "I want an engineering program but I also care about interdisciplinary exposure."
    )

    assert result.intent == "programs"

    assert (
        result.constraints[
            "interdisciplinary_preference"
        ]
        is True
    )


def test_program_shortlisting_situation():
    result = understand_student_situation(
        "I'm shortlisting programs before applying and want options that fit an engineering background."
    )

    assert result.intent == "admission"
    assert result.goal == "plan_application"


# =========================================================
# Financial assistance situations
# =========================================================

def test_financial_assistance_is_distinct_from_admission():
    result = understand_student_situation(
        "I'm already checking my Ph.D. eligibility, but I also want to know whether I could receive financial assistance."
    )

    assert result.intent == "phd_eligibility"


def test_financial_assistance_question():
    result = understand_student_situation(
        "My qualification seems fine, but I'm mainly concerned about financial assistance for a Ph.D."
    )

    assert result.intent in {
        "phd_eligibility",
        "financial_assistance",
    }


# =========================================================
# Navigation / emergency
# =========================================================

def test_emergency_situation():
    result = understand_student_situation(
        "My friend got hurt near the academic block. Where should I take him?"
    )

    assert result.intent == "emergency"
    assert result.goal in {
        "locate_service",
        "get_contact",
    }


def test_navigation_situation():
    result = understand_student_situation(
        "I'm new on campus and need directions to the library."
    )

    assert result.intent == "navigation"


# =========================================================
# Natural conversational wording
# =========================================================

def test_natural_student_wording_without_question_prefix():
    result = understand_student_situation(
        "I haven't decided on a department yet and I'm trying to find an area that suits me."
    )

    assert result.intent in {
        "department_or_school",
        "research",
        "programs",
    }

    assert result.goal in {
        "choose_option",
        "find_relevant_research",
        "find_information",
    }


def test_suppose_style_question():
    result = understand_student_situation(
        "Suppose I finish my bachelor's and then become interested in research. What path could make sense?"
    )

    assert result.intent in {
        "research",
        "programs",
        "admission",
    }


def test_because_clause_is_preserved_as_signal():
    result = understand_student_situation(
        "I'm choosing a shared hostel room because I want to reduce the cost."
    )

    assert result.intent == "hostel"
    assert result.constraints["occupancy"] == "double"
    assert result.constraints["cost_sensitive"] is True
    assert "because" in result.signals


def test_before_applying_signal():
    result = understand_student_situation(
        "Before applying, I want to make sure my background matches the program requirements."
    )

    assert result.goal in {
        "plan_application",
        "determine_eligibility",
        "understand_requirements",
    }

    assert "before applying" in result.signals


# =========================================================
# Numeric robustness
# =========================================================

def test_multiple_percentages_are_preserved():
    result = understand_student_situation(
        "My score is 62%, while the published threshold I found is 60%."
    )

    assert result.user_facts["percentages"] == (
        62.0,
        60.0,
    )


def test_cgpa_is_extracted():
    result = understand_student_situation(
        "I have a 7.2/10 CGPA and I am considering Ph.D. admission."
    )

    assert result.intent == "phd_eligibility"
    assert 7.2 in result.user_facts["cgpa_values"]


def test_duration_units_are_preserved():
    result = understand_student_situation(
        "I may need accommodation for 3 weeks."
    )

    assert result.intent == "hostel"

    assert result.constraints["stay_duration"] == {
        "value": 3.0,
        "unit": "weeks",
    }


# =========================================================
# Unknown / conservative behavior
# =========================================================

def test_unknown_information_is_not_invented():
    result = understand_student_situation(
        "I'm thinking about applying next year."
    )

    assert result.intent in {
        "admission",
        "general_information",
    }

    assert result.user_facts == {}


def test_empty_input_is_safe():
    result = understand_student_situation("")

    assert result.intent == "general_information"
    assert result.goal == "find_information"
    assert result.confidence == 0.0
    assert result.requires_clarification is True


def test_whitespace_input_is_safe():
    result = understand_student_situation(
        "   "
    )

    assert result.intent == "general_information"
    assert result.requires_clarification is True


def test_unsupported_personal_detail_is_not_fabricated():
    result = understand_student_situation(
        "I'm Rahul, I'm 22, and I love cricket. I may apply here."
    )

    assert "age" not in result.user_facts
    assert "name" not in result.user_facts


# =========================================================
# Entity configuration
# =========================================================

def test_custom_entity_terms_are_supported():
    result = understand_student_situation(
        "I'm considering the School of Quantum Computing.",
        entity_terms=(
            "School of Quantum Computing",
        ),
    )

    assert (
        "School of Quantum Computing"
        in result.entities
    )


def test_entity_extraction_does_not_require_custom_config():
    result = understand_student_situation(
        "I'm interested in Electrical Engineering and VLSI."
    )

    assert "electrical engineering" in result.entities
    assert "vlsi" in result.entities


# =========================================================
# Clarification behavior
# =========================================================

def test_phd_eligibility_without_personal_facts_can_request_clarification():
    result = understand_student_situation(
        "Can I apply for a regular Ph.D.?"
    )

    assert result.intent == "phd_eligibility"
    assert result.goal == "determine_eligibility"
    assert result.requires_clarification is True


def test_hostel_cost_goal_without_constraints_can_request_clarification():
    result = understand_student_situation(
        "I want the cheapest hostel option."
    )

    assert result.intent == "hostel"
    assert result.goal == "minimize_cost"
    assert result.requires_clarification is True


def test_hostel_situation_with_constraints_needs_less_clarification():
    result = understand_student_situation(
        "I need a single room for 5 days and want the cost."
    )

    assert result.intent == "hostel"
    assert result.constraints["occupancy"] == "single"
    assert result.constraints["stay_duration"] == {
        "value": 5.0,
        "unit": "days",
    }


# =========================================================
# Serialization
# =========================================================

def test_situation_serialization_is_stable():
    result = understand_student_situation(
        "I have a four-year B.Tech with 72% and want regular Ph.D. admission."
    )

    payload = situation_to_dict(
        result
    )

    assert payload["raw_text"]
    assert payload["intent"] == "phd_eligibility"
    assert isinstance(
        payload["entities"],
        list,
    )
    assert isinstance(
        payload["signals"],
        list,
    )
    assert isinstance(
        payload["constraints"],
        dict,
    )
    assert isinstance(
        payload["user_facts"],
        dict,
    )


# =========================================================
# Regression-style real-world cases
# =========================================================

def test_student_wants_research_and_program_connection():
    result = understand_student_situation(
        "I'm interested in control systems and I also want to know which degree path connects to that interest."
    )

    assert result.intent in {
        "research",
        "programs",
    }

    assert (
        "control systems"
        in result.constraints["research_interests"]
    )


def test_student_has_nonmatching_background():
    result = understand_student_situation(
        "My bachelor's is in a different field, but I want to apply for M.Tech. Can my background still work?"
    )

    assert result.intent == "mtech_eligibility"
    assert result.goal == "determine_eligibility"
    assert result.user_facts["degree"] == "bachelors_degree"


def test_student_is_comparing_hostel_categories():
    result = understand_student_situation(
        "I'm comparing single and double occupancy because I haven't decided whether to share a room."
    )

    assert result.intent == "hostel"
    assert result.goal == "compare_options"


def test_student_wants_research_before_supervisor_contact():
    result = understand_student_situation(
        "I want to identify a research area before approaching a potential supervisor."
    )

    assert result.intent == "research"
    assert result.goal == "find_relevant_research"


def test_personal_application_decision():
    result = understand_student_situation(
        "I don't want to apply to a program unless my qualification clearly matches the published criteria."
    )

    assert result.goal in {
        "determine_eligibility",
        "plan_application",
        "understand_requirements",
    }


def test_future_cycle_is_not_treated_as_known_fact():
    result = understand_student_situation(
        "I'm planning for 2028 and want to know whether I should prepare for admission."
    )

    assert result.intent == "admission"
    assert "2028" not in result.user_facts


def test_irrelevant_hobby_does_not_change_academic_intent():
    result = understand_student_situation(
        "I play football every weekend, but my main concern is whether my B.Tech qualifies me for M.Tech."
    )

    assert result.intent == "mtech_eligibility"
    assert result.goal == "determine_eligibility"
    assert result.user_facts["degree"] == "bachelors_degree"
