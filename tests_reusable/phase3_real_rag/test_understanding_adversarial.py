from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.query.understanding import understand_query

class FakeSemantic:
    def __init__(self, response):
        self.response=response
    def invoke(self, payload):
        return self.response

class FakeSafety:
    def invoke(self, payload):
        return SimpleNamespace(approved=True, reason='ok')

def run(q, response):
    return understand_query(q, semantic_model=FakeSemantic(response), safety_model=FakeSafety())

def base(**kw):
    d={'target':None,'request_type':None,'semantic_query':'','confidence':0.0}
    d.update(kw)
    return d

def test_mtech_compound_target_is_decomposed():
    r=run('What are the M.Tech eligibility requirements?', base(target='M.Tech eligibility requirements'))
    assert r.target.text == 'M.Tech'
    assert r.request_type == 'eligibility'
    assert r.retrieval_requirement.mode == 'exact'

def test_mtech_compound_target_criteria_is_decomposed():
    r=run('What are the M.Tech eligibility criteria?', base(target='M.Tech eligibility criteria'))
    assert r.target.text == 'M.Tech'
    assert r.request_type == 'eligibility'

def test_phd_compound_target_is_decomposed():
    r=run('What are the Ph.D. admission requirements?', base(target='Ph.D. admission requirements'))
    assert r.target.text == 'Ph.D'
    assert r.request_type == 'eligibility'

def test_btech_compound_target_is_decomposed():
    r=run('What are the B.Tech admission requirements?', base(target='B.Tech admission requirements'))
    assert r.target.text == 'B.Tech'
    assert r.request_type == 'eligibility'

def test_msc_routes_target_recovers_entity_after_for():
    r=run('What are the admission routes for M.Sc.?', base(target='admission routes',request_type='admission'))
    assert r.target.text == 'M.Sc'
    assert r.request_type == 'admission'

def test_msc_routes_with_tell_me():
    r=run('Tell me about admission routes for M.Sc.', base(target='admission routes',request_type='admission'))
    assert r.target.text == 'M.Sc'

def test_btech_requirements_entity_after_for():
    r=run('What are the admission requirements for B.Tech?', base(target='admission requirements',request_type='eligibility'))
    assert r.target.text == 'B.Tech'

def test_procedure_phrase_overrides_generic_application_label():
    r=run('What is the procedure for applying?', base(target='',request_type='application'))
    assert r.request_type == 'procedure'

def test_how_apply_overrides_application_label():
    r=run('How do I apply for admission?', base(target='admission',request_type='application'))
    assert r.request_type == 'procedure'

def test_admission_process_is_procedure():
    r=run('What is the admission process?', base(target='admission',request_type='admission'))
    assert r.request_type == 'procedure'

def test_mtech_registration_stays_targeted():
    r=run('M.Tech registration kaise hota hai?', base(target='M.Tech',request_type='registration'))
    assert r.target.text == 'M.Tech'
    assert r.request_type == 'registration'

def test_hostel_fee_target_is_not_collapsed():
    r=run('What are the hostel fees for students?', base(target='hostel fees',request_type='cost'))
    assert r.target.text == 'hostel fees'
    assert r.request_type == 'cost'

def test_tuition_fee_target_can_remain_literal():
    r=run('What is the tuition fee?', base(target='tuition fee',request_type='information'))
    assert r.request_type == 'cost'

def test_admission_fee_remains_distinct():
    r=run('What is the admission fee?', base(target='admission',request_type='information'))
    assert r.target.text == 'admission fee'
    assert r.request_type == 'cost'

def test_application_deadlines_are_not_list():
    r=run('What are the application deadlines?', base(target='application deadlines',request_type='information'))
    assert r.target.text == 'application deadlines'
    assert r.list_intent.is_list is False

def test_department_collection_remains_list():
    r=run('What departments are there?', base(target='departments',request_type='information'))
    assert r.target.text == 'departments'
    assert r.list_intent.is_list is True

def test_program_collection_remains_list():
    r=run('What academic programs are available?', base(target='academic programs',request_type='information'))
    assert r.target.text == 'academic programs'
    assert r.list_intent.is_list is True

def test_facility_collection_remains_list():
    r=run('What facilities are available?', base(target='facilities',request_type='information'))
    assert r.target.text == 'facilities'
    assert r.list_intent.is_list is True

def test_hostel_facility_collection_remains_list():
    r=run('What hostel facilities are available?', base(target='hostel facilities',request_type='information'))
    assert r.target.text == 'hostel facilities'
    assert r.list_intent.is_list is True

def test_research_collection_remains_list():
    r=run('What research areas are available?', base(target='research areas',request_type='information'))
    assert r.request_type == 'research'
    assert r.list_intent.is_list is True

def test_research_opportunities_collection_remains_list():
    r=run('What research opportunities are available?', base(target='research opportunities',request_type='information'))
    assert r.request_type == 'research'
    assert r.list_intent.is_list is True

def test_specific_target_not_replaced_by_generic_literal():
    r=run('What is the tuition fee for Program Alpha?', base(target='Program Alpha',request_type='cost'))
    assert r.target.text == 'Program Alpha'

def test_unknown_model_target_is_rejected_by_grounding():
    r=run('What departments are there?', base(target='zorpulax',request_type='information',semantic_query='zorpulax',confidence=.99))
    assert r.target.text == 'departments'
    assert 'zorpulax' not in r.search_query

def test_target_suffix_stripping_requires_query_support():
    r=run('Tell me about Program Alpha.', base(target='Program Alpha admission requirements',request_type='information'))
    assert r.target.text == 'Program Alpha'

def test_no_target_invented_from_generic_for_students():
    r=run('What are the requirements for students?', base(target='requirements'))
    assert r.target.text == 'requirements'

def test_no_target_invented_from_the_institute():
    r=run('What is the process for the institute?', base(target='process'))
    assert r.target.text == 'process'

def test_hinglish_mtech_preserved():
    r=run('M.Tech ka registration kaise hota hai?', base(target='M.Tech',request_type='registration'))
    assert r.target.text == 'M.Tech'
    assert r.request_type == 'registration'

def test_hinglish_hostel_fee_preserved():
    r=run('students ke hostel fees kitne hain?', base(target='hostel fees',request_type='cost'))
    assert r.target.text == 'hostel fees'
    assert r.request_type == 'cost'

def test_msc_routes_entity_is_not_phantom_attribute():
    r=run('Which admission routes are available for M.Sc.?', base(target='admission routes',request_type='admission'))
    assert r.target.text == 'M.Sc'

def test_phd_target_with_parenthetical_suffix():
    r=run('What are the Ph.D. (regular) admission requirements?', base(target='Ph.D. admission requirements',request_type='eligibility'))
    assert r.target.text == 'Ph.D'

def test_target_attribute_separation_keeps_request_attribute():
    r=run('What are the M.Tech eligibility requirements?', base(target='M.Tech eligibility requirements'))
    attrs=[f.value for f in getattr(r,'qualifiers',())]
    assert r.request_type == 'eligibility'
    assert r.retrieval_requirement.require_attribute_alignment is True

def test_confidence_floor_for_deterministic_structure():
    r=run('What are the M.Tech eligibility requirements?', base(target='M.Tech eligibility requirements'))
    assert r.confidence >= .92

def test_empty_query_still_rejected():
    try:
        understand_query('   ')
    except ValueError:
        return
    raise AssertionError('empty query must fail')