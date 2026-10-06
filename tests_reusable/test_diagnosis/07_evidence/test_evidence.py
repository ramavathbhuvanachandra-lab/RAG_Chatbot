from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules
def test_evidence_apis():
    m=load_ai_modules(); assert hasattr(m['evidence'],'assess_evidence'); assert hasattr(m['coverage'],'assess_coverage'); assert hasattr(m['claims'],'audit_claims'); assert hasattr(m['packaging'],'build_evidence_package')
