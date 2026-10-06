from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules
def test_verification_api():
    m=load_ai_modules()['verification']; assert all(hasattr(m,n) for n in ('verify_candidate','verify_candidates','select_verified_candidates','explain_verification'))
