from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules
def test_alignment_api():
    m=load_ai_modules()['semantic_alignment']; assert all(hasattr(m,n) for n in ('build_query_meaning','build_document_meaning','align_query_to_document'))
