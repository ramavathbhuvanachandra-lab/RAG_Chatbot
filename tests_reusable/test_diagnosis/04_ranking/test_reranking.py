from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules
def test_ranking_api():
    m=load_ai_modules()['ranking']; assert hasattr(m,'rank_candidates'); assert hasattr(m,'select_ranked_documents')
