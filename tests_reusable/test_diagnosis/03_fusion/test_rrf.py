from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules
def test_rrf_api():
    m=load_ai_modules()['rrf']; assert hasattr(m,'reciprocal_rank_fusion'); assert hasattr(m,'fuse_ranked_lists')
