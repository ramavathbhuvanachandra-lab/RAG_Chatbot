from tests_reusable.test_diagnosis.run_diagnosis import run_case
from tests_reusable.test_diagnosis.cases.benchmark import get_cases
def test_hybrid_stage_runs():
    for c in get_cases()[:6]:
        s=run_case(c)['stages'].get('hybrid_retrieve_node',{}); assert 'error' not in s
