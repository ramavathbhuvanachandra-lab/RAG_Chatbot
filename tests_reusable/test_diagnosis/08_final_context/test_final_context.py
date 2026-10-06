from tests_reusable.test_diagnosis.run_diagnosis import run_case
from tests_reusable.test_diagnosis.cases.benchmark import get_cases
def test_package_stage_runs():
    for c in get_cases()[:6]:
        s=run_case(c)['stages'].get('package_evidence_node',{}); assert 'error' not in s
