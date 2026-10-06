from tests_reusable.test_diagnosis.run_diagnosis import run_case
from tests_reusable.test_diagnosis.cases.benchmark import get_cases
def test_six_cases_trace():
    results=[run_case(c) for c in get_cases()[:6]]; assert len(results)==6
