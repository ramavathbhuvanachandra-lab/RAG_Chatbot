from tests_reusable.test_diagnosis.cases.benchmark import get_cases
from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules,make_initial_state,run_node
def test_cases(): assert len(get_cases())>=6
def test_query_node_runs():
    node=load_ai_modules()['nodes'].understand_query_node
    for c in get_cases()[:6]:
        st=run_node(node,make_initial_state(c['question'])); assert isinstance(st,dict)
