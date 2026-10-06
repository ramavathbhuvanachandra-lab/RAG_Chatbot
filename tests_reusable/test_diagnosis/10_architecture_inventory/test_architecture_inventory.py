from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules,backend_imports
def test_expected_modules():
    m=load_ai_modules(); assert all(k in m for k in ('understanding','hybrid','rrf','ranking','semantic_alignment','verification','evidence','coverage','claims','packaging','nodes'))
def test_core_has_no_backend_imports(): assert backend_imports()==[]
