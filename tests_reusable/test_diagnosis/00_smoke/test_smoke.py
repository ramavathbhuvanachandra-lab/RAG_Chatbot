from tests_reusable.test_diagnosis.common.diagnostics import load_ai_modules,graph_nodes,backend_imports
def test_core_imports(): assert load_ai_modules()
def test_graph_contract():
    names={n for n,_ in graph_nodes(load_ai_modules()['nodes'])}
    assert {'understand_query_node','hybrid_retrieve_node','fuse_retrieved_documents_node','verify_and_rank_node','evidence_context_node','assess_evidence_node','assess_coverage_node','audit_claims_node','package_evidence_node'} <= names
def test_no_backend_imports(): assert backend_imports()==[]
