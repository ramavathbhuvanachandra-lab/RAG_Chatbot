from __future__ import annotations
import sys, types
from pathlib import Path

core_semantic = types.ModuleType("backend.core.semantic_registry")
class SemanticRegistry: pass
core_semantic.SemanticRegistry = SemanticRegistry
sys.modules["backend.core.semantic_registry"] = core_semantic

core_institution = types.ModuleType("backend.core.institution")
class InstitutionProfile:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def validate(self):
        assert self.institution_id and self.display_name
core_institution.InstitutionProfile = InstitutionProfile
sys.modules["backend.core.institution"] = core_institution

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ai_platform.institutions.iitj.lexical import canonical_terms, normalize_for_institution, expand_query, score_pair
from ai_platform.institutions.iitj.semantic_registry import SEMANTIC_REGISTRY
from ai_platform.institutions.iitj.source_policy import SOURCE_POLICY
from ai_platform.institutions.iitj.scope_policy import SCOPE_POLICY
from ai_platform.institutions.iitj.navigation import NAVIGATION_MODEL

def test_program_boundary():
    x = normalize_for_institution("M.Technique is not M.Tech.")
    assert "mtech" in canonical_terms("M.Technique is not M.Tech.")
    assert "mtech" not in canonical_terms("M.Technique is not M.Technique.")

def test_ms_research():
    assert "ms_by_research" in SEMANTIC_REGISTRY.detect_programs("M.S. by Research admission")

def test_semantics():
    d = SEMANTIC_REGISTRY.describe("What are the M.Tech eligibility requirements in Electrical Engineering?")
    assert "mtech" in d["programs"]
    assert "eligibility" in d["topics"]
    assert "electrical_engineering" in d["entities"]
    assert "eligibility" in d["attributes"]

def test_lexical_score():
    r = score_pair("M.Tech eligibility requirements", "M.Tech eligibility requirements are listed here.")
    assert 0.0 <= r.query_overlap <= 1.0
    assert r.matched_phrases

def test_expansion_bounded():
    q = "What are the M.Tech eligibility requirements?"
    e = expand_query(q)
    assert e[0] == q
    assert len(e) <= 3

def test_source_policy():
    path = "data/data_iitj/iitj_rag_v1_docs_production/admissions/mtech_admissions.docx"
    assert SOURCE_POLICY.classify_source(path) == ("admissions",)
    assert SOURCE_POLICY.source_alignment("eligibility", path) == 1.0

def test_scope_conflict():
    q = SCOPE_POLICY.detect("M.Tech eligibility")
    e = SCOPE_POLICY.detect("B.Tech eligibility")
    assert "program" in SCOPE_POLICY.conflicting_dimensions(q, e)

def test_navigation():
    assert "directions" in NAVIGATION_MODEL.detect("How do I reach the library?")

if __name__ == "__main__":
    tests = [v for k,v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests: t()
    print(f"IITJ INSTITUTION ADAPTER CONTRACT TESTS: PASS ({len(tests)} tests)")