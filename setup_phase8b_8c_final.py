from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
TESTS = ROOT / "tests" / "phase8"
REPORTS = ROOT / "reports" / "phase8"
NODES = BACKEND / "nodes.py"
BACKUP = BACKEND / "nodes.py.before_phase8b_8c_final"
BROAD = BACKEND / "broad_institutional_retrieval.py"
COVERAGE = BACKEND / "institutional_coverage.py"
TEST_FILE = TESTS / "test_phase8b_8c_final.py"

BROAD_SOURCE = r'''"""Generic broad institutional retrieval for Phase 8B."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BROAD_MAX_CANDIDATES = 12
BROAD_MAX_PER_SOURCE = 2
MIN_BROAD_CANDIDATE_SCORE = 1.0
BROAD_MAX_ADDITIONAL_QUERIES = 2

@dataclass(frozen=True)
class BroadScope:
    name: str
    query_markers: tuple[str, ...]
    source_categories: tuple[str, ...]
    index_source_tokens: tuple[str, ...]
    content_markers: tuple[str, ...]
    retrieval_formulations: tuple[str, ...]

BROAD_SCOPES = (
    BroadScope("programs", ("what programs", "which programs", "academic programs", "programs available", "what degrees", "degrees available"), ("/programs/", "/academics/", "/admissions/", "/schools/", "/departments/"), ("program", "programme", "degree", "academic", "overview", "index"), ("program", "programme", "degree", "bachelor", "master", "ph.d", "b.tech", "m.tech", "m.sc", "mba"), ("academic programs degrees offered institute overview", "programmes degrees academic offerings institute")),
    BroadScope("departments", ("what departments", "which departments", "departments available", "departments at", "academic departments", "what schools and departments", "schools and departments"), ("/departments/", "/schools/", "/academics/"), ("department", "departments", "school", "schools", "academic", "overview", "index"), ("department", "departments", "school", "schools", "academic unit", "academic units"), ("academic departments schools institute overview", "departments schools academic units institute")),
    BroadScope("research", ("what research", "research opportunities", "research areas available", "research areas at", "research opportunities at", "what research areas", "research themes available", "research groups available"), ("/research/", "/departments/", "/schools/", "/research_and_technology_facilities/"), ("research", "research_overview", "research-overview", "overview", "index"), ("research areas", "research themes", "research groups", "research opportunities", "faculty research", "research overview"), ("research areas themes institute overview", "research groups opportunities institute")),
    BroadScope("facilities", ("what facilities", "which facilities", "facilities available", "facilities at", "campus facilities", "what infrastructure", "campus amenities"), ("/facilities/", "/research_and_technology_facilities/", "/hostel_accommodation/"), ("facilities", "facility", "infrastructure", "amenities", "overview", "index"), ("facilities", "facility", "laboratories", "laboratory", "infrastructure", "amenities"), ("campus facilities laboratories infrastructure overview", "student research campus facilities amenities")),
    BroadScope("admissions", ("what admission opportunities", "admission opportunities", "admissions available", "admission options", "ways to get admission", "how can i get admission", "admission routes"), ("/admissions/", "/programs/", "/schools/", "/departments/", "/academics/"), ("admission", "admissions", "academic_admissions", "academic-admissions", "overview", "index"), ("admission", "admissions", "application", "eligibility", "entrance exam", "entrance examination"), ("admission routes programs institute overview", "admissions programs application routes institute")),
)

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("\\", "/")).strip()

def _source(document) -> str:
    metadata = getattr(document, "metadata", {})
    return _normalize(metadata.get("source", "")) if isinstance(metadata, dict) else ""

def _content(document) -> str:
    return _normalize(getattr(document, "page_content", ""))

def detect_broad_scope(query: str) -> BroadScope | None:
    q = _normalize(query)
    for scope in BROAD_SCOPES:
        if any(marker in q for marker in scope.query_markers):
            return scope
    return None

def is_broad_institutional_question(query: str) -> bool:
    return detect_broad_scope(query) is not None

def build_broad_retrieval_queries(query: str) -> list[str]:
    scope = detect_broad_scope(query)
    if scope is None:
        return []
    result = []
    for formulation in scope.retrieval_formulations:
        if not formulation.strip():
            continue
        if formulation.casefold() == query.strip().casefold():
            continue
        if formulation.casefold() in {x.casefold() for x in result}:
            continue
        result.append(formulation)
        if len(result) >= BROAD_MAX_ADDITIONAL_QUERIES:
            break
    return result

def _matches_source_category(source: str, scope: BroadScope) -> bool:
    return any(category in source for category in scope.source_categories)

def _looks_like_scope_index_source(source: str, scope: BroadScope) -> bool:
    stem = Path(source).stem.lower()
    for token in scope.index_source_tokens:
        if token in stem:
            if token == "index" and not _matches_source_category(source, scope):
                continue
            return True
    return False

def score_broad_candidate(query: str, document, scope: BroadScope | None = None) -> float:
    scope = scope or detect_broad_scope(query)
    if scope is None:
        return 0.0
    source = _source(document)
    content = _content(document)
    source_match = _matches_source_category(source, scope)
    index_source = _looks_like_scope_index_source(source, scope)
    content_hits = sum(1 for marker in scope.content_markers if marker in content)
    score = min(5.0, float(content_hits))
    if source_match:
        score += 4.0
    if index_source and source_match and content_hits:
        score += 8.0
    if any(x in source for x in ("/placements/", "/registration/", "/student_records/", "/finance/")) and not source_match and content_hits == 0:
        score -= 8.0
    if "source original source urls" in content:
        score -= 2.0
    if "retrieval representation" in content:
        score -= 2.0
    return score

def assemble_broad_candidates(query: str, documents: Iterable, max_candidates: int = BROAD_MAX_CANDIDATES) -> list:
    if detect_broad_scope(query) is None:
        return list(documents)[:max_candidates]
    ranked = []
    scope = detect_broad_scope(query)
    for rank, document in enumerate(documents, start=1):
        score = score_broad_candidate(query, document, scope)
        if score >= MIN_BROAD_CANDIDATE_SCORE:
            ranked.append((score, rank, document))
    ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    selected, source_counts = [], {}
    for score, rank, document in ranked:
        source = _source(document)
        count = source_counts.get(source, 0)
        if count >= BROAD_MAX_PER_SOURCE:
            continue
        selected.append(document)
        source_counts[source] = count + 1
        if len(selected) >= max_candidates:
            break
    return selected
'''

COVERAGE_SOURCE = r'''"""Generic institutional evidence coverage for Phase 8C."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
from backend.evidence_coverage import assess_evidence_coverage, detect_question_type

BROAD_SCOPES = {
    "programs": ("what programs", "which programs", "academic programs", "programs available", "what degrees", "degrees available"),
    "departments": ("what departments", "which departments", "departments available", "academic departments", "schools and departments"),
    "research": ("what research", "research opportunities", "research areas available", "what research areas", "research themes available", "research groups available"),
    "facilities": ("what facilities", "which facilities", "facilities available", "what infrastructure", "campus amenities"),
    "admissions": ("what admission opportunities", "admission opportunities", "admissions available", "admission options", "admission routes"),
}

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()

def _source(document: Any) -> str:
    metadata = getattr(document, "metadata", {})
    return _normalize(metadata.get("source", "")) if isinstance(metadata, dict) else ""

def _content(document: Any) -> str:
    return _normalize(getattr(document, "page_content", ""))

def detect_broad_scope(query: str) -> str | None:
    q = _normalize(query)
    for scope, markers in BROAD_SCOPES.items():
        if any(marker in q for marker in markers):
            return scope
    return None

def _looks_like_index(source: str, content: str, scope: str) -> bool:
    filename = Path(source).stem.lower()
    filename_terms = {
        "programs": ("program", "programme", "degree", "academic", "overview", "index"),
        "departments": ("department", "school", "academic", "overview", "index"),
        "research": ("research", "overview", "index"),
        "facilities": ("facility", "facilities", "infrastructure", "amenities", "overview", "index"),
        "admissions": ("admission", "academic", "overview", "index"),
    }
    return any(term in filename for term in filename_terms.get(scope, ())) or any(marker in content for marker in {
        "programs": ("academic programs", "programmes offered", "degrees offered"),
        "departments": ("academic departments", "departments and schools"),
        "research": ("research overview", "research areas", "research themes"),
        "facilities": ("campus facilities", "facilities overview"),
        "admissions": ("admission opportunities", "admission routes", "admissions overview"),
    }.get(scope, ()))

def _breadth(content: str, scope: str) -> int:
    patterns = {
        "programs": (r"\bb\.?tech\b", r"\bm\.?tech\b", r"\bm\.?sc\b", r"\bmba\b", r"\bph\.?d\b", r"\bbachelor", r"\bmaster", r"\bdegree"),
        "departments": (r"\bdepartment\b", r"\bdepartments\b", r"\bschool\b", r"\bschools\b", r"\bcentre\b", r"\bcenter\b"),
        "research": (r"\bresearch area", r"\bresearch theme", r"\bresearch group", r"\bresearch laboratory"),
        "facilities": (r"\bfacilit(?:y|ies)\b", r"\blaborator(?:y|ies)\b", r"\binfrastructure\b", r"\bamenit(?:y|ies)\b"),
        "admissions": (r"\badmission\b", r"\badmissions\b", r"\bapplication\b", r"\bentrance\b", r"\bjam\b", r"\bgate\b", r"\bnet\b"),
    }
    return min(12, sum(len(re.findall(pattern, content)) for pattern in patterns.get(scope, ())))

def assess_institutional_coverage(query: str, documents) -> dict[str, Any]:
    documents = list(documents or [])
    base = assess_evidence_coverage(query=query, documents=documents)
    scope = detect_broad_scope(query)
    question_type = base.get("question_type", detect_question_type(query))
    if scope is None:
        return base

    # Broad institutional questions are treated as landscape/list questions
    # for coverage even if the baseline classifier calls them descriptive.
    if scope:
        question_type = "list"
    useful = [d for d in documents if _content(d)]
    sources = {_source(d) for d in useful if _source(d)}
    combined = "\n".join(_content(d) for d in useful)
    index_count = sum(_looks_like_index(_source(d), _content(d), scope) for d in useful)
    breadth = _breadth(combined, scope)
    if index_count >= 1 and breadth >= 3:
        status = "supported"
    elif len(sources) >= 2 and breadth >= 4:
        status = "supported"
    elif breadth >= 2 or index_count >= 1 or len(sources) >= 2 or useful:
        status = "partially_supported"
    else:
        status = "insufficient"
    result = dict(base)
    result.update({"status": status, "question_type": question_type, "institutional_coverage_scope": scope, "institutional_coverage_sources": len(sources), "institutional_coverage_index_documents": index_count, "institutional_coverage_breadth": breadth})
    return result
'''

TEST_SOURCE = r'''from langchain_core.documents import Document
from backend.broad_institutional_retrieval import assemble_broad_candidates, build_broad_retrieval_queries, detect_broad_scope
from backend.institutional_coverage import assess_institutional_coverage

def d(text, source):
    return Document(page_content=text, metadata={"source": source})

def test_broad_queries_are_bounded():
    q = build_broad_retrieval_queries("What departments are there?")
    assert 1 <= len(q) <= 2

def test_department_scope():
    assert detect_broad_scope("What departments are there?").name == "departments"

def test_unrelated_hostel_not_program_candidate():
    program = d("Academic programs include B.Tech, M.Tech and M.Sc.", "/programs/programs_overview.docx")
    hostel = d("Hostel accommodation and room facilities.", "/hostel_accommodation/general_information.docx")
    result = assemble_broad_candidates("What academic programs are available?", [program, hostel])
    assert program in result
    assert hostel not in result

def test_facilities_can_use_hostel():
    hostel = d("Hostel facilities include furnished rooms and common amenities.", "/hostel_accommodation/general_information.docx")
    result = assemble_broad_candidates("What facilities are available?", [hostel])
    assert hostel in result

def test_program_coverage_without_overview_is_partial():
    result = assess_institutional_coverage("What academic programs are available?", [d("M.Tech program details.", "/admissions/mtech.docx"), d("M.Sc program details.", "/admissions/msc.docx")])
    assert result["status"] == "partially_supported"

def test_program_coverage_with_overview_is_supported():
    result = assess_institutional_coverage("What academic programs are available?", [d("Academic programs overview. B.Tech M.Tech M.Sc. Ph.D. MBA.", "/programs/programs_overview.docx")])
    assert result["status"] == "supported"
'''

HYBRID = r'''def hybrid_retrieve(state: GraphState) -> GraphState:
    """Dense/BM25 retrieval with bounded Phase-8B broad-query recovery."""
    question = state.get("resolved_question", state["question"])
    broad = is_broad_institutional_question(question)
    queries = [question]
    planner_limit = 1 if broad else 2
    for candidate in state.get("generated_queries", []):
        candidate = str(candidate or "").strip()
        if not candidate or candidate.casefold() in {x.casefold() for x in queries}:
            continue
        queries.append(candidate)
        if len(queries) - 1 >= planner_limit:
            break
    if broad:
        for candidate in build_broad_retrieval_queries(question):
            if candidate.casefold() not in {x.casefold() for x in queries}:
                queries.append(candidate)
            if len(queries) >= 4:
                break
    results, weights = [], []
    decays = (1.0, 0.72, 0.50, 0.36)
    for i, query in enumerate(queries):
        dense = dense_retrieve(query)
        keyword = keyword_retrieve(query)
        decay = decays[min(i, len(decays) - 1)]
        results.extend([dense, keyword])
        weights.extend([0.70 * decay, 0.30 * decay])
    return {"retrieval_results": results, "retrieval_queries": queries, "retrieval_weights": weights}
'''

FUSE = r'''def fuse_retrieved_documents(state: GraphState) -> GraphState:
    """Weighted RRF + deduplication + Phase-8B broad candidate assembly."""
    results = state.get("retrieval_results", [])
    weights = state.get("retrieval_weights", [])
    if weights and len(weights) == len(results):
        fused_docs = reciprocal_rank_fusion(results, weights=weights)
    else:
        fused_docs = reciprocal_rank_fusion(results)
    fused_docs = deduplicate_documents(fused_docs)
    question = state.get("resolved_question", state["question"])
    if is_broad_institutional_question(question):
        fused_docs = assemble_broad_candidates(query=question, documents=fused_docs)
    return {"fused_docs": fused_docs}
'''

RERANK = r'''def initial_rerank_documents(state: GraphState) -> GraphState:
    """Select initial anchors; broad queries retain bounded recovery variants."""
    question = state.get("resolved_question", state["question"])
    budget = _initial_rerank_budget(question)
    variants = []
    if is_broad_institutional_question(question):
        variants = [q for q in state.get("retrieval_queries", [])[1:] if q]
    return {"initial_reranked_docs": rerank_documents(query=question, documents=state.get("fused_docs", []), top_k=budget, query_variants=variants)}
'''

COVERAGE_NODE = r'''def assess_evidence_coverage_node(state: GraphState) -> GraphState:
    """Run Phase-8C institutional completeness-aware evidence coverage."""
    question = state.get("resolved_question", state["question"])
    documents = state.get("reranked_docs", [])
    result = assess_institutional_coverage(query=question, documents=documents)
    return {
        "evidence_coverage_status": result["status"],
        "evidence_question_type": result["question_type"],
        "evidence_strong_documents": result["strong_documents"],
        "evidence_partial_documents": result["partial_documents"],
        "evidence_combined_characters": result["combined_characters"],
    }
'''

IMPORTS = '''from backend.broad_institutional_retrieval import (
    assemble_broad_candidates,
    build_broad_retrieval_queries,
    is_broad_institutional_question,
)
from backend.institutional_coverage import assess_institutional_coverage
'''


def replace_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    target = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    if target is None:
        raise RuntimeError(f"Top-level function not found: {name}")
    lines = source.splitlines(keepends=True)
    return "".join(lines[:target.lineno - 1]) + replacement.strip() + "\n\n" + "".join(lines[target.end_lineno:])


def add_imports(source: str) -> str:
    """Normalize required Phase 8B/8C imports without duplicating them."""
    tree = ast.parse(source)
    required = {
        "backend.broad_institutional_retrieval": {
            "assemble_broad_candidates",
            "build_broad_retrieval_queries",
            "is_broad_institutional_question",
        },
        "backend.institutional_coverage": {
            "assess_institutional_coverage",
        },
    }

    lines = source.splitlines(keepends=True)
    replacements = []
    present = set()

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module not in required:
            continue

        present.add(node.module)
        names = {alias.name for alias in node.names if alias.name != "*"}
        merged = sorted(names | required[node.module])
        text = f"from {node.module} import (\n" + "".join(f"    {name},\n" for name in merged) + ")\n"
        replacements.append((node.lineno - 1, node.end_lineno, text))

    for module in required:
        if module not in present:
            import_text = (
                f"from {module} import (\n"
                + "".join(f"    {name},\n" for name in sorted(required[module]))
                + ")\n"
            )
            insert_at = 0
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_at = node.end_lineno
            replacements.append((insert_at, insert_at, import_text))

    for start, end, text in sorted(replacements, reverse=True):
        lines[start:end] = [text]

    return "".join(lines)


def main() -> None:
    BACKEND.mkdir(parents=True, exist_ok=True)
    TESTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    BROAD.write_text(BROAD_SOURCE.strip() + "\n", encoding="utf-8")
    COVERAGE.write_text(COVERAGE_SOURCE.strip() + "\n", encoding="utf-8")
    TEST_FILE.write_text(TEST_SOURCE.strip() + "\n", encoding="utf-8")

    if not NODES.exists():
        raise SystemExit(f"Missing production file: {NODES}")
    if not BACKUP.exists():
        shutil.copy2(NODES, BACKUP)

    source = NODES.read_text(encoding="utf-8")
    source = add_imports(source)
    source = replace_function(source, "hybrid_retrieve", HYBRID)
    source = replace_function(source, "fuse_retrieved_documents", FUSE)
    source = replace_function(source, "initial_rerank_documents", RERANK)
    source = replace_function(source, "assess_evidence_coverage_node", COVERAGE_NODE)
    ast.parse(source)
    NODES.write_text(source, encoding="utf-8")

    print("PHASE 8B + 8C FINAL HARDENING INSTALLED")
    print(f"Updated: {BROAD}")
    print(f"Updated: {COVERAGE}")
    print(f"Updated: {NODES}")
    print(f"Tests:   {TEST_FILE}")
    print(f"Backup:  {BACKUP}")


if __name__ == "__main__":
    main()
