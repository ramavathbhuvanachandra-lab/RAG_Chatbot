"""
Phase 8C — Institutional Evidence Coverage.

Separates relevance from completeness for broad institutional questions.

Focused questions keep the existing evidence_coverage engine.
Broad questions receive an additional institutional-coverage assessment.

No institution-specific factual inventory is hardcoded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.evidence_coverage import assess_evidence_coverage


BROAD_SCOPES = {
    "programs": (
        "what programs",
        "which programs",
        "academic programs",
        "programs available",
        "what degrees",
        "which degrees",
        "degrees available",
        "academic programme",
        "academic programmes",
    ),
    "departments": (
        "what departments",
        "which departments",
        "departments available",
        "departments at",
        "academic departments",
        "schools and departments",
        "what schools and departments",
    ),
    "research": (
        "what research",
        "research opportunities",
        "research areas available",
        "research areas at",
        "research opportunities at",
        "what research areas",
        "research themes available",
        "research groups available",
    ),
    "facilities": (
        "what facilities",
        "which facilities",
        "facilities available",
        "facilities at",
        "campus facilities",
        "what infrastructure",
        "campus amenities",
    ),
    "admissions": (
        "what admission opportunities",
        "admission opportunities",
        "admissions available",
        "admission options",
        "ways to get admission",
        "how can i get admission",
        "admission routes",
    ),
}


_SCOPE_CONTENT_MARKERS = {
    "programs": (
        "program",
        "programme",
        "degree",
        "bachelor",
        "master",
        "ph.d",
        "b.tech",
        "m.tech",
        "m.sc",
        "mba",
    ),
    "departments": (
        "department",
        "departments",
        "school",
        "schools",
        "academic unit",
        "academic units",
        "centre",
        "center",
    ),
    "research": (
        "research",
        "research area",
        "research areas",
        "research theme",
        "research themes",
        "research group",
        "research groups",
        "research opportunity",
        "research opportunities",
    ),
    "facilities": (
        "facility",
        "facilities",
        "laboratory",
        "laboratories",
        "infrastructure",
        "amenity",
        "amenities",
    ),
    "admissions": (
        "admission",
        "admissions",
        "application",
        "apply",
        "entrance",
        "entrance exam",
        "entrance examination",
    ),
}


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text or "").lower(),
    ).strip()


def _source(document: Any) -> str:
    metadata = getattr(
        document,
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        return ""

    return _normalize(
        metadata.get("source", "")
    )


def _content(document: Any) -> str:
    return _normalize(
        getattr(
            document,
            "page_content",
            "",
        )
    )


def detect_broad_scope(
    query: str,
) -> str | None:

    normalized = _normalize(query)

    for scope, markers in BROAD_SCOPES.items():

        if any(
            marker in normalized
            for marker in markers
        ):
            return scope

    return None


def _content_relevance(
    document: Any,
    scope: str,
) -> int:
    """
    Content-only relevance.

    A source path can never create relevance by itself.
    """

    content = _content(document)

    if not content:
        return 0

    return sum(
        1
        for marker in _SCOPE_CONTENT_MARKERS.get(
            scope,
            (),
        )
        if marker in content
    )


def _is_relevant_document(
    document: Any,
    scope: str,
) -> bool:
    return (
        _content_relevance(
            document,
            scope,
        )
        > 0
    )


def _looks_like_institutional_index(
    document: Any,
    scope: str,
) -> bool:

    source = _source(document)
    content = _content(document)
    filename = Path(source).stem.lower()

    filename_terms = {
        "programs": (
            "program",
            "programs",
            "programme",
            "programmes",
            "degree",
            "academic",
            "overview",
            "index",
        ),
        "departments": (
            "department",
            "departments",
            "school",
            "schools",
            "academic",
            "overview",
            "index",
        ),
        "research": (
            "research",
            "overview",
            "index",
        ),
        "facilities": (
            "facility",
            "facilities",
            "infrastructure",
            "amenities",
            "overview",
            "index",
        ),
        "admissions": (
            "admission",
            "admissions",
            "academic",
            "overview",
            "index",
        ),
    }

    if any(
        term in filename
        for term in filename_terms.get(
            scope,
            (),
        )
    ):
        return True

    content_terms = {
        "programs": (
            "academic programs",
            "academic programmes",
            "degrees offered",
            "programmes offered",
        ),
        "departments": (
            "academic departments",
            "departments and schools",
            "academic units",
        ),
        "research": (
            "research overview",
            "research areas",
            "research themes",
        ),
        "facilities": (
            "facilities overview",
            "campus facilities",
        ),
        "admissions": (
            "admission overview",
            "admission routes",
            "admission opportunities",
        ),
    }

    return any(
        marker in content
        for marker in content_terms.get(
            scope,
            (),
        )
    )


def _breadth_signal_count(
    content: str,
    scope: str,
) -> int:
    """
    Bounded coverage signal.

    This does not represent the factual number of programs, departments,
    facilities, or other entities.
    """

    patterns = {
        "programs": (
            r"\bb\.?tech\b",
            r"\bm\.?tech\b",
            r"\bm\.?sc\b",
            r"\bmba\b",
            r"\bph\.?d\b",
            r"\bbachelor",
            r"\bmaster",
            r"\bdegree",
        ),
        "departments": (
            r"\bdepartment\b",
            r"\bdepartments\b",
            r"\bschool\b",
            r"\bschools\b",
            r"\bacademic unit\b",
            r"\bcentre\b",
            r"\bcenter\b",
        ),
        "research": (
            r"\bresearch area\b",
            r"\bresearch areas\b",
            r"\bresearch theme\b",
            r"\bresearch themes\b",
            r"\bresearch group\b",
            r"\bresearch groups\b",
            r"\bresearch opportunity\b",
            r"\bresearch opportunities\b",
        ),
        "facilities": (
            r"\bfacilit(?:y|ies)\b",
            r"\blaborator(?:y|ies)\b",
            r"\binfrastructure\b",
            r"\bamenit(?:y|ies)\b",
        ),
        "admissions": (
            r"\badmission\b",
            r"\badmissions\b",
            r"\bapplication\b",
            r"\bentrance\b",
            r"\bjam\b",
            r"\bgate\b",
            r"\bnet\b",
        ),
    }

    total = 0

    for pattern in patterns.get(
        scope,
        (),
    ):
        total += len(
            re.findall(
                pattern,
                content,
            )
        )

    return min(12, total)


def assess_institutional_coverage(
    query: str,
    documents,
) -> dict[str, Any]:
    """
    Assess evidence for broad institutional questions.

    Semantics:
        supported
            Strong institutional or distributed evidence.

        partially_supported
            Relevant evidence exists, but completeness is not established.

        insufficient
            No relevant evidence exists.
    """

    documents = list(
        documents or []
    )

    baseline = assess_evidence_coverage(
        query=query,
        documents=documents,
    )

    scope = detect_broad_scope(query)

    # Focused question: preserve the established engine.
    if scope is None:
        return baseline

    # ---------------------------------------------------------
    # Relevance gate
    # ---------------------------------------------------------

    relevant_documents = [
        document
        for document in documents
        if _is_relevant_document(
            document,
            scope,
        )
    ]

    # Absolutely no relevant evidence.
    if not relevant_documents:

        result = dict(
            baseline
        )

        result.update(
            {
                "status": "insufficient",
                "question_type": "list",
                "institutional_coverage_scope": scope,
                "institutional_coverage_sources": 0,
                "institutional_coverage_index_documents": 0,
                "institutional_coverage_breadth": 0,
            }
        )

        return result

    # ---------------------------------------------------------
    # Source diversity
    # ---------------------------------------------------------

    relevant_sources = {
        _source(document)
        for document in relevant_documents
        if _source(document)
    }

    source_count = len(
        relevant_sources
    )

    # ---------------------------------------------------------
    # Institutional overview/index
    # ---------------------------------------------------------

    index_count = sum(
        1
        for document in relevant_documents
        if _looks_like_institutional_index(
            document,
            scope,
        )
    )

    # ---------------------------------------------------------
    # Combined breadth
    # ---------------------------------------------------------

    combined_content = "\n".join(
        _content(document)
        for document in relevant_documents
    )

    breadth = _breadth_signal_count(
        combined_content,
        scope,
    )

    # ---------------------------------------------------------
    # Coverage decision
    # ---------------------------------------------------------

    # Department overviews need only modest breadth because the document
    # itself explicitly represents the institutional unit list.
    if (
        index_count >= 1
        and (
            (
                scope == "departments"
                and breadth >= 2
            )
            or breadth >= 3
        )
    ):
        status = "supported"

    # Distributed evidence can support a broad answer when it has enough
    # independent sources and breadth.
    elif (
        source_count >= 2
        and breadth >= 4
    ):
        status = "supported"

    # Otherwise, evidence is relevant but not demonstrably complete.
    else:
        status = "partially_supported"

    result = dict(
        baseline
    )

    result.update(
        {
            "status": status,
            "question_type": "list",
            "institutional_coverage_scope": scope,
            "institutional_coverage_sources": source_count,
            "institutional_coverage_index_documents": index_count,
            "institutional_coverage_breadth": breadth,
        }
    )

    return result
