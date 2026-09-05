"""
Phase 8B — Broad Institutional Retrieval

Purpose
-------
Improve recall and candidate quality for broad institutional questions without
embedding institution-specific facts.

Design
------
- The resolved user question remains the primary query.
- At most two generic recovery formulations are added.
- Broad candidate selection happens after Dense/BM25/RRF.
- Source-category hints are used only as weak relevance signals.
- No college/program/department inventory is stored here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BROAD_MAX_CANDIDATES = 12
BROAD_MAX_PER_SOURCE = 2
BROAD_MAX_ADDITIONAL_QUERIES = 2
MIN_BROAD_CANDIDATE_SCORE = 1.0


@dataclass(frozen=True)
class BroadScope:
    name: str
    query_markers: tuple[str, ...]
    source_categories: tuple[str, ...]
    index_tokens: tuple[str, ...]
    content_markers: tuple[str, ...]
    retrieval_formulations: tuple[str, ...]


BROAD_SCOPES = (
    BroadScope(
        name="programs",
        query_markers=(
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
        source_categories=(
            "/programs/",
            "/academics/",
            "/admissions/",
            "/schools/",
            "/departments/",
        ),
        index_tokens=(
            "program",
            "programs",
            "programme",
            "programmes",
            "degree",
            "degrees",
            "academic",
            "overview",
            "index",
        ),
        content_markers=(
            "academic program",
            "academic programme",
            "degree",
            "bachelor",
            "master",
            "ph.d",
            "b.tech",
            "m.tech",
            "m.sc",
            "mba",
        ),
        retrieval_formulations=(
            "academic programs degrees offered institute overview",
            "programmes degrees academic offerings institute",
        ),
    ),
    BroadScope(
        name="departments",
        query_markers=(
            "what departments",
            "which departments",
            "departments available",
            "departments at",
            "academic departments",
            "what schools and departments",
            "schools and departments",
        ),
        source_categories=(
            "/departments/",
            "/schools/",
            "/academics/",
        ),
        index_tokens=(
            "department",
            "departments",
            "school",
            "schools",
            "academic",
            "overview",
            "index",
        ),
        content_markers=(
            "academic department",
            "department",
            "departments",
            "school",
            "schools",
            "academic unit",
            "academic units",
        ),
        retrieval_formulations=(
            "academic departments schools institute overview",
            "departments schools academic units institute",
        ),
    ),
    BroadScope(
        name="research",
        query_markers=(
            "what research",
            "research opportunities",
            "research areas available",
            "research areas at",
            "research opportunities at",
            "what research areas",
            "research themes available",
            "research groups available",
        ),
        source_categories=(
            "/research/",
            "/departments/",
            "/schools/",
            "/research_and_technology_facilities/",
        ),
        index_tokens=(
            "research",
            "overview",
            "index",
        ),
        content_markers=(
            "research area",
            "research areas",
            "research theme",
            "research themes",
            "research group",
            "research groups",
            "research opportunities",
            "research overview",
        ),
        retrieval_formulations=(
            "research areas themes institute overview",
            "research groups opportunities institute",
        ),
    ),
    BroadScope(
        name="facilities",
        query_markers=(
            "what facilities",
            "which facilities",
            "facilities available",
            "facilities at",
            "campus facilities",
            "what infrastructure",
            "campus amenities",
        ),
        source_categories=(
            "/facilities/",
            "/research_and_technology_facilities/",
            "/hostel_accommodation/",
        ),
        index_tokens=(
            "facility",
            "facilities",
            "infrastructure",
            "amenities",
            "overview",
            "index",
        ),
        content_markers=(
            "facility",
            "facilities",
            "laboratory",
            "laboratories",
            "infrastructure",
            "amenities",
        ),
        retrieval_formulations=(
            "campus facilities laboratories infrastructure overview",
            "student research campus facilities amenities",
        ),
    ),
    BroadScope(
        name="admissions",
        query_markers=(
            "what admission opportunities",
            "admission opportunities",
            "admissions available",
            "admission options",
            "ways to get admission",
            "how can i get admission",
            "admission routes",
        ),
        source_categories=(
            "/admissions/",
            "/programs/",
            "/schools/",
            "/departments/",
            "/academics/",
        ),
        index_tokens=(
            "admission",
            "admissions",
            "academic",
            "overview",
            "index",
        ),
        content_markers=(
            "admission",
            "admissions",
            "application",
            "apply",
            "entrance exam",
            "entrance examination",
        ),
        retrieval_formulations=(
            "admission routes programs institute overview",
            "admissions programs application routes institute",
        ),
    ),
)


def _normalize(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("\\", "/")
    return re.sub(r"\s+", " ", value).strip()


def _source(document) -> str:
    metadata = getattr(document, "metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return _normalize(metadata.get("source", ""))


def _content(document) -> str:
    return _normalize(
        getattr(document, "page_content", "")
    )


def detect_broad_scope(query: str) -> BroadScope | None:
    normalized = _normalize(query)

    for scope in BROAD_SCOPES:
        if any(
            marker in normalized
            for marker in scope.query_markers
        ):
            return scope

    return None


def is_broad_institutional_question(query: str) -> bool:
    return detect_broad_scope(query) is not None


def build_broad_retrieval_queries(query: str) -> list[str]:
    """
    Return bounded generic retrieval formulations.

    The caller MUST keep the original resolved question as query zero.
    """

    scope = detect_broad_scope(query)

    if scope is None:
        return []

    result: list[str] = []

    for formulation in scope.retrieval_formulations:
        formulation = formulation.strip()

        if not formulation:
            continue

        if formulation.casefold() == query.strip().casefold():
            continue

        if formulation.casefold() in {
            value.casefold()
            for value in result
        }:
            continue

        result.append(formulation)

        if len(result) >= BROAD_MAX_ADDITIONAL_QUERIES:
            break

    return result


def _source_matches_scope(
    source: str,
    scope: BroadScope,
) -> bool:
    return any(
        category in source
        for category in scope.source_categories
    )


def _looks_like_index_source(
    source: str,
    scope: BroadScope,
) -> bool:
    stem = Path(
        source
    ).stem.lower()

    return any(
        token in stem
        for token in scope.index_tokens
    )


def score_broad_candidate(
    query: str,
    document,
    scope: BroadScope | None = None,
) -> float:
    """
    Score relevance for an already-retrieved broad candidate.

    This is deliberately conservative: source paths provide weak context,
    while actual document content carries most of the score.
    """

    if scope is None:
        scope = detect_broad_scope(query)

    if scope is None:
        return 0.0

    source = _source(document)
    content = _content(document)

    content_hits = sum(
        marker in content
        for marker in scope.content_markers
    )

    if content_hits == 0:
        return 0.0

    score = min(
        6.0,
        float(content_hits),
    )

    if _source_matches_scope(
        source,
        scope,
    ):
        score += 3.0

    if _looks_like_index_source(
        source,
        scope,
    ):
        score += 4.0

    # Generic ingestion wrappers must never strengthen a document.
    if "source original source urls" in content:
        score -= 2.0

    if "retrieval representation" in content:
        score -= 2.0

    return score


def assemble_broad_candidates(
    query: str,
    documents: Iterable,
    max_candidates: int = BROAD_MAX_CANDIDATES,
) -> list:
    """
    Select broad candidates without inventing evidence.

    Relevance is established first. Source diversity is then preferred.
    """

    scope = detect_broad_scope(query)

    documents = list(
        documents
    )

    if scope is None:
        return documents[:max_candidates]

    ranked = []

    for original_rank, document in enumerate(
        documents,
        start=1,
    ):
        score = score_broad_candidate(
            query=query,
            document=document,
            scope=scope,
        )

        if score < MIN_BROAD_CANDIDATE_SCORE:
            continue

        ranked.append(
            (
                score,
                original_rank,
                document,
            )
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )

    selected = []
    source_counts: dict[str, int] = {}

    for _, _, document in ranked:
        source = _source(document)

        count = source_counts.get(
            source,
            0,
        )

        if count >= BROAD_MAX_PER_SOURCE:
            continue

        selected.append(
            document
        )

        source_counts[source] = (
            count + 1
        )

        if len(selected) >= max_candidates:
            break

    return selected
