"""
Retrieval V2 public API.

The V2 retrieval core is intentionally separated from the legacy
retrieval architecture.

Public layers:
    query
    retrieve
    rank
    evidence units
    evidence filtering
    context construction
"""

from .context import ContextBlock, build_context

from .evidence import (
    filter_evidence,
    score_evidence_unit,
)

from .evidence_units import (
    build_all_evidence_units,
    build_evidence_units,
)

from .models import (
    EvidenceDecision,
    EvidenceScore,
    EvidenceSet,
    EvidenceUnit,
    EvidenceUnitCandidate,
    QuerySpec,
    RankedCandidate,
    RetrievedDocument,
    RetrievalIntent,
)

from .query import (
    build_query_spec,
    normalize_query,
    with_intents,
)

from .rank import (
    deduplicate,
    fuse,
)

from .retrieve import (
    RetrievalBatch,
    retrieve,
)


__all__ = [
    # Models
    "ContextBlock",
    "EvidenceDecision",
    "EvidenceScore",
    "EvidenceSet",
    "EvidenceUnit",
    "EvidenceUnitCandidate",
    "QuerySpec",
    "RankedCandidate",
    "RetrievedDocument",
    "RetrievalIntent",
    "RetrievalBatch",

    # Query
    "build_query_spec",
    "normalize_query",
    "with_intents",

    # Retrieval
    "retrieve",

    # Ranking
    "fuse",
    "deduplicate",

    # Evidence units
    "build_evidence_units",
    "build_all_evidence_units",

    # Evidence filtering
    "filter_evidence",
    "score_evidence_unit",

    # Context
    "build_context",
]