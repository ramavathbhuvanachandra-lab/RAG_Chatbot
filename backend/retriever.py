"""
IIT Jodhpur V1 — Retrieval

Purpose
-------
Provide the production retrieval layer for the chatbot.

Pipeline:

    Dense Retrieval
        +
    BM25 Retrieval
        ↓
    Weighted RRF
        ↓
    Duplicate-safe candidate handling
        ↓
    Conservative relevance reranking
        ↓
    Final evidence

Important invariants:

    - Dense remains the primary retrieval signal.
    - BM25 remains a secondary recall signal.
    - RRF uses Dense=0.7 and BM25=0.3.
    - Same-source chunks are allowed to coexist.
    - Duplicate chunks are removed only when their content is
      actually identical/near-identical.
    - Program/topic/entity signals are soft preferences.
    - Reranking uses RRF rank as a prior, not a dominant signal.
    - Discriminative query terms are weighted from the loaded corpus.
    - Exact query phrases provide an additional generic relevance signal.
    - Reranking must not allow metadata heuristics to completely
      override strong lexical/semantic evidence.
"""

from collections import defaultdict
from importlib import import_module
from pathlib import Path
import math
import re

from langchain_community.retrievers import BM25Retriever

from backend.config import (
    DATA_PATH,
    INSTITUTION_ID,
)
from backend.ingestion import (
    load_documents,
    split_documents,
)
from backend.vectorstore import vectorstore
from backend.retrieval_diversity import (
    select_diverse_documents,
)
from backend.core.retrieval_contracts import (
    RetrievalCandidate,
    document_identity,
)
from backend.core.rrf import (
    fuse_ranked_lists,
)

# =========================================================
# Configuration
# =========================================================

RETRIEVER_K = 20

RRF_K = 60

FINAL_CONTEXT_DOCUMENTS = 5

MAX_DOCUMENTS_PER_SOURCE = 2

NEAR_DUPLICATE_THRESHOLD = 0.92

# Bounded query fan-out. The resolved user question is always primary;
# optional planner queries are recall-recovery variants.
MAX_RETRIEVAL_QUERIES = 3

# Alternate queries are deliberately capped so they can recover missed
# lexical/semantic matches without replacing the user's primary intent.
MAX_ALTERNATE_SCORE_CONTRIBUTION = 2.0

# =========================================================
# Active Institution Semantic Registry
# =========================================================


def _load_semantic_registry():
    """Load the semantic registry for the active deployment."""

    module = import_module(
        f"backend.institutions.{INSTITUTION_ID}.semantic_registry"
    )

    registry = getattr(
        module,
        "SEMANTIC_REGISTRY",
        None,
    )

    # Backward-compatible support for the IITJ-specific export name used
    # by the deployment registry created in the previous migration step.
    if registry is None:
        registry = getattr(
            module,
            f"{INSTITUTION_ID.upper()}_SEMANTIC_REGISTRY",
            None,
        )

    if registry is None:
        raise AttributeError(
            f"backend.institutions.{INSTITUTION_ID}.semantic_registry "
            "must export SEMANTIC_REGISTRY"
        )

    registry.validate()
    return registry

SEMANTIC_REGISTRY = _load_semantic_registry()

# Known ingestion-generated wrapper lines. They are removed only from
# answer-model context; source metadata remains available for diagnostics.
INGESTION_WRAPPER_PATTERNS = (
    re.compile(
        r"(?im)^\s*command\s+\d+\s*[•|:]\s*retrieval\s+representation\s*[•|:]?\s*$"
    ),
    re.compile(
        r"(?im)^\s*source\s+original\s+source\s+urls\s+preserved\s+from\s+the\s+command\s+5\s+knowledge\s+source\.?\s*$"
    ),
)


# =========================================================
# Load Documents
# =========================================================

documents = load_documents(
    DATA_PATH
)

chunks = split_documents(
    documents
)


# =========================================================
# Corpus Statistics for Discriminative Query Terms
# =========================================================

# These statistics are derived from the actual loaded corpus.
# No institution-specific vocabulary is hardcoded here.
# The statistics are used only by the reranker to distinguish
# common institutional words from query-specific terms.

CORPUS_DOCUMENT_COUNT = len(chunks)
CORPUS_DOCUMENT_FREQUENCY = defaultdict(int)

for _chunk in chunks:

    _tokens = set(
        token
        for token in re.sub(
            r"[^a-z0-9\s]",
            " ",
            str(_chunk.page_content).lower(),
        ).split()
        if len(token) > 2
    )

    for _token in _tokens:
        CORPUS_DOCUMENT_FREQUENCY[_token] += 1

del _chunk

del _tokens


# =========================================================
# BM25 Retriever
# =========================================================

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = RETRIEVER_K


# =========================================================
# Dense Retriever
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVER_K
    }
)


# =========================================================
# Dense Retrieval
# =========================================================

def dense_retrieve(query: str):
    """
    Retrieve documents using dense vector similarity.
    """

    return retriever.invoke(
        query
    )


# =========================================================
# Keyword Retrieval
# =========================================================

def keyword_retrieve(query: str):
    """
    Retrieve documents using BM25 lexical matching.
    """

    return bm25_retriever.invoke(
        query
    )


# =========================================================
# Text Normalization
# =========================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparisons.
    """

    text = str(text).lower()

    replacements = {
        "b.tech.": "btech",
        "b.tech": "btech",
        "m.tech.": "mtech",
        "m.tech": "mtech",
        "m.sc.": "msc",
        "m.sc": "msc",
        "ph.d.": "phd",
        "ph.d": "phd",
        "_": " ",
        "-": " ",
        "/": " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"[^a-z0-9\s/&+]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize_content(text: str):
    """
    Convert text into normalized tokens.
    """

    return {
        token
        for token in normalize_text(
            text
        ).split()
        if len(token) > 2
    }


# =========================================================
# Source Identity
# =========================================================

def get_source(document):
    """
    Return a canonical absolute source path.
    """

    source = document.metadata.get(
        "source",
        "",
    )

    return str(
        Path(source).resolve()
    )


# =========================================================
# Stable Document Identity
# =========================================================

def get_document_id(document):
    """
    Return the canonical core document identity.

    The retrieval layer keeps this compatibility helper because existing
    callers use the historical function name, while the actual identity
    contract now lives in backend.core.retrieval_contracts.
    """

    return document_identity(
        document
    )

# =========================================================
# Semantic Registry Detection
# =========================================================


def detect_programs(text: str):
    """Detect deployment-configured program signals."""

    return SEMANTIC_REGISTRY.detect_programs(
        text
    )


def detect_topics(text: str):
    """Detect deployment-configured topic signals."""

    return SEMANTIC_REGISTRY.detect_topics(
        text
    )


def detect_entities(text: str):
    """Detect deployment-configured entity signals."""

    return SEMANTIC_REGISTRY.detect_entities(
        text
    )


# =========================================================
# Weighted Reciprocal Rank Fusion
# =========================================================

def fuse_retrieval_candidates(
    ranked_lists,
    k=RRF_K,
    weights=None,
):
    """
    Fuse ranked retrieval results into RetrievalCandidate objects.

    This is the canonical retrieval path for the migrated core.

    All RRF provenance is preserved:
        - Dense rank/score
        - BM25 rank/score
        - per-channel retrieval signals
        - actual weighted RRF score

    The function intentionally does not perform semantic reranking.
    """

    return fuse_ranked_lists(
        ranked_lists,
        weights=weights,
        rrf_k=k,
    )


def reciprocal_rank_fusion(
    ranked_lists,
    k=RRF_K,
    weights=None,
):
    """
    Backward-compatible RRF wrapper.

    Existing callers still receive plain documents. New code should use
    fuse_retrieval_candidates() so retrieval provenance is retained.
    """

    candidates = fuse_retrieval_candidates(
        ranked_lists,
        k=k,
        weights=weights,
    )

    return [
        candidate.document
        for candidate in candidates
    ]

def as_retrieval_candidates(
    items,
):
    """
    Convert a sequence of documents/candidates into RetrievalCandidate objects.

    Existing document callers remain supported. Existing candidates retain
    their retrieval provenance.
    """

    normalized = []

    for item in items:
        if isinstance(item, RetrievalCandidate):
            normalized.append(item)
        else:
            normalized.append(
                RetrievalCandidate.from_document(
                    item
                )
            )

    return normalized


# =========================================================
# Exact Duplicate Removal
# =========================================================

def remove_exact_duplicates(
    documents,
):
    """
    Remove identical chunks.

    Same-source but genuinely different chunks are preserved.
    """

    seen = set()

    unique_documents = []

    for document in documents:

        document_id = get_document_id(
            document
        )

        if document_id in seen:
            continue

        seen.add(
            document_id
        )

        unique_documents.append(
            document
        )

    return unique_documents


# =========================================================
# Near Duplicate Detection
# =========================================================

def is_near_duplicate(
    first_document,
    second_document,
):
    """
    Detect highly similar chunks from the same source.

    Different sources are never treated as near duplicates.
    """

    first_source = get_source(
        first_document
    )

    second_source = get_source(
        second_document
    )

    if first_source != second_source:
        return False

    first_tokens = tokenize_content(
        first_document.page_content
    )

    second_tokens = tokenize_content(
        second_document.page_content
    )

    if not first_tokens or not second_tokens:
        return False

    intersection = (
        first_tokens
        & second_tokens
    )

    union = (
        first_tokens
        | second_tokens
    )

    similarity = (
        len(intersection)
        / len(union)
    )

    return (
        similarity
        >= NEAR_DUPLICATE_THRESHOLD
    )


# =========================================================
# Duplicate Removal
# =========================================================

def deduplicate_documents(
    documents,
):
    """
    Remove exact and near-duplicate chunks while preserving
    genuinely different chunks from the same source.
    """

    documents = remove_exact_duplicates(
        documents
    )

    unique_documents = []

    for document in documents:

        duplicate = False

        for kept_document in unique_documents:

            if is_near_duplicate(
                kept_document,
                document,
            ):

                duplicate = True
                break

        if duplicate:
            continue

        unique_documents.append(
            document
        )

    return unique_documents


# =========================================================
# Relevance Scoring
# =========================================================



# =========================================================
# Query Scope + Source Scope
# =========================================================

PROGRAM_QUERY_MARKERS = (
    "what programs",
    "which programs",
    "programs offered",
    "programme offered",
    "programmes offered",
    "academic programs",
    "academic programmes",
    "what degrees",
    "which degrees",
)


def detect_query_scope(query: str) -> str | None:
    """
    Detect a broad requested knowledge scope from natural query wording.

    This is intentionally generic. It does not know institution-specific
    program names or files; it only identifies that the user is asking for
    program/degree offerings.
    """

    normalized = normalize_text(query)

    if any(
        marker in normalized
        for marker in PROGRAM_QUERY_MARKERS
    ):
        return "programs"

    return None


def score_source_scope(
    query: str,
    source: str,
) -> float:
    """
    Score source-path alignment with the requested query scope.

    For broad program-list questions:

        dedicated programs.docx          -> strong positive
        other files under /programs/      -> moderate positive
        mixed programs/general_information -> small positive
        unrelated administrative scopes   -> modest negative

    The scoring is deliberately bounded and remains only one component
    of the overall relevance score, so source metadata cannot dominate
    strong semantic retrieval evidence.
    """

    scope = detect_query_scope(query)

    if scope != "programs":
        return 0.0

    normalized_source = str(source or "").replace("\\", "/").lower()

    if normalized_source.endswith("/programs.docx"):
        return 6.0

    if "/programs/" in normalized_source:
        if normalized_source.endswith("/general_information.docx"):
            return 2.0
        return 4.0

    non_program_scopes = (
        "/hostel_accommodation/",
        "/admissions/",
        "/research_platforms/",
        "/research/",
        "/finance/",
    )

    if any(
        marker in normalized_source
        for marker in non_program_scopes
    ):
        return -3.0

    return 0.0
def _query_term_weights(query: str):
    """
    Return corpus-derived weights for query terms.

    Rare terms receive more weight than words that occur throughout the
    institutional corpus. This is generic and adapts automatically when
    the corpus changes between colleges.
    """

    tokens = [
        token
        for token in normalize_text(query).split()
        if len(token) > 2
    ]

    if not tokens:
        return {}

    document_count = max(
        1,
        CORPUS_DOCUMENT_COUNT,
    )

    weights = {}

    for token in set(tokens):

        document_frequency = (
            CORPUS_DOCUMENT_FREQUENCY.get(
                token,
                0,
            )
        )

        # Smoothed inverse-document-frequency signal.
        idf = (
            math.log(
                (document_count + 1)
                / (document_frequency + 1)
            )
            + 1.0
        )

        # Keep the signal bounded so one extremely rare token cannot
        # completely dominate semantic retrieval or scope signals.
        weight = max(
            0.75,
            min(
                3.0,
                1.0 + (idf - 1.0) / 3.0,
            ),
        )

        weights[token] = weight

    return weights


def _query_ngrams(
    tokens,
    n,
):
    """
    Return contiguous normalized n-grams from a token sequence.
    """

    if len(tokens) < n:
        return []

    return [
        " ".join(tokens[index:index + n])
        for index in range(
            len(tokens) - n + 1
        )
    ]


def _score_exact_query_phrases(
    query: str,
    content: str,
    term_weights,
):
    """
    Score exact query phrases using corpus-derived term importance.

    Only phrases containing at least one above-baseline query term receive
    the bonus. This prevents common phrases such as generic question wording
    from dominating the reranker.
    """

    tokens = [
        token
        for token in normalize_text(query).split()
        if len(token) > 2
    ]

    if len(tokens) < 2:
        return 0.0

    score = 0.0

    for n in (3, 2):

        for phrase in _query_ngrams(
            tokens,
            n,
        ):

            if phrase not in content:
                continue

            phrase_tokens = phrase.split()

            phrase_weights = [
                term_weights.get(
                    token,
                    1.0,
                )
                for token in phrase_tokens
            ]

            if not any(
                weight > 1.1
                for weight in phrase_weights
            ):
                continue

            average_weight = (
                sum(phrase_weights)
                / len(phrase_weights)
            )

            # Trigrams get a slightly stronger signal than bigrams.
            phrase_bonus = (
                1.75
                if n == 3
                else 1.25
            )

            score += (
                average_weight
                * phrase_bonus
            )

    return min(
        6.0,
        score,
    )


def _score_discriminative_overlap(
    query: str,
    content: str,
):
    """
    Score weighted query/content overlap.

    Unlike plain overlap, common corpus-wide words contribute less while
    distinctive query terms contribute more.
    """

    term_weights = _query_term_weights(
        query
    )

    if not term_weights:
        return 0.0, term_weights

    content_tokens = set(
        normalize_text(content).split()
    )

    matched_weight = sum(
        weight
        for token, weight in term_weights.items()
        if token in content_tokens
    )

    total_weight = sum(
        term_weights.values()
    )

    if total_weight <= 0.0:
        return 0.0, term_weights

    normalized_overlap = (
        matched_weight
        / total_weight
    )

    return (
        normalized_overlap
        * 12.0,
        term_weights,
    )


def score_document_relevance(
    query: str,
    document,
    original_rank: int,
):
    """
    Query-aware relevance score.

    Priority:

        1. Discriminative query/content relevance
        2. Exact query phrase relevance
        3. RRF rank prior
        4. Program consistency
        5. Topic consistency
        6. Entity consistency
        7. Requested-scope source alignment
        8. Answer-bearing content
        9. Junk/noise penalty

    RRF remains important, but it is deliberately a bounded prior so that
    a strong query-specific evidence document can overtake a generic high-
    ranked candidate.
    """

    normalized_query = normalize_text(
        query
    )

    content = normalize_text(
        document.page_content
    )

    raw_source = str(
        document.metadata.get(
            "source",
            "",
        )
    )

    source = normalize_text(
        raw_source
    )

    score = 0.0

    # -----------------------------------------------------
    # 1. Discriminative query overlap
    # -----------------------------------------------------

    lexical_score, term_weights = (
        _score_discriminative_overlap(
            normalized_query,
            content,
        )
    )

    score += lexical_score

    # -----------------------------------------------------
    # 2. Exact query phrases
    # -----------------------------------------------------

    score += _score_exact_query_phrases(
        normalized_query,
        content,
        term_weights,
    )

    # -----------------------------------------------------
    # 3. RRF rank prior
    #
    # The prior is intentionally softer than the previous fixed
    # 30/(rank+1) term. Retrieval rank remains useful, but it cannot
    # permanently suppress a document with strong query-specific evidence.
    # -----------------------------------------------------

    score += (
        10.0
        / math.sqrt(
            max(
                1,
                original_rank,
            )
        )
    )

    # -----------------------------------------------------
    # 4. Program consistency
    # -----------------------------------------------------

    query_programs = detect_programs(
        query
    )

    document_programs = detect_programs(
        f"{source} {content}"
    )

    if query_programs:

        matched_programs = (
            query_programs
            & document_programs
        )

        missing_programs = (
            query_programs
            - document_programs
        )

        score += (
            len(matched_programs)
            * 12.0
        )

        score -= (
            len(missing_programs)
            * 4.0
        )

        mismatched_programs = (
            document_programs
            - query_programs
        )

        score -= (
            len(mismatched_programs)
            * 3.0
        )

    # -----------------------------------------------------
    # 5. Topic consistency
    # -----------------------------------------------------

    query_topics = detect_topics(
        query
    )

    document_topics = detect_topics(
        f"{source} {content}"
    )

    matched_topics = (
        query_topics
        & document_topics
    )

    score += (
        len(matched_topics)
        * 5.0
    )

    # -----------------------------------------------------
    # 6. Entity consistency
    # -----------------------------------------------------

    query_entities = detect_entities(
        query
    )

    document_entities = detect_entities(
        f"{source} {content}"
    )

    matched_entities = (
        query_entities
        & document_entities
    )

    mismatched_entities = (
        document_entities
        - query_entities
    )

    score += (
        len(matched_entities)
        * 4.0
    )

    score -= (
        len(mismatched_entities)
        * 1.5
    )

    # -----------------------------------------------------
    # 7. Answer-bearing phrases
    # -----------------------------------------------------

    answer_patterns = [
        "research themes",
        "research areas",
        "key facilities",
        "facilities include",
        "admission",
        "eligibility",
        "fee structure",
        "tuition fee",
        "fees",
        "dining facilities",
        "mess",
        "programmes",
        "programs",
    ]

    for pattern in answer_patterns:

        if pattern in content:
            score += 3.0

    # -----------------------------------------------------
    # 8. Requested-scope source alignment
    # -----------------------------------------------------

    score += score_source_scope(
        query=query,
        source=raw_source,
    )

    # -----------------------------------------------------
    # 9. Metadata / navigation noise
    # -----------------------------------------------------

    url_count = (
        document.page_content.count(
            "http"
        )
    )

    if url_count >= 3:

        score -= 5.0

    elif url_count >= 1:

        score -= 2.0

    navigation_markers = [
        "original source urls",
        "back to index",
        "click here",
    ]

    for marker in navigation_markers:

        if marker in content:
            score -= 2.0

    return score


# =========================================================
# Final Context Sanitization
# =========================================================

def sanitize_context_text(
    text: str,
) -> str:
    """
    Remove only known ingestion-generated wrapper lines before
    context is passed to the answer model.
    """

    cleaned = str(
        text or ""
    )

    for pattern in INGESTION_WRAPPER_PATTERNS:
        cleaned = pattern.sub(
            "",
            cleaned,
        )

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


# =========================================================
# Variant-Aware Relevance
# =========================================================

def score_document_relevance_variants(
    query_variants,
    document,
    original_rank: int,
):
    """
    Score a candidate against a bounded primary + alternate query set.

    Primary query is authoritative.

    Alternate queries are recovery signals only:
        - positive contribution only
        - capped contribution
        - cannot erase a strong primary-query match
    """

    variants = []

    for query in query_variants or ():

        normalized = str(
            query or ""
        ).strip()

        if not normalized:
            continue

        if normalized.casefold() in {
            item.casefold()
            for item in variants
        }:
            continue

        variants.append(
            normalized
        )

        if len(variants) >= MAX_RETRIEVAL_QUERIES:
            break

    if not variants:

        return score_document_relevance(
            query="",
            document=document,
            original_rank=original_rank,
        )

    primary_score = score_document_relevance(
        query=variants[0],
        document=document,
        original_rank=original_rank,
    )

    # Primary query stays the foundation of the score.
    total = primary_score

    for index, alternate in enumerate(
        variants[1:],
        start=1,
    ):

        alternate_score = score_document_relevance(
            query=alternate,
            document=document,
            original_rank=original_rank,
        )

        # Recovery only: alternate queries can only add bounded positive
        # evidence. Their contribution is deliberately capped.
        positive_alternate = min(
            MAX_ALTERNATE_SCORE_CONTRIBUTION,
            max(
                0.0,
                alternate_score,
            ),
        )

        decay = (
            0.60
            if index == 1
            else 0.35
        )

        total += (
            positive_alternate
            * decay
        )

    return total


# =========================================================
# Reranking
# =========================================================

def rerank_documents(
    query: str,
    documents,
    top_k: int = FINAL_CONTEXT_DOCUMENTS,
    query_variants=None,
):
    """
    Rerank retrieved candidates using the resolved primary question and,
    optionally, bounded planner-produced query variants.

    Backward compatible with all existing callers that pass only:
        query, documents, top_k

    Query variants are recall recovery only. The primary query remains
    authoritative and alternate contribution is bounded.
    """

    documents = deduplicate_documents(
        documents
    )

    variants = []

    if query:
        variants.append(
            str(query).strip()
        )

    for variant in query_variants or ():

        normalized = str(
            variant or ""
        ).strip()

        if not normalized:
            continue

        if normalized.casefold() in {
            item.casefold()
            for item in variants
        }:
            continue

        variants.append(
            normalized
        )

        if len(variants) >= MAX_RETRIEVAL_QUERIES:
            break

    scored_documents = []

    for original_rank, document in enumerate(
        documents,
        start=1,
    ):

        if len(variants) > 1:

            score = score_document_relevance_variants(
                query_variants=variants,
                document=document,
                original_rank=original_rank,
            )

        else:

            score = score_document_relevance(
                query=query,
                document=document,
                original_rank=original_rank,
            )

        scored_documents.append(
            {
                "document":
                    document,
                "score":
                    score,
                "original_rank":
                    original_rank,
                "source":
                    get_source(
                        document
                    ),
            }
        )

    scored_documents.sort(
        key=lambda item: (
            item["score"],
            -item["original_rank"],
        ),
        reverse=True,
    )

    return select_diverse_documents(
        scored_documents,
        top_k=top_k,
    )

# =========================================================
# Context Formatting
# =========================================================

def format_context(
    documents,
):
    """
    Format cleaned evidence for the answer model.

    Internal retrieval labels and known ingestion wrappers are never
    deliberately exposed as part of the factual evidence text.
    """

    formatted_context = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        cleaned_text = sanitize_context_text(
            document.page_content
        )

        if not cleaned_text:
            continue

        formatted_context.append(
            f"Evidence {index}\n"
            f"{cleaned_text}"
        )

    return "\n\n".join(
        formatted_context
    )