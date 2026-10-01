"""
IITJ lexical A/B retrieval gate — isolated from legacy backend.vectorstore.

This benchmark deliberately avoids importing backend.retriever or
backend.vectorstore because those compatibility layers may eagerly construct
Chroma with incomplete deployment configuration.

It measures the same retrieval ingredients directly:

    Dense (Chroma, existing IITJ collection)
        +
    BM25 (same canonical DOCX corpus + same 800/150 splitter)
        +
    IITJ institution lexical lane

Run from the RAG_Chatbot repository root:

    PYTHONPATH=. python tests_reusable/iitj_lexical_ab_gate.py

Optional:

    PYTHONPATH=. python tests_reusable/iitj_lexical_ab_gate.py --json
    PYTHONPATH=. python tests_reusable/iitj_lexical_ab_gate.py --limit 12
    PYTHONPATH=. python tests_reusable/iitj_lexical_ab_gate.py --k 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CASES = (
    {
        "id": "mtech_eligibility",
        "query": "What are the M.Tech eligibility requirements?",
        "expected_sources": ("admissions/mtech_admissions.docx",),
    },
    {
        "id": "mtech_eligibility_punctuation",
        "query": "M.Tech. eligibility requirements?",
        "expected_sources": ("admissions/mtech_admissions.docx",),
    },
    {
        "id": "msc_admission_routes",
        "query": "What are the M.Sc admission routes?",
        "expected_sources": ("admissions/msc_admissions.docx",),
    },
    {
        "id": "msc_admission_routes_alias",
        "query": "What are the routes for admission to the Master of Science programme?",
        "expected_sources": ("admissions/msc_admissions.docx",),
    },
    {
        "id": "tuition_fee",
        "query": "What is the tuition fee?",
        "expected_sources": ("fees_and_finance/fees_and_finance.docx", "finance/fees_and_finance.docx"),
    },
    {
        "id": "tuition_fee_alias",
        "query": "How much does tuition cost?",
        "expected_sources": ("fees_and_finance/fees_and_finance.docx", "finance/fees_and_finance.docx"),
    },
    {
        "id": "hostel_fees",
        "query": "What are the hostel fees?",
        "expected_sources": ("hostel_accommodation/general_information.docx",),
    },
    {
        "id": "hostel_fees_alias",
        "query": "How much does hostel accommodation cost?",
        "expected_sources": ("hostel_accommodation/general_information.docx",),
    },
    {
        "id": "apply_admission",
        "query": "How do I apply for admission?",
        "expected_sources": ("admissions/general_admissions.docx",),
    },
    {
        "id": "apply_admission_alias",
        "query": "What is the application process for admission?",
        "expected_sources": ("admissions/general_admissions.docx",),
    },
    {
        "id": "dining_halls",
        "query": "How many dining halls are there?",
        "expected_sources": ("hostel_accommodation/general_information.docx",),
    },
    {
        "id": "mess_facilities",
        "query": "What mess and dining facilities are available?",
        "expected_sources": ("hostel_accommodation/general_information.docx",),
    },
    {
        "id": "research_information",
        "query": "What research areas are available?",
        "expected_sources": ("research/research.docx",),
    },
    {
        "id": "campus_overview",
        "query": "What is the vision of IIT Jodhpur?",
        "expected_sources": ("institute_overview/overview.docx",),
    },
    {
        "id": "phd_requirements",
        "query": "What are the Ph.D. eligibility requirements?",
        "expected_sources": ("admissions/phd_admissions.docx",),
    },
    {
        "id": "phd_requirements_alias",
        "query": "What qualifications are needed for the doctoral programme?",
        "expected_sources": ("admissions/phd_admissions.docx",),
    },
)

BASELINE_WEIGHTS = (0.55, 0.30)
INTEGRATED_WEIGHTS = (0.55, 0.30, 0.15)
TOP_K_CHECKS = (1, 5, 10, 20)


def _project_root() -> Path:
    # tests_reusable/iitj_lexical_ab_gate.py -> project root is one level up.
    return Path(__file__).resolve().parents[1]


def _data_path() -> Path:
    return _project_root() / "data" / "data_iitj"


def _vectorstore_path() -> Path:
    return _project_root() / "chroma_db"


def _collection_name() -> str:
    return os.getenv("IITJ_VECTORSTORE_COLLECTION", "iitj_v1").strip() or "iitj_v1"


def _ollama_url() -> str:
    return (
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        or "http://localhost:11434"
    )


def _embedding_model() -> str:
    return (
        os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest").strip()
        or "nomic-embed-text:latest"
    )


def _source(document: Any) -> str:
    if isinstance(document, dict):
        metadata = document.get("metadata") or {}
    else:
        metadata = getattr(document, "metadata", {}) or {}

    if not isinstance(metadata, dict):
        return ""

    return str(
        metadata.get("source")
        or metadata.get("source_path")
        or metadata.get("path")
        or ""
    )


def _content(document: Any) -> str:
    if isinstance(document, dict):
        return str(document.get("page_content") or "")
    return str(getattr(document, "page_content", "") or "")


def _source_match(source: str, expected: Iterable[str]) -> bool:
    value = source.casefold()
    return any(str(item).casefold() in value for item in expected)


def _source_rank(items: Iterable[Any], expected: Iterable[str]) -> int | None:
    for rank, item in enumerate(items, start=1):
        document = getattr(item, "document", item)
        if _source_match(_source(document), expected):
            return rank
    return None


def _recall_at(items: Iterable[Any], expected: Iterable[str], k: int) -> bool:
    values = list(items)
    return _source_rank(values[:k], expected) is not None


def _preview(item: Any) -> dict[str, Any]:
    document = getattr(item, "document", item)
    provenance = getattr(item, "provenance", None)
    signals = getattr(provenance, "signals", ()) if provenance is not None else ()

    payload = {
        "source": _source(document),
        "text": _content(document)[:180].replace("\n", " "),
    }

    if hasattr(item, "final_score"):
        payload["final_score"] = round(
            float(getattr(item, "final_score", 0.0) or 0.0),
            6,
        )

    if provenance is not None:
        payload["rrf_score"] = round(
            float(getattr(provenance, "rrf_score", 0.0) or 0.0),
            6,
        )
        payload["channels"] = [
            str(getattr(signal, "channel", ""))
            for signal in signals
        ]

    return payload


def _load_corpus() -> tuple[Any, ...]:
    data_path = _data_path()

    if not data_path.exists():
        raise FileNotFoundError(
            f"IITJ data path does not exist: {data_path}"
        )

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=(
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ),
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError(
            f"No .docx chunks were produced from {data_path}"
        )

    return tuple(chunks)


def _load_dense_retriever(k: int) -> Any:
    path = _vectorstore_path()

    if not path.exists():
        raise FileNotFoundError(
            f"IITJ Chroma directory does not exist: {path}"
        )

    collection = _collection_name()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", collection):
        raise ValueError(
            "IITJ_VECTORSTORE_COLLECTION must contain only letters, "
            "numbers, underscores, or hyphens."
        )

    embeddings = OllamaEmbeddings(
        model=_embedding_model(),
        base_url=_ollama_url(),
    )

    store = Chroma(
        collection_name=collection,
        persist_directory=str(path),
        embedding_function=embeddings,
    )

    return store.as_retriever(
        search_kwargs={"k": int(k)}
    )


def _load_bm25_retriever(corpus: tuple[Any, ...], k: int) -> Any:
    retriever = BM25Retriever.from_documents(corpus)
    retriever.k = int(k)
    return retriever


def _fuse(
    rrf_module: Any,
    lists: list[list[Any]],
    weights: tuple[float, ...],
    query: str,
) -> tuple[Any, ...]:
    return tuple(
        rrf_module.fuse_ranked_lists(
            lists,
            weights=weights,
            primary_query=query,
            retrieval_queries=(query,),
            channels=(
                "dense",
                "bm25",
                "institution_lexical",
            )[: len(lists)],
        )
    )


def _run_case(
    case: dict[str, Any],
    *,
    dense_retriever: Any,
    bm25_retriever: Any,
    corpus: tuple[Any, ...],
    lexical_module: Any,
    rrf_module: Any,
    retrieval_k: int,
) -> dict[str, Any]:
    query = str(case["query"])
    expected = tuple(case["expected_sources"])

    dense = list(dense_retriever.invoke(query))
    bm25 = list(bm25_retriever.invoke(query))
    lexical_hits = list(
        lexical_module.retrieve_lexical(
            query,
            corpus,
            limit=max(30, retrieval_k),
            min_score=0.28,
        )
    )
    lexical_docs = [hit.document for hit in lexical_hits]

    baseline = _fuse(
        rrf_module,
        [dense, bm25],
        BASELINE_WEIGHTS,
        query,
    )
    integrated = _fuse(
        rrf_module,
        [dense, bm25, lexical_docs],
        INTEGRATED_WEIGHTS,
        query,
    )

    baseline_rank = _source_rank(baseline, expected)
    lexical_rank = _source_rank(lexical_hits, expected)
    integrated_rank = _source_rank(integrated, expected)

    return {
        "id": case["id"],
        "query": query,
        "expected_sources": expected,
        "baseline": {
            "rank": baseline_rank,
            "recall": {
                str(k): _recall_at(baseline, expected, k)
                for k in TOP_K_CHECKS
            },
            "top5": [_preview(item) for item in baseline[:5]],
        },
        "institution_lexical": {
            "rank": lexical_rank,
            "recall": {
                str(k): _recall_at(lexical_hits, expected, k)
                for k in TOP_K_CHECKS
            },
            "top5": [_preview(item) for item in lexical_hits[:5]],
        },
        "integrated": {
            "rank": integrated_rank,
            "recall": {
                str(k): _recall_at(integrated, expected, k)
                for k in TOP_K_CHECKS
            },
            "top5": [_preview(item) for item in integrated[:5]],
        },
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def count_hits(section: str, k: int) -> int:
        return sum(
            1
            for result in results
            if result[section]["recall"][str(k)]
        )

    baseline_top1 = count_hits("baseline", 1)
    baseline_top5 = count_hits("baseline", 5)
    baseline_top10 = count_hits("baseline", 10)
    baseline_top20 = count_hits("baseline", 20)

    integrated_top1 = count_hits("integrated", 1)
    integrated_top5 = count_hits("integrated", 5)
    integrated_top10 = count_hits("integrated", 10)
    integrated_top20 = count_hits("integrated", 20)

    regressions: list[str] = []
    improvements: list[str] = []
    rescues: list[str] = []

    for result in results:
        baseline_rank = result["baseline"]["rank"]
        integrated_rank = result["integrated"]["rank"]

        if baseline_rank is not None and integrated_rank is None:
            regressions.append(result["id"])
        elif baseline_rank is None and integrated_rank is not None:
            rescues.append(result["id"])
        elif (
            baseline_rank is not None
            and integrated_rank is not None
            and integrated_rank < baseline_rank
        ):
            improvements.append(result["id"])

    n = len(results)

    return {
        "cases": n,
        "baseline": {
            "top1_hits": baseline_top1,
            "top5_hits": baseline_top5,
            "top10_hits": baseline_top10,
            "top20_hits": baseline_top20,
            "top1_rate": round(baseline_top1 / n, 4) if n else 0.0,
            "top5_rate": round(baseline_top5 / n, 4) if n else 0.0,
            "top10_rate": round(baseline_top10 / n, 4) if n else 0.0,
            "top20_rate": round(baseline_top20 / n, 4) if n else 0.0,
        },
        "integrated": {
            "top1_hits": integrated_top1,
            "top5_hits": integrated_top5,
            "top10_hits": integrated_top10,
            "top20_hits": integrated_top20,
            "top1_rate": round(integrated_top1 / n, 4) if n else 0.0,
            "top5_rate": round(integrated_top5 / n, 4) if n else 0.0,
            "top10_rate": round(integrated_top10 / n, 4) if n else 0.0,
            "top20_rate": round(integrated_top20 / n, 4) if n else 0.0,
        },
        "regressions": regressions,
        "improvements": improvements,
        "rescues": rescues,
        "gate": "FAIL" if regressions else "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=len(CASES),
        help="Number of cases to run.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Dense/BM25 retrieval k.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.k < 1:
        raise SystemExit("--k must be >= 1")

    case_slice = CASES[: min(args.limit, len(CASES))]

    print(
        "Loading IITJ corpus directly "
        "(bypassing legacy backend.vectorstore)..."
    ) if not args.json else None

    corpus = _load_corpus()
    dense_retriever = _load_dense_retriever(args.k)
    bm25_retriever = _load_bm25_retriever(corpus, args.k)

    lexical_module = __import__(
        "backend.core.retrieval.lexical_recall",
        fromlist=["*"],
    )
    rrf_module = __import__(
        "backend.core.rrf",
        fromlist=["*"],
    )

    results: list[dict[str, Any]] = []
    for case in case_slice:
        results.append(
            _run_case(
                case,
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                corpus=corpus,
                lexical_module=lexical_module,
                rrf_module=rrf_module,
                retrieval_k=args.k,
            )
        )

    summary = _summarize(results)

    payload = {
        "collection": _collection_name(),
        "vectorstore_path": str(_vectorstore_path()),
        "data_path": str(_data_path()),
        "embedding_model": _embedding_model(),
        "ollama_url": _ollama_url(),
        "corpus_chunks": len(corpus),
        "baseline_weights": BASELINE_WEIGHTS,
        "integrated_weights": INTEGRATED_WEIGHTS,
        "summary": summary,
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 1 if summary["gate"] == "FAIL" else 0

    print()
    print("=" * 110)
    print("IITJ LEXICAL A/B RETRIEVAL GATE")
    print("=" * 110)
    print(f"Collection     : {_collection_name()}")
    print(f"Corpus chunks   : {len(corpus)}")
    print(f"Cases           : {len(results)}")
    print(f"Dense/BM25 k    : {args.k}")
    print(
        f"Baseline        : Dense={BASELINE_WEIGHTS[0]:.2f} + "
        f"BM25={BASELINE_WEIGHTS[1]:.2f}"
    )
    print(
        f"Integrated      : Dense={INTEGRATED_WEIGHTS[0]:.2f} + "
        f"BM25={INTEGRATED_WEIGHTS[1]:.2f} + "
        f"IITJ Lexical={INTEGRATED_WEIGHTS[2]:.2f}"
    )
    print("-" * 110)

    for result in results:
        b = result["baseline"]["rank"]
        l = result["institution_lexical"]["rank"]
        i = result["integrated"]["rank"]

        print(
            f"{result['id']:<30} "
            f"baseline={str(b):>4} "
            f"lexical={str(l):>4} "
            f"integrated={str(i):>4}"
        )
        print(
            "  baseline   "
            f"{result['baseline']['recall']}"
        )
        print(
            "  lexical    "
            f"{result['institution_lexical']['recall']}"
        )
        print(
            "  integrated "
            f"{result['integrated']['recall']}"
        )

        print("  integrated top-5:")
        for rank, item in enumerate(result["integrated"]["top5"], 1):
            source = item["source"]
            text_preview = item["text"]
            print(f"    #{rank} {source}")
            print(f"       {text_preview}")
        print()

    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    s = summary
    print(
        "Baseline Recall:   "
        f"R@1={s['baseline']['top1_hits']}/{s['cases']}  "
        f"R@5={s['baseline']['top5_hits']}/{s['cases']}  "
        f"R@10={s['baseline']['top10_hits']}/{s['cases']}  "
        f"R@20={s['baseline']['top20_hits']}/{s['cases']}"
    )
    print(
        "Integrated Recall: "
        f"R@1={s['integrated']['top1_hits']}/{s['cases']}  "
        f"R@5={s['integrated']['top5_hits']}/{s['cases']}  "
        f"R@10={s['integrated']['top10_hits']}/{s['cases']}  "
        f"R@20={s['integrated']['top20_hits']}/{s['cases']}"
    )
    print(
        f"Regressions : {s['regressions'] or 'none'}"
    )
    print(
        f"Improvements: {s['improvements'] or 'none'}"
    )
    print(
        f"Rescues     : {s['rescues'] or 'none'}"
    )
    print(
        f"GATE        : {s['gate']}"
    )

    return 1 if s["gate"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
