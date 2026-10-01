"""Source-level retrieval recall audit for the reusable RAG core.

The normal E2E gate answers whether the pipeline produced a usable answer. This
script answers a different engineering question:

    Did the expected source actually make it into the ranked evidence candidates?

Usage:
    PYTHONPATH=. python tests_reusable/final_rag_core/retrieval_recall_audit.py \
        --report tests_reusable/phase3_real_rag/reports/node2_real_e2e_report.json \
        --gold tests_reusable/final_rag_core/retrieval_gold.json

Gold format:
    {
      "What are the M.Tech eligibility requirements?": [
        "admissions/mtech_admissions.docx"
      ]
    }

Matching is substring-based against the normalized source path so a gold file
can be portable across machines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _norm(value: Any) -> str:
    return _clean(value).casefold().replace("\\", "/")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("cases")
    if not isinstance(rows, list):
        raise ValueError("report must contain a list named 'cases'")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            question = _clean(row.get("question"))
            if question:
                output[question] = row
    return output


def _expected_sources(gold: Any, question: str) -> tuple[str, ...]:
    if not isinstance(gold, dict):
        raise ValueError("gold file must be a JSON object keyed by question")
    values = gold.get(question, ())
    if isinstance(values, str):
        values = (values,)
    return tuple(_norm(value) for value in values if _norm(value))


def _ranked_sources(case: dict[str, Any]) -> tuple[str, ...]:
    trace = (case.get("retrieval", {}) or {}).get("ranked_trace", ()) or ()
    output: list[str] = []
    seen: set[str] = set()
    for row in trace:
        if not isinstance(row, dict):
            continue
        source = _norm(row.get("source"))
        if not source or source in seen:
            continue
        seen.add(source)
        output.append(source)
    return tuple(output)


def _hit_at(ranked: Iterable[str], expected: Iterable[str], k: int) -> bool:
    top = tuple(ranked)[:k]
    for source in top:
        if any(exp in source or source.endswith(exp) for exp in expected):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source recall in the RAG ranking layer")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()

    report = _load_json(args.report)
    gold = _load_json(args.gold)
    cases = _cases(report)

    total = 0
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    no_label = 0

    print("RETRIEVAL SOURCE RECALL AUDIT")
    print("=" * 88)

    for question, case in cases.items():
        expected = _expected_sources(gold, question)
        if not expected:
            no_label += 1
            continue

        ranked = _ranked_sources(case)
        total += 1
        row_hits = {k: _hit_at(ranked, expected, k) for k in hits}
        for k, ok in row_hits.items():
            hits[k] += int(ok)

        print(f"QUESTION: {question}")
        print(f"  expected: {list(expected)}")
        print(f"  top_sources: {list(ranked[:10])}")
        print(
            "  hit@1={} hit@5={} hit@10={} hit@20={}".format(
                *["PASS" if row_hits[k] else "MISS" for k in (1, 5, 10, 20)]
            )
        )

    print("=" * 88)
    print(f"Labeled cases: {total}")
    print(f"Unlabeled report cases skipped: {no_label}")
    if total == 0:
        print("No labeled cases available; source recall cannot be measured yet.")
        return 2

    for k in (1, 5, 10, 20):
        print(f"Recall@{k}: {hits[k]}/{total} ({hits[k] / total:.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
