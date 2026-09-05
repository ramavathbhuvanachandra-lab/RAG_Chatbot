"""IITJ V1 — Phase 8B real-world evaluation.

Run only through:

    python scripts/run_phase8b.py

Generated reports:

    reports/phase8/phase8b_summary.txt
    reports/phase8/phase8b_latest.json
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.graph import create_graph


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase8"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "phase8b_summary.txt"
)

JSON_PATH = (
    REPORT_DIR
    / "phase8b_latest.json"
)


# =========================================================
# Evaluation Cases
# =========================================================

CASES = [
    (
        "broad_programs",
        "broad",
        "What academic programs are available at IIT Jodhpur?",
    ),
    (
        "broad_departments",
        "broad",
        "What departments are there at IIT Jodhpur?",
    ),
    (
        "broad_research",
        "broad",
        "What research opportunities are available at IIT Jodhpur?",
    ),
    (
        "broad_facilities",
        "broad",
        "What facilities are available at IIT Jodhpur?",
    ),
    (
        "broad_admissions",
        "broad",
        "What admission opportunities are available at IIT Jodhpur?",
    ),
    (
        "narrow_mtech",
        "narrow",
        "What is the M.Tech eligibility?",
    ),
    (
        "narrow_phd",
        "narrow",
        "What are the regular Ph.D. eligibility requirements?",
    ),
    (
        "narrow_robotics",
        "narrow",
        "Which Electrical Engineering research area explicitly mentions robotics?",
    ),
    (
        "narrow_hostel",
        "narrow",
        "What are the short-term hostel rates?",
    ),
    (
        "narrow_msc",
        "narrow",
        "What are the admission routes for M.Sc.?",
    ),
]


# =========================================================
# Diagnostics
# =========================================================

INTERNAL_NOISE_PATTERNS = (
    r"\bcommand\s+\d+\b",
    r"\bdocument\s+\d+\b",
    r"\bretrieval representation\b",
)


def text(
    value: Any,
) -> str:
    return str(
        value or ""
    ).strip()


def source(
    document: Any,
) -> str:

    metadata = getattr(
        document,
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        return ""

    return text(
        metadata.get(
            "source",
            "",
        )
    )


def snippet(
    document: Any,
    limit: int = 260,
) -> str:

    value = text(
        getattr(
            document,
            "page_content",
            "",
        )
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    if len(value) > limit:
        return (
            value[:limit]
            + "..."
        )

    return value


def get_documents(
    result: dict[str, Any],
) -> list:

    for key in (
        "compressed_docs",
        "reranked_docs",
        "initial_reranked_docs",
        "fused_docs",
    ):

        value = result.get(
            key
        )

        if value:
            return list(
                value
            )

    return []


def get_queries(
    result: dict[str, Any],
) -> list[str]:

    for key in (
        "retrieval_queries",
        "generated_queries",
    ):

        value = result.get(
            key
        )

        if value:
            return [
                text(query)
                for query in value
                if text(query)
            ]

    return []


def detect_noise(
    answer: str,
) -> list[str]:

    lowered = answer.lower()

    return [
        pattern
        for pattern in INTERNAL_NOISE_PATTERNS
        if re.search(
            pattern,
            lowered,
        )
    ]


# =========================================================
# Known Scope Diagnostic
# =========================================================

def scope_warnings(
    case_id: str,
    answer: str,
) -> list[str]:

    lowered = answer.lower()

    warnings = []

    # M.Sc. route questions should not silently import unrelated
    # GATE/percentage eligibility claims from another admission rule.
    if case_id == "narrow_msc":

        suspicious_terms = [
            term
            for term in (
                "gate",
                "70%",
                "7.0",
                "gen/gen-ews/obc",
                "65%",
                "sc/st/pd",
            )
            if term in lowered
        ]

        if suspicious_terms:

            warnings.append(
                "M.Sc. route answer contains extra "
                "eligibility/GATE claims: "
                + ", ".join(
                    suspicious_terms
                )
            )

    return warnings


# =========================================================
# Case Evaluation
# =========================================================

def evaluate_case(
    graph,
    case_id: str,
    category: str,
    question: str,
) -> dict[str, Any]:

    started = time.perf_counter()

    try:

        result = graph.invoke(
            {
                "question": question,
                "chat_history": [],
            }
        )

    except Exception as exc:

        return {
            "id": case_id,
            "category": category,
            "question": question,
            "resolved_question": "",
            "retrieval_queries": [],
            "evidence_status": "",
            "evidence_score": None,
            "coverage_status": "",
            "question_type": "",
            "top_sources": [],
            "answer": "",
            "latency_s": round(
                time.perf_counter()
                - started,
                3,
            ),
            "answer_internal_noise": [],
            "scope_warnings": [],
            "status": "FAIL",
            "error": repr(
                exc
            ),
        }

    answer = text(
        result.get(
            "answer",
            "",
        )
    )

    evidence_status = text(
        result.get(
            "evidence_status",
            "",
        )
    )

    coverage_status = text(
        result.get(
            "evidence_coverage_status",
            "",
        )
    )

    evidence_score = result.get(
        "evidence_score"
    )

    question_type = text(
        result.get(
            "evidence_question_type",
            "",
        )
    )

    internal_noise = detect_noise(
        answer
    )

    scope_warning_list = scope_warnings(
        case_id,
        answer,
    )

    if internal_noise:

        status = "FAIL"

    elif not answer:

        status = "FAIL"

    elif scope_warning_list:

        status = "CHECK"

    elif (
        evidence_status == "supported"
        and coverage_status
        in {
            "supported",
            "partially_supported",
        }
    ):

        status = "PASS"

    else:

        status = "CHECK"

    documents = get_documents(
        result
    )

    return {
        "id": case_id,
        "category": category,
        "question": question,
        "resolved_question": text(
            result.get(
                "resolved_question",
                "",
            )
        ),
        "retrieval_queries": get_queries(
            result
        ),
        "evidence_status": evidence_status,
        "evidence_score": evidence_score,
        "coverage_status": coverage_status,
        "question_type": question_type,
        "top_sources": [
            {
                "source": source(document),
                "snippet": snippet(document),
            }
            for document in documents[:5]
        ],
        "answer": answer,
        "latency_s": round(
            time.perf_counter()
            - started,
            3,
        ),
        "answer_internal_noise": internal_noise,
        "scope_warnings": scope_warning_list,
        "status": status,
    }


# =========================================================
# Summary
# =========================================================

def build_summary(
    results: list[dict[str, Any]],
) -> str:

    total = len(
        results
    )

    passed = sum(
        item["status"] == "PASS"
        for item in results
    )

    checked = sum(
        item["status"] == "CHECK"
        for item in results
    )

    failed = sum(
        item["status"] == "FAIL"
        for item in results
    )

    broad = [
        item
        for item in results
        if item["category"] == "broad"
    ]

    narrow = [
        item
        for item in results
        if item["category"] == "narrow"
    ]

    average_latency = (
        sum(
            item["latency_s"]
            for item in results
        )
        / total
        if total
        else 0.0
    )

    maximum_latency = max(
        (
            item["latency_s"]
            for item in results
        ),
        default=0.0,
    )

    lines = [
        "PHASE 8B — REAL-WORLD EVALUATION",
        "=" * 72,
        (
            "Generated: "
            + datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        f"Cases: {total}",
        (
            f"PASS: {passed} | "
            f"CHECK: {checked} | "
            f"FAIL: {failed}"
        ),
        (
            "Broad: "
            f"{sum(x['status'] == 'PASS' for x in broad)}/"
            f"{len(broad)} PASS"
        ),
        (
            "Narrow: "
            f"{sum(x['status'] == 'PASS' for x in narrow)}/"
            f"{len(narrow)} PASS"
        ),
        f"Average latency: {average_latency:.2f}s",
        f"Max latency: {maximum_latency:.2f}s",
        "",
    ]

    for index, item in enumerate(
        results,
        start=1,
    ):

        lines.extend(
            [
                "-" * 72,
                (
                    f"[{index:02d}] "
                    f"{item['id']} | "
                    f"{item['status']}"
                ),
                (
                    f"QUESTION: "
                    f"{item['question']}"
                ),
                (
                    f"RESOLVED: "
                    f"{item['resolved_question'] or 'N/A'}"
                ),
                (
                    "RETRIEVAL: "
                    + (
                        " | ".join(
                            item[
                                "retrieval_queries"
                            ]
                        )
                        if item[
                            "retrieval_queries"
                        ]
                        else "N/A"
                    )
                ),
                (
                    f"EVIDENCE: "
                    f"{item['evidence_status'] or 'N/A'} "
                    f"(score={item['evidence_score']})"
                ),
                (
                    f"COVERAGE: "
                    f"{item['coverage_status'] or 'N/A'} "
                    f"(type={item['question_type'] or 'N/A'})"
                ),
                (
                    f"LATENCY: "
                    f"{item['latency_s']:.3f}s"
                ),
                "TOP SOURCES:",
            ]
        )

        for source_index, item_source in enumerate(
            item["top_sources"][:5],
            start=1,
        ):

            lines.append(
                f"  {source_index}. "
                f"{item_source['source'] or 'UNKNOWN SOURCE'}"
            )

            if item_source["snippet"]:

                lines.append(
                    "     "
                    + item_source[
                        "snippet"
                    ]
                )

        answer = (
            item["answer"]
            or "[EMPTY ANSWER]"
        )

        if len(answer) > 1000:

            answer = (
                answer[:1000]
                + "\n"
                + "[TRUNCATED — FULL ANSWER IN JSON]"
            )

        lines.extend(
            [
                "ANSWER:",
                answer,
            ]
        )

        if item["scope_warnings"]:

            lines.append(
                "SCOPE / GROUNDING WARNING:"
            )

            for warning in item[
                "scope_warnings"
            ]:

                lines.append(
                    f"  - {warning}"
                )

        if item[
            "answer_internal_noise"
        ]:

            lines.append(
                "ANSWER INTERNAL NOISE: "
                + ", ".join(
                    item[
                        "answer_internal_noise"
                    ]
                )
            )

        if item.get("error"):

            lines.append(
                "RUNTIME ERROR: "
                + item["error"]
            )

    lines.extend(
        [
            "",
            "=" * 72,
            "INTERPRETATION",
            "=" * 72,
            (
                "PASS = structurally healthy "
                "evidence/answer path."
            ),
            (
                "CHECK = pipeline worked but "
                "requires human factual/scope review."
            ),
            (
                "FAIL = runtime failure, "
                "empty answer, or internal leakage."
            ),
            (
                "This is not a factual accuracy score."
            ),
        ]
    )

    return "\n".join(
        lines
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "PHASE 8B — starting 10 real-world cases"
    )

    print(
        f"REPORT DIRECTORY: {REPORT_DIR}"
    )

    graph = create_graph()

    results = []

    for index, (
        case_id,
        category,
        question,
    ) in enumerate(
        CASES,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(CASES):02d}] "
            f"{case_id}"
        )

        results.append(
            evaluate_case(
                graph,
                case_id,
                category,
                question,
            )
        )

    payload = {
        "suite": (
            "phase8b_real_world"
        ),
        "generated_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "cases": results,
    }

    JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        build_summary(
            results
        ),
        encoding="utf-8",
    )

    print()
    print(
        "PHASE 8B REPORTS CREATED"
    )
    print(
        f"SUMMARY: {SUMMARY_PATH}"
    )
    print(
        f"JSON:    {JSON_PATH}"
    )


if __name__ == "__main__":
    main()
