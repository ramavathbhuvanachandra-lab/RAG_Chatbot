from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai_platform.runtime.retrieval import (
    corpus_document_count,
    dense_retrieve,
    keyword_retrieve,
)

from ai_platform.core.retrieval.rrf import fuse_ranked_lists
from ai_platform.core.retrieval.semantic_alignment import (
    align_query_to_document,
)
from ai_platform.core.retrieval.verification import (
    verify_candidates,
    verify_candidate,
)
from ai_platform.core.retrieval.contracts import RetrievalCandidate

from ai_platform.core.query.models import (
    QueryFacet,
    RetrievalRequirement,
    SemanticQueryFrame,
)


OUTPUT_DIR = Path("tests_reusable/test_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "forensic_verifier_branch_trace.json"


CASES = (
    {
        "name": "MTECH_ELIGIBILITY",
        "question": (
            "What are the eligibility requirements "
            "for regular M.Tech admission?"
        ),
        "target": "M.Tech",
        "request_type": "eligibility",
        "facets": (
            "eligibility",
            "requirements",
        ),
    },
    {
        "name": "MTECH_WRITTEN_TEST",
        "question": (
            "Is there a written test for "
            "regular M.Tech admission?"
        ),
        "target": "M.Tech",
        "request_type": "written_test",
        "facets": (
            "written test",
            "admission",
        ),
    },
    {
        "name": "MSC_ELIGIBILITY",
        "question": (
            "What are the eligibility requirements "
            "for M.Sc admission?"
        ),
        "target": "M.Sc",
        "request_type": "eligibility",
        "facets": (
            "eligibility",
            "requirements",
        ),
    },
    {
        "name": "PHD_ELIGIBILITY",
        "question": (
            "What are the eligibility requirements "
            "for Ph.D admission?"
        ),
        "target": "Ph.D",
        "request_type": "eligibility",
        "facets": (
            "eligibility",
            "requirements",
        ),
    },
    {
        "name": "REGISTRATION_DOCUMENTS",
        "question": (
            "What documents are required for registration?"
        ),
        "target": "registration",
        "request_type": "registration",
        "facets": (
            "documents",
            "registration",
        ),
    },
)


def make_frame(
    case: dict[str, Any],
) -> SemanticQueryFrame:

    return SemanticQueryFrame(
        original_query=case["question"],
        normalized_query=case["question"],
        semantic_query=case["question"],
        target=case["target"],
        request_type=case["request_type"],
        facets=tuple(
            QueryFacet(
                name="requested_attribute",
                value=value,
            )
            for value in case["facets"]
        ),
        requirement=RetrievalRequirement(
            mode="standard",
            require_target_alignment=True,
            require_attribute_alignment=True,
            allow_partial_evidence=True,
            reject_explicit_conflict=True,
        ),
    )


def source_of(
    candidate: RetrievalCandidate,
) -> str:

    source = getattr(
        candidate,
        "source",
        None,
    )

    if source:
        return str(source)

    metadata = getattr(
        getattr(
            candidate,
            "document",
            None,
        ),
        "metadata",
        {},
    ) or {}

    return str(
        metadata.get("source", "")
        or metadata.get("file_path", "")
        or ""
    )


def text_of(
    candidate: RetrievalCandidate,
) -> str:

    document = getattr(
        candidate,
        "document",
        None,
    )

    return str(
        getattr(
            document,
            "page_content",
            "",
        )
        or ""
    )


def compact(
    value: Any,
    limit: int = 260,
) -> str:

    value = " ".join(
        str(value or "").split()
    )

    if len(value) <= limit:
        return value

    return value[:limit] + "..."


def scalar(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


# ---------------------------------------------------------------------------
# Build a map of every return statement in verify_candidate()
# together with its enclosing if-condition path.
# ---------------------------------------------------------------------------

VERIFY_FILE = Path(
    inspect.getsourcefile(verify_candidate)
).resolve()

VERIFY_SOURCE = VERIFY_FILE.read_text(
    encoding="utf-8"
)

VERIFY_LINES = VERIFY_SOURCE.splitlines()


def build_return_branch_map() -> dict[int, dict[str, Any]]:

    tree = ast.parse(
        VERIFY_SOURCE,
        filename=str(VERIFY_FILE),
    )

    function_node = None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "verify_candidate"
        ):
            function_node = node
            break

    if function_node is None:
        raise RuntimeError(
            "Could not find verify_candidate() in source."
        )

    result: dict[int, dict[str, Any]] = {}

    def visit(
        node: ast.AST,
        conditions: list[str],
    ) -> None:

        if isinstance(node, ast.If):

            condition = ast.unparse(
                node.test
            )

            # Body inherits this condition.
            for child in node.body:
                visit(
                    child,
                    conditions + [condition],
                )

            # Else branch carries explicit negation.
            for child in node.orelse:
                visit(
                    child,
                    conditions + [
                        f"NOT ({condition})"
                    ],
                )

            return

        if isinstance(node, ast.Return):

            value = node.value

            is_verification_return = (
                isinstance(value, ast.Call)
                and isinstance(
                    value.func,
                    ast.Name,
                )
                and value.func.id
                == "VerificationDecision"
            )

            if is_verification_return:
                result[node.lineno] = {
                    "return_line": node.lineno,
                    "conditions": conditions.copy(),
                    "source": (
                        VERIFY_LINES[
                            node.lineno - 1
                        ].strip()
                    ),
                }

            return

        for child in ast.iter_child_nodes(node):
            visit(
                child,
                conditions,
            )

    visit(
        function_node,
        [],
    )

    return result


RETURN_BRANCH_MAP = build_return_branch_map()


def nearest_return_branch(
    line_no: int,
) -> dict[str, Any]:

    if line_no in RETURN_BRANCH_MAP:
        return RETURN_BRANCH_MAP[line_no]

    candidates = [
        line
        for line in RETURN_BRANCH_MAP
        if line <= line_no
    ]

    if not candidates:
        candidates = list(
            RETURN_BRANCH_MAP
        )

    nearest = min(
        candidates,
        key=lambda line: abs(
            line - line_no
        ),
    )

    return RETURN_BRANCH_MAP[
        nearest
    ]


# ---------------------------------------------------------------------------
# Runtime tracer
# ---------------------------------------------------------------------------

RELEVANT_LOCALS = (
    "mode",
    "strict_exact_request",
    "exact_request",
    "requires_target",
    "requires_attribute",
    "requires_scope",
    "target_grounded",
    "attribute_grounded",
    "target_attribute_relation",
    "controlled_support",
    "controlled_family_requested",
    "placeholder_exact_support",
    "lexical_rescue",
    "lexical_score",
    "semantic_compatible",
    "semantic_score",
    "conflict_detected",
    "scope_compatible",
    "coverage",
    "resolved_entities",
    "entity_grounded",
)


class BranchTracer:

    def __init__(self) -> None:

        self.events: list[
            dict[str, Any]
        ] = []

        self._candidate_state: dict[
            int,
            dict[str, Any]
        ] = {}

    def tracer(
        self,
        frame,
        event,
        arg,
    ):

        if (
            frame.f_code.co_filename
            != str(VERIFY_FILE)
        ):
            return self.tracer

        if (
            frame.f_code.co_name
            != "verify_candidate"
        ):
            return self.tracer

        frame_key = id(frame)

        if event == "call":

            candidate = frame.f_locals.get(
                "candidate"
            )

            candidate_id = str(
                getattr(
                    candidate,
                    "document_id",
                    "",
                )
                or ""
            )

            self._candidate_state[
                frame_key
            ] = {
                "candidate_id": candidate_id,
                "line_history": [],
                "return_branch_line": None,
                "locals_at_return_branch": {},
            }

            return self.tracer

        state = self._candidate_state.get(
            frame_key
        )

        if state is None:
            return self.tracer

        if event == "line":

            line_no = frame.f_lineno

            state[
                "line_history"
            ].append(line_no)

            source_line = (
                VERIFY_LINES[
                    line_no - 1
                ].strip()
                if 1 <= line_no
                <= len(VERIFY_LINES)
                else ""
            )

            # Capture the exact return statement
            # before it executes.
            if source_line.startswith(
                "return VerificationDecision"
            ):

                state[
                    "return_branch_line"
                ] = line_no

                state[
                    "locals_at_return_branch"
                ] = {
                    name: scalar(
                        frame.f_locals.get(
                            name
                        )
                    )
                    for name
                    in RELEVANT_LOCALS
                }

        elif event == "return":

            candidate_id = state[
                "candidate_id"
            ]

            branch_line = state[
                "return_branch_line"
            ]

            returned_status = (
                getattr(
                    arg,
                    "status",
                    None,
                )
                if arg is not None
                else None
            )

            if branch_line is not None:

                branch = nearest_return_branch(
                    branch_line
                )

                conditions = branch[
                    "conditions"
                ]

            else:

                conditions = []

            self.events.append(
                {
                    "candidate_id": candidate_id,
                    "status": scalar(
                        returned_status
                    ),
                    "return_line": branch_line,
                    "conditions": conditions,
                    "locals": state[
                        "locals_at_return_branch"
                    ],
                    "line_history_tail": (
                        state[
                            "line_history"
                        ][-20:]
                    ),
                }
            )

            self._candidate_state.pop(
                frame_key,
                None,
            )

        return self.tracer


def run_with_trace(
    frame: SemanticQueryFrame,
    candidates: tuple[RetrievalCandidate, ...],
) -> tuple[Any, list[dict[str, Any]]]:

    tracer = BranchTracer()

    old_trace = sys.gettrace()

    try:

        sys.settrace(
            tracer.tracer
        )

        batch = verify_candidates(
            frame,
            candidates,
        )

    finally:

        sys.settrace(
            old_trace
        )

    return (
        batch,
        tracer.events,
    )


# ---------------------------------------------------------------------------
# Retrieval + alignment
# ---------------------------------------------------------------------------

def prepare_candidates(
    question: str,
    fused: tuple[RetrievalCandidate, ...],
) -> tuple[RetrievalCandidate, ...]:

    registry_module = __import__(
        "ai_platform.institutions.iitj.semantic_registry",
        fromlist=[
            "SEMANTIC_REGISTRY"
        ],
    )

    registry = getattr(
        registry_module,
        "SEMANTIC_REGISTRY",
    )

    aligned = []

    for candidate in fused:

        (
            _,
            document_meaning,
            alignment,
        ) = align_query_to_document(
            question,
            candidate.document,
            registry,
        )

        aligned.append(
            replace(
                candidate,
                meaning=document_meaning,
                alignment=alignment,
            )
        )

    return tuple(aligned)


# ---------------------------------------------------------------------------
# Forensic analysis
# ---------------------------------------------------------------------------

def analyse_case(
    case: dict[str, Any],
) -> dict[str, Any]:

    question = case["question"]

    frame = make_frame(
        case
    )

    dense = tuple(
        dense_retrieve(
            question
        )
    )

    bm25 = tuple(
        keyword_retrieve(
            question
        )
    )

    fused = tuple(
        fuse_ranked_lists(
            [dense, bm25],
            weights=(0.70, 0.30),
            primary_query=question,
            retrieval_queries=(question,),
        )
    )

    aligned = prepare_candidates(
        question,
        fused,
    )

    batch, trace_events = run_with_trace(
        frame,
        aligned,
    )

    event_map = {
        event["candidate_id"]: event
        for event in trace_events
    }

    decision_map = {
        decision.candidate.document_id: decision
        for decision
        in batch.decisions
    }

    rows = []

    for fused_rank, candidate in enumerate(
        aligned,
        start=1,
    ):

        decision = decision_map[
            candidate.document_id
        ]

        event = event_map.get(
            candidate.document_id,
            {},
        )

        status = str(
            decision.status
        )

        locals_at_branch = event.get(
            "locals",
            {},
        )

        branch_conditions = event.get(
            "conditions",
            [],
        )

        weak_verified = (
            status == "verified"
            and (
                not bool(
                    getattr(
                        decision,
                        "target_grounded",
                        False,
                    )
                )
                or not bool(
                    getattr(
                        decision,
                        "attribute_grounded",
                        False,
                    )
                )
                or (
                    "target_attribute_relation_supported"
                    not in tuple(
                        getattr(
                            decision,
                            "reasons",
                            (),
                        )
                    )
                )
                or not bool(
                    getattr(
                        decision,
                        "semantic_compatible",
                        False,
                    )
                )
                or float(
                    getattr(
                        decision,
                        "coverage",
                        0.0,
                    )
                    or 0.0
                ) < 0.50
            )
        )

        rows.append(
            {
                "fused_rank": fused_rank,
                "candidate_id": candidate.document_id,
                "source": source_of(
                    candidate
                ),
                "status": status,
                "return_line": event.get(
                    "return_line"
                ),
                "branch_conditions": (
                    branch_conditions
                ),
                "runtime_locals_at_return_branch": (
                    locals_at_branch
                ),
                "target_grounded": bool(
                    getattr(
                        decision,
                        "target_grounded",
                        False,
                    )
                ),
                "attribute_grounded": bool(
                    getattr(
                        decision,
                        "attribute_grounded",
                        False,
                    )
                ),
                "semantic_compatible": bool(
                    getattr(
                        decision,
                        "semantic_compatible",
                        False,
                    )
                ),
                "conflict_detected": bool(
                    getattr(
                        decision,
                        "conflict_detected",
                        False,
                    )
                ),
                "scope_compatible": bool(
                    getattr(
                        decision,
                        "scope_compatible",
                        False,
                    )
                ),
                "coverage": float(
                    getattr(
                        decision,
                        "coverage",
                        0.0,
                    )
                    or 0.0
                ),
                "verifier_score": float(
                    getattr(
                        decision,
                        "score",
                        0.0,
                    )
                    or 0.0
                ),
                "reasons": list(
                    getattr(
                        decision,
                        "reasons",
                        (),
                    )
                    or ()
                ),
                "weak_verified": weak_verified,
                "text": compact(
                    text_of(candidate)
                ),
            }
        )

    return {
        "case": case,
        "corpus_chunks": corpus_document_count(),
        "dense_hits": len(dense),
        "bm25_hits": len(bm25),
        "rrf_candidates": len(fused),
        "verification": {
            "verified": len(
                batch.verified
            ),
            "uncertain": len(
                batch.uncertain
            ),
            "rejected": len(
                batch.rejected
            ),
        },
        "trace_event_count": len(
            trace_events
        ),
        "candidates": rows,
    }


def print_candidate(
    row: dict[str, Any],
) -> None:

    print()
    print(
        f"[FUSED #{row['fused_rank']:02d}] "
        f"{row['status'].upper()}"
        + (
            " <<< WEAK VERIFIED"
            if row["weak_verified"]
            else ""
        )
    )

    print(
        f"  candidate_id        : "
        f"{row['candidate_id']}"
    )

    print(
        f"  source              : "
        f"{row['source']}"
    )

    print(
        f"  RETURN LINE         : "
        f"{row['return_line']}"
    )

    print(
        f"  target_grounded     : "
        f"{row['target_grounded']}"
    )

    print(
        f"  attribute_grounded  : "
        f"{row['attribute_grounded']}"
    )

    print(
        f"  semantic_compatible : "
        f"{row['semantic_compatible']}"
    )

    print(
        f"  conflict_detected   : "
        f"{row['conflict_detected']}"
    )

    print(
        f"  scope_compatible    : "
        f"{row['scope_compatible']}"
    )

    print(
        f"  coverage            : "
        f"{row['coverage']:.4f}"
    )

    print(
        f"  verifier_score      : "
        f"{row['verifier_score']:.4f}"
    )

    print(
        f"  reasons             : "
        f"{', '.join(row['reasons'])}"
    )

    print(
        "  BRANCH CONDITIONS   :"
    )

    for condition in row[
        "branch_conditions"
    ]:
        print(
            f"      {condition}"
        )

    print(
        "  RUNTIME SWITCHES   :"
    )

    for key, value in row[
        "runtime_locals_at_return_branch"
    ].items():
        print(
            f"      {key:<32} = {value}"
        )

    print(
        f"  text                : "
        f"{row['text']}"
    )


def print_case(
    result: dict[str, Any],
) -> None:

    case = result["case"]

    print()
    print("=" * 120)
    print(
        f"{case['name']}"
    )
    print("=" * 120)

    print(
        f"Question       : "
        f"{case['question']}"
    )

    print(
        f"Corpus         : "
        f"{result['corpus_chunks']}"
    )

    print(
        f"Dense/BM25/RRF : "
        f"{result['dense_hits']}/"
        f"{result['bm25_hits']}/"
        f"{result['rrf_candidates']}"
    )

    print(
        f"Verification   : "
        f"verified={result['verification']['verified']} "
        f"uncertain={result['verification']['uncertain']} "
        f"rejected={result['verification']['rejected']}"
    )

    print(
        f"Trace events   : "
        f"{result['trace_event_count']}"
    )

    print()
    print(
        "ALL CANDIDATE VERIFICATION BRANCHES"
    )

    for row in result[
        "candidates"
    ]:
        print_candidate(
            row
        )


def aggregate(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    summary = []

    for result in results:

        rows = result[
            "candidates"
        ]

        verified = [
            row
            for row in rows
            if row["status"]
            == "verified"
        ]

        branch_counts: dict[
            str,
            int
        ] = {}

        for row in rows:

            line = str(
                row["return_line"]
            )

            branch_counts[
                line
            ] = (
                branch_counts.get(
                    line,
                    0,
                )
                + 1
            )

        weak = [
            row
            for row in verified
            if row["weak_verified"]
        ]

        summary.append(
            {
                "case": result[
                    "case"
                ]["name"],
                "rrf_candidates": result[
                    "rrf_candidates"
                ],
                "verified": result[
                    "verification"
                ]["verified"],
                "uncertain": result[
                    "verification"
                ]["uncertain"],
                "rejected": result[
                    "verification"
                ]["rejected"],
                "weak_verified": len(
                    weak
                ),
                "return_line_counts": branch_counts,
            }
        )

    return {
        "cases": len(
            results
        ),
        "source_file": str(
            VERIFY_FILE
        ),
        "verification_function": (
            "verify_candidate"
        ),
        "summary": summary,
    }


def main() -> None:

    print(
        "IITJ VERIFIER BRANCH TRACE"
    )

    print(
        "NO production files are modified."
    )

    print(
        f"Tracing: {VERIFY_FILE}"
    )

    print(
        f"verify_candidate() return branches found: "
        f"{len(RETURN_BRANCH_MAP)}"
    )

    all_results = []

    for case in CASES:

        result = analyse_case(
            case
        )

        print_case(
            result
        )

        all_results.append(
            result
        )

    payload = {
        "test": (
            "IITJ 5-case runtime "
            "verify_candidate branch trace"
        ),
        "source_file": str(
            VERIFY_FILE
        ),
        "return_branch_map": {
            str(k): v
            for k, v in sorted(
                RETURN_BRANCH_MAP.items()
            )
        },
        "aggregate": aggregate(
            all_results
        ),
        "results": all_results,
    }

    REPORT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print(
        "BRANCH TRACE AGGREGATE"
    )
    print("=" * 120)

    for item in payload[
        "aggregate"
    ]["summary"]:

        print()
        print(
            item["case"]
        )

        print(
            f"  RRF={item['rrf_candidates']} "
            f"Verified={item['verified']} "
            f"Uncertain={item['uncertain']} "
            f"Rejected={item['rejected']} "
            f"WeakVerified={item['weak_verified']}"
        )

        print(
            "  RETURN-LINE COUNTS:"
        )

        for line, count in sorted(
            item[
                "return_line_counts"
            ].items(),
            key=lambda x: int(
                x[0]
            ),
        ):
            print(
                f"      line {line}: {count}"
            )

    print()
    print(
        f"JSON REPORT: {REPORT_JSON}"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
