# RAG V3 Validation Report

> [!NOTE]
> **Historical Validation Report**: This report captures the October 4, 2026 validation of the RAG V3 bundle.
> For current active test suites and canonical benchmark baselines, see [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

Date: 2026-10-04

## Component + real-source validation

```text
22 passed in 1.42s
```

Tested:

- simplified verification unit tests
- adversarial verification tests
- final-context builder tests
- 4 real-source context integration cases
- 15 real IIT Jodhpur question/context cases

Aggregate over the 15 deterministic real-source cases:

```text
candidate decisions: 66
verified:            59
rejected:             7
uncertain:            0
cases with context: 15/15
cases with empty context: 0/15
```

## Active graph contract

```text
3 passed
```

The contract test proves:

- the active graph points to `ai_platform.core.retrieval.verification`
- `retrieve_and_qualify_node` no longer calls the old evidence/coverage/claim/package gates
- `evidence_context_node` calls the final context builder and does not re-verify or scope-filter
- `answer_node` keeps the answer generator, grounding, and guard stages

## Installer validation

The installer was syntax-checked and run against an old graph snapshot.

```text
PASS: compile
3 passed
INSTALLER_CHECK=PASS
```

## Two real pre-LLM context inspections

### Question 1

`What are the eligibility requirements for regular M.Tech admission?`

```text
VERIFIED: 2
CONTEXT CHARS: 860
```

The final context contains the regular M.Tech percentage/CGPA requirement, qualifying degree requirement, and GATE/written-test requirement. The obvious dual-degree/JAM/work-experience noise was removed from the final prompt context.

### Question 2

`What documents are required for registration?`

```text
VERIFIED: 5
CONTEXT CHARS: 247
```

The final context contains the required certificates, JEE score cards, proof of address, identity proof, caste/domicile certificate, and photographs.

## What this proves

The pre-LLM path now has a clean responsibility split:

```text
Verification   = can this candidate participate?
Ranking        = which approved candidate is most useful?
Context Builder= which parts of approved candidates should the LLM read?
Grounding      = did the generated answer stay grounded?
Guard          = is the final response acceptable?
```

The old evidence/coverage/claim-audit machinery is no longer a generation blocker on the single-intent path.

## Remaining boundary

This validation deliberately does not claim that the live Chroma/BM25/Ollama stack is perfect. The 15 real-source checks validate the verification/context layer against actual IIT Jodhpur source documents. A live end-to-end run should still be performed after installation before submission.
