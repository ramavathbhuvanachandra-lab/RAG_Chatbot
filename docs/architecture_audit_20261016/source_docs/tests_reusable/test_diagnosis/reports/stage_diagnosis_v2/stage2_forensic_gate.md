# Stage 2 — Verification Forensic Gate

> [!NOTE]
> **Historical Forensic Record**: This document captures early Stage 2 candidate tracking forensics.
> For current canonical platform state and benchmark results, see [AI_PLATFORM_MASTER_HANDOFF.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/AI_PLATFORM_MASTER_HANDOFF.md).

Cases analyzed: **15**

## Decision

Stage 2 is treated as the active gate only if the same candidate key is tracked from RRF → verification → ranking.
A broad benchmark label is not enough to call a verification failure, because another candidate may also match the benchmark oracle.

## Per-case findings

| Case | Expected candidates | Verified | Uncertain | Rejected | Same-key destructive? | Grouped docs | Scope-filtered docs |
|---|---:|---:|---:|---:|:---:|---:|---:|
| mtech_regular_eligibility | 0 | 0 | 0 | 0 | no | — | — |
| mtech_percentage | 0 | 0 | 0 | 0 | no | — | — |
| mtech_gate | 0 | 0 | 0 | 0 | no | — | — |
| mtech_qualification | 0 | 0 | 0 | 0 | no | — | — |
| mtech_written_test | 0 | 0 | 0 | 0 | no | — | — |
| mtech_work_experience | 0 | 0 | 0 | 0 | no | — | — |
| msc_eligibility | 0 | 0 | 0 | 0 | no | — | — |
| msc_percentage | 0 | 0 | 0 | 0 | no | — | — |
| msc_jam | 0 | 0 | 0 | 0 | no | — | — |
| msc_bachelor_route | 0 | 0 | 0 | 0 | no | — | — |
| phd_eligibility | 0 | 0 | 0 | 0 | no | — | — |
| phd_application_fee | 0 | 0 | 0 | 0 | no | — | — |
| mba_steps | 0 | 0 | 0 | 0 | no | — | — |
| registration_documents | 0 | 0 | 0 | 0 | no | — | — |
| registration_provisional | 0 | 0 | 0 | 0 | no | — | — |

## Verification rejection reasons

- none

## Gate metrics

- Expected-candidate statuses: {}
- Cases with an expected candidate rejected: **0**
- Cases where an expected candidate was rejected/uncertain and did not survive as the same ranked key: **0**
- Cases where evidence grouping produced documents but scope filtering reduced the set to zero: **0**

## Stage-2 rebuild rule

Do not loosen verification just to increase recall. Rebuild only after the destructive same-key failures are confirmed from candidate text + query frame + verification decision reasons.

Recommended boundary after rebuild:
`fused_candidates -> verify_candidates -> verified_candidates/uncertain_candidates/rejected_candidates -> rank verified candidates only`.

## Stage-3 rule

Ranking is frozen until Stage 2 passes. Once Stage 2 passes, rerun at least 15 adversarial cases and evaluate same-key top-1/top-k preservation without allowing ranking to resurrect rejected candidates.

## Explicit non-goals

- No backend imports.
- No answer LLM.
- No production code edits.
- No benchmark changes to make failures disappear.
