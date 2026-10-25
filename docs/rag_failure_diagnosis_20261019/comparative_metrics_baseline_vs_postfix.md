# Baseline vs. Post-Fix 40-Question Comparative Metrics

| Evaluation Metric | Baseline (Pre-Fix) | Post-Fix (Repaired) | Absolute Delta | Relative Improvement |
|---|---|---|---|---|
| **Clean PASS Rate** | 8/40 (20.0%) | 12/40 (30.0%) | +4 | +50.0% |
| **Partial / Impaired Rate** | 11/40 (27.5%) | 17/40 (42.5%) | +6 | Converted from FAIL |
| **FAIL Rate** | 21/40 (52.5%) | 11/40 (27.5%) | -10 | **-47.6% (47.6% drop)** |
| **Query Understanding / Scope Stops** | 5/40 (12.5%) | 0/40 (0.0%) | -5 | **-100% (all resolved)** |
| **False Refusals on Valid Queries** | 5/5 (100.0%) | 0/5 (0.0%) | -5 | **100% Eliminated** |
| **Candidate Ranking Survival (>0 cands)** | 34/40 (85.0%) | 40/40 (100.0%) | +6 | +17.6% |
| **Average Citations per Turn** | 2.65 | 2.30 | -0.35 | Controlled Overlap |
| **Average Turn Latency** | 35.32s | 47.69s | +12.38s | Neutral within bound |

## Key Diagnosed Defect Verifications

| Test ID | Defect Category | Baseline Result | Post-Fix Result | Status |
|---|---|---|---|---|
| **Q2** | DADT Proposal Year | Hallucinated 2026-27 (Rank demoted) | Grounded 2025-26, cited Aerospace doc | **REPAIRED** |
| **Q7** | Extractive Repair Header Leak | Leaked `######` tag and fragmented text | Clean markdown list of both minors | **REPAIRED** |
| **Q10** | Career Pathways Budget Cut | Dropped career pathways chunk | Included pathways chunk & generated list | **REPAIRED** |
| **Q13** | False Scope Refusal on PMKVY | Hard refusal by planner scope guard | Answered all 4 roles, enrolments & OJT | **REPAIRED** |
| **Q20** | Deictic 'the degree' Clarification | False clarification refusal on standalone query | Retrieved UG table, answered 141 graded / 153 total | **REPAIRED** |
| **Q23** | Deictic 'the course' Clarification | False clarification refusal on standalone query | Retrieved 3 elective groups, preserved exact codes | **REPAIRED** |
| **Q24** | Incomplete Course Codes | Truncated list at `4.` prefix | Preserved CS7XXX, CSLXXXX, QCL4020 | **REPAIRED** |
| **Q25** | PMKVY Aggregate Calculation | False scope refusal | Enrolments: 104, Certifications: 62 verified | **REPAIRED** |
| **Q30** | Cross-Department Verification Block | 0 candidates survived verification | Electronics & Materials both verified, synthesized | **REPAIRED** |
| **Q38** | Arithmetic Agreement Trap | Sycophantly agreed 42+12+6+2+12=72 | Corrected: 42+12+6+2+12=74 != 72 | **REPAIRED** |
