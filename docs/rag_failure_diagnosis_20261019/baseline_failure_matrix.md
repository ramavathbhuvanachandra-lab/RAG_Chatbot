# Baseline Failure Matrix (Q1–Q40)

**Total Questions Analyzed**: 40/40

| ID | Title | QU | Scope | Plan | Source | Retr | Retr Rank | Verif | Rank | Pack | Fact in Ctx | Raw Gen | Post-Proc | Cits | Final Ans | Earliest Stage & Function | Confidence | Repair / Action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | Aerospace vision | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PARTIAL | PARTIAL | PARTIAL | PASS | PASS | **PARTIAL** | Generation (generator.py:sentence_termination) | High | Fix terminal punct detection for numbered items |
| Q2 | Drone and anti-drone programme | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | FAIL | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Ranking (ranking.py:rank_candidates) | High | Fix ranking penalization in ranking.py |
| Q3 | Aerospace mission | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PARTIAL | PARTIAL | FAIL | FAIL | PASS | **FAIL** | Post-Processing (generator.py:_extractive_repair) | High | Fix extractive repair & regex terminal punctuation in generator.py |
| Q4 | Electronics focus areas | PASS | Correct | PARTIAL | Present | PASS | Rank 2 | PASS | PASS | PASS | YES | PARTIAL | PASS | PASS | **PARTIAL** | Generation (generator.py:sentence_termination) | High | Fix terminal punct detection for numbered items |
| Q5 | Electronics research | PASS | Correct | PASS | Present | PASS | Rank 5 | PASS | PASS | PASS | YES | PARTIAL | PASS | PASS | **PARTIAL** | Generation (generator.py:sentence_termination) | High | Fix terminal punct detection for numbered items |
| Q6 | Materials Engineering | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PASS | YES | PASS | PASS | PASS | **PASS** | N/A | High | None needed |
| Q7 | Materials Engineering minor options | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PASS | YES | FAIL | FAIL | PASS | **FAIL** | Post-Processing (generator.py:_extractive_repair) | High | Fix extractive repair & regex terminal punctuation in generator.py |
| Q8 | Training and Upskilling | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PASS | YES | PASS | PASS | PASS | **PASS** | N/A | High | None needed |
| Q9 | Mathematics and Computing | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PARTIAL | PARTIAL | PASS | PASS | PASS | **PARTIAL** | N/A | High | None needed |
| Q10 | Career pathways | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q11 | GeM training | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PASS | YES | FAIL | PASS | PASS | **FAIL** | Generation (generator.py / LLM prompt) | High | Simplify prompt guidance and enforce fact extraction |
| Q12 | GFR training | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PASS | YES | FAIL | PASS | PASS | **FAIL** | Generation (generator.py / LLM prompt) | High | Simplify prompt guidance and enforce fact extraction |
| Q13 | PMKVY training | FAIL | Incorrect | FAIL | Present | PASS | absent | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Query Understanding (planner.py:should_stop_pipeline) | High | Fix scope classifier & markers in planner.py |
| Q14 | Materials Engineering curriculum | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PASS | YES | PASS | PASS | PASS | **PASS** | N/A | High | None needed |
| Q15 | Mathematics learning outcomes | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PASS | YES | FAIL | PASS | PASS | **FAIL** | Generation (generator.py / LLM prompt) | High | Simplify prompt guidance and enforce fact extraction |
| Q16 | DADT learning outcomes | PASS | Correct | PASS | Present | PARTIAL | absent | PASS | PASS | FAIL | NO | PASS | PASS | PASS | **PASS** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q17 | Aerospace objectives versus DADT outcomes | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PARTIAL | PARTIAL | PASS | PASS | PASS | **PARTIAL** | N/A | High | None needed |
| Q18 | Electronics focus-area mapping | PASS | Correct | PARTIAL | Present | PASS | Rank 9 | PASS | PASS | PARTIAL | PARTIAL | PARTIAL | PASS | PASS | **PARTIAL** | Generation (generator.py:sentence_termination) | High | Fix terminal punct detection for numbered items |
| Q19 | Materials departmental versus interdisciplinary specialization | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PASS | YES | PARTIAL | PASS | PASS | **PARTIAL** | Generation (generator.py:sentence_termination) | High | Fix terminal punct detection for numbered items |
| Q20 | Institute UG credit structure | FAIL | Correct | FAIL | Present | PASS | absent | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Query Understanding (conversation.py:resolve_conversation) | High | Fix deictic program regex to require conversational antecedent in conversation.py |
| Q21 | B.S. Mathematics and Computing credits | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PASS | YES | PASS | PASS | PASS | **PASS** | N/A | High | None needed |
| Q22 | Comparing two 72-credit totals | PASS | Correct | PASS | Present | PASS | Rank 1 (unfused) | PASS | PASS | FAIL | NO | PASS | PASS | PASS | **PASS** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q23 | Mathematics elective groups | FAIL | Correct | FAIL | Present | PASS | absent | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Query Understanding (conversation.py:resolve_conversation) | High | Fix deictic program regex to require conversational antecedent in conversation.py |
| Q24 | Course codes | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PASS | YES | PARTIAL | PASS | FAIL | **PARTIAL** | Generation (generator.py:sentence_termination) | High | Fix terminal punct detection for numbered items |
| Q25 | PMKVY aggregate calculation | FAIL | Incorrect | FAIL | Present | PASS | absent | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Query Understanding (planner.py:should_stop_pipeline) | High | Fix scope classifier & markers in planner.py |
| Q26 | GeM versus GFR | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | PASS | YES | PASS | PASS | PASS | **PASS** | N/A | High | None needed |
| Q27 | Electronics graduate attributes | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PASS | YES | PASS | PASS | PASS | **PASS** | N/A | High | None needed |
| Q28 | Materials B.Tech. versus Ph.D. | PASS | Correct | PASS | Present | PASS | Rank 7 | PASS | PASS | PARTIAL | PARTIAL | PASS | PASS | PASS | **PARTIAL** | N/A | High | None needed |
| Q29 | Mathematics objectives and career pathways | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | PASS | YES | PARTIAL | PASS | PASS | **PARTIAL** | Generation (generator.py:sentence_termination) | High | Fix terminal punct detection for numbered items |
| Q30 | Electronics versus Materials Engineering | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q31 | Missing DADT admission details | PASS | Correct | PASS | Present | PASS | Rank 3 (unfused) | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q32 | B.S. Mathematics and Computing admissions | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q33 | Historical versus current training dates | PASS | Correct | PASS | Present | PASS | Rank 5 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q34 | Ambiguous graduation-credit query | PASS | Correct | PASS | Present | PARTIAL | absent | PASS | PASS | PARTIAL | PARTIAL | FAIL | PASS | PASS | **FAIL** | Generation (generator.py / LLM prompt) | High | Simplify prompt guidance and enforce fact extraction |
| Q35 | Materials M.Tech. eligibility | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q36 | PMKVY duration false premise | FAIL | Incorrect | FAIL | Present | PASS | absent | PASS | PASS | FAIL | NO | PARTIAL | PASS | PASS | **FAIL** | Query Understanding (planner.py:should_stop_pipeline) | High | Fix scope classifier & markers in planner.py |
| Q37 | Incomplete Mathematics course codes | PASS | Correct | PASS | Present | PASS | Rank 3 | PASS | PASS | PARTIAL | PARTIAL | FAIL | PASS | PASS | **FAIL** | Generation (generator.py / LLM prompt) | High | Simplify prompt guidance and enforce fact extraction |
| Q38 | Arithmetic trap | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q39 | Semiconductor chip-design recommendation | PASS | Correct | PASS | Present | PASS | Rank 6 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |
| Q40 | Incomplete Training programme information | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | PARTIAL | PASS | PASS | **PARTIAL** | Packaging (evidence_package.py) | High | Fix context packaging budget and chunk survival |

## Summary Metrics
- **PASS**: 8 (20.0%)
- **PARTIAL**: 11 (27.5%)
- **FAIL**: 21 (52.5%)

## Confirmed Stage Breakdown of Earliest Root Causes
- **Packaging (evidence_package.py)**: 11 questions
- **N/A**: 9 questions
- **Generation (generator.py**: 7 questions
- **Generation (generator.py / LLM prompt)**: 5 questions
- **Query Understanding (planner.py**: 3 questions
- **Post-Processing (generator.py**: 2 questions
- **Query Understanding (conversation.py**: 2 questions
- **Ranking (ranking.py**: 1 questions
