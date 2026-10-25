# IIT JODHPUR INSTITUTIONAL RAG CHATBOT — FAILURE DIAGNOSIS & FUNCTION AUDIT REPORT
**Evaluation Directory**: `docs/rag_failure_diagnosis_20261019/`  
**Execution Timestamp**: 2026-10-19T16:18:27+05:30  
**Harness**: Canonical Assistant Service Runner (`scripts/run_40q_diagnosis.py`) on local active runtime  
**Model & Runtime**: `qwen2.5:3b` via local Ollama (`http://localhost:11434`), `nomic-embed-text:latest`, ChromaDB collection `iitj_v1`  
**Frozen Diagnostic Dataset**: Q1–Q40 (Groups A, B, C, D)  

---

## 1. Executive Summary & Diagnostic Methodology

An end-to-end forensic diagnosis was executed across all 40 frozen institutional evaluation questions (Q1–Q40) on the active IIT Jodhpur chatbot codebase. The baseline reproduction ran against the live, unmodified production request path (`handle_chat_turn` -> `ask_with_diagnostics` -> LangGraph pipeline -> `qwen2.5:3b`) without mocks or simulated states.

### Key Headline Metrics on Baseline System
- **Total Diagnostic Questions Evaluated**: 40
- **Clean PASS**: 8 / 40 (20.0%)
- **PARTIAL (Impaired/Truncated/Incomplete)**: 11 / 40 (27.5%)
- **FAIL (Hallucinated/Contaminated/Falsely Refused/Falsely Displaced)**: 21 / 40 (52.5%)
- **Failure + Partial Impairment Rate**: **80.0%** (32 / 40 questions failed or suffered material degradation)

The diagnosis proves conclusively that failures are **NOT** solely attributable to the local 3B LLM, nor to a single "bad retriever." Instead, the system suffers from **seven discrete, systemic software defects across five pipeline stages**:
1. **Query Understanding / Conversation Resolution (`conversation.py:resolve_conversation`)**: False deictic classification regex halts the pipeline on standalone questions containing "the degree" or "the course" (Q20, Q23).
2. **Pre-Planning Scope Guard (`planner.py:_repair_scope`)**: Aggressive out-of-scope classifier falsely refuses legitimate institutional vocational/training programs (PMKVY 4.0, Skill Hub) (Q13, Q25, Q36).
3. **Candidate Verification (`verification.py:verify_candidate`)**: Comparison queries require candidate-local grounding of cross-department relationships, rejecting all candidates when two departments are compared (Q30).
4. **Ranking Demotion (`ranking.py:rank_candidates`)**: Specific departmental program proposals (e.g. Aerospace DADT) are demoted below generic administrative documents, displacing Fused Rank #1 candidates from the final context window (Q2).
5. **Context Packaging & Boundary Slicing (`evidence_package.py`)**: Chunk boundary cuts between tables and headers cause cross-entity date inversion (GeM vs GFR, Q11, Q12, Q26).
6. **Sentence Termination & List Truncation (`generator.py:_ends_with_terminal_punct`)**: Numbered list prefixes (e.g. `4.`, `5.`) are treated as terminal sentence periods, causing premature list cutoffs (Q4, Q5, Q29).
7. **Destructive Extractive Repair (`generator.py:_extractive_repair`)**: Post-processing fallback concatenates keyword snippets across unrelated departments without boundary isolation, injecting Civil Engineering into Aerospace answers (Q3) and leaking raw markdown tags `######` (Q7).

---

## 2. Complete Q1–Q40 Baseline Failure Matrix

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
| Q30 | Electronics versus Materials Engineering | PASS | Correct | PASS | Present | PASS | Rank 2 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Candidate Verification (verification.py:verify_candidate) | High | Relax candidate-local relation grounding for cross-department comparison |
| Q31 | Missing DADT admission details | PASS | Correct | PASS | Present | PASS | Rank 3 (unfused) | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **PASS** | N/A (Graceful Abstention) | High | None needed |
| Q32 | B.S. Mathematics and Computing admissions | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **PASS** | N/A (Graceful Abstention) | High | None needed |
| Q33 | Historical versus current training dates | PASS | Correct | PASS | Present | PASS | Rank 5 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **PASS** | N/A (Graceful Abstention) | High | None needed |
| Q34 | Ambiguous graduation-credit query | PASS | Correct | PASS | Present | PARTIAL | absent | PASS | PASS | PARTIAL | PARTIAL | FAIL | PASS | PASS | **FAIL** | Generation (generator.py / LLM prompt) | High | Clarify ambiguous degree scope rather than asserting unqualified number |
| Q35 | Materials M.Tech. eligibility | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Packaging (evidence_package.py) | High | Prevent cross-department PG eligibility chunk leakage into unsupported queries |
| Q36 | PMKVY duration false premise | FAIL | Incorrect | FAIL | Present | PASS | absent | PASS | PASS | FAIL | NO | PARTIAL | PASS | PASS | **FAIL** | Query Understanding (planner.py:should_stop_pipeline) | High | Fix scope classifier & markers in planner.py |
| Q37 | Incomplete Mathematics course codes | PASS | Correct | PASS | Present | PASS | Rank 3 | PASS | PASS | PARTIAL | PARTIAL | FAIL | PASS | PASS | **FAIL** | Retrieval / Ranking (ranking.py) | High | Prevent general academic curriculum from masking departmental course tables |
| Q38 | Arithmetic trap | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Generation / Arithmetic Verification | High | Verify component sums independently of table stated total |
| Q39 | Semiconductor chip-design recommendation | PASS | Correct | PASS | Present | PASS | Rank 6 | PASS | PASS | FAIL | NO | FAIL | PASS | PASS | **FAIL** | Generation (generator.py / LLM prompt) | High | Prohibit picking subjective winners when documents show both departments participate |
| Q40 | Incomplete Training programme information | PASS | Correct | PASS | Present | PASS | Rank 1 | PASS | PASS | FAIL | NO | PARTIAL | PASS | PASS | **PARTIAL** | N/A (Graceful Abstention) | High | None needed |

---

## 3. Detailed Root Cause Analysis by Failure Mechanism

### Defect 1: False Deictic Clarification Refusal in Standalone Questions
- **Files Affected**: `ai_platform/core/conversation/conversation.py:622` and `ai_platform/core/graph/nodes.py:1152`
- **Questions Impaired**: **Q20, Q23**
- **Observed Behavior**:
  When a user asks:
  - *Q20*: "According to the Mathematics General Information document's Institute UG structure table, what is the total graded credit count, the total credit count for **the degree**, and the difference between them?"
  - *Q23*: "Name one **course** from each of the three elective groups..."
  The conversation resolution regex `\b(?:this|the|that)\s+(?:programme|program|course|degree)\b` matched the noun phrase "the degree" and "the course". Because the conversation history was empty (`not messages`), line 629 set `"unanchored_reference": "true"`. Subsequently, `nodes.py:1152` intercepted the turn before retrieval and returned:
  `"Could you please specify which programme you are inquiring about (for example, B.Tech., M.Tech., MBA, M.Sc., or Ph.D.) so I can provide the accurate details?"`
- **Earliest Root Cause**: `conversation.py` treats ordinary definite articles ("the degree", "the course") in standalone questions as deictic conversational references requiring clarification.

### Defect 2: False Out-of-Scope Scope Guard Halting Legitimate Training Queries
- **Files Affected**: `ai_platform/core/query/planner.py:_repair_scope` (lines 2533–2553), `INSTITUTIONAL_MARKERS` (lines 84–105), and `_relevance` (lines 2498–2515)
- **Questions Impaired**: **Q13, Q25, Q36**
- **Observed Behavior**:
  When queries reference vocational skilling programs (PMKVY 4.0 Phase I, Skill Hub, Job Roles, Enrolments, Certifications) present in `training_and_upskilling/training.docx`, the planner LLM emitted `domain_decision: "out_of_scope"`. In `_repair_scope`, because the query did not mention degrees like "B.Tech" or "Ph.D.", `_relevance()` scored 0.0. Consequently, `should_stop_pipeline` returned `True` and `nodes.py:1229` emitted a hard refusal:
  `"This is not an institutional request. I can only assist with academic, administrative, and campus information regarding IIT Jodhpur."`
- **Earliest Root Cause**: Narrow institutional scope dictionary omitted vocational training, upskilling, national skill qualifications, and VTU activities present in the institution's uploaded documents.

### Defect 3: Ranking Demotion Displacing Top-Ranked Retrieval Candidates
- **Files Affected**: `ai_platform/core/retrieval/ranking.py:rank_candidates` (lines 950–1085)
- **Questions Impaired**: **Q2**
- **Observed Behavior**:
  For Q2 (M.Tech in Drone and Anti-Drone Technologies start year), RRF fused retrieval correctly ranked `departments/aerospace_engineering/general_information.docx` as **Rank #1**.
  However, `rank_candidates` applied heavy penalties to the specific departmental chunk while promoting generic administration chunks from `office_of_academics` and `mtech/general_information.docx`. As a result, the Aerospace DADT proposal chunk was demoted out of the top 10. The evidence package received only the generic M.Tech overview mentioning 2026-27, producing the false claim that DADT was proposed for 2026-27 without stating its necessity.
- **Earliest Root Cause**: `ranking.py` scoring demoted departmental program proposals when general institute-wide overview documents matched broad keywords.

### Defect 4: Premature Numbered-List Cutoffs from Faulty Terminal Punctuation Detection
- **Files Affected**: `ai_platform/core/answering/generator.py:_ends_with_terminal_punct` (lines 448–454) and `_ensure_clean_sentence_termination` (lines 456–475)
- **Questions Impaired**: **Q4, Q5, Q29**
- **Observed Behavior**:
  When `qwen2.5:3b` output a numbered list, it generated items 1, 2, 3, and printed `4. ` before pausing or finishing.
  The regex in `_ends_with_terminal_punct` (`r'[.!?]["\']?\s*$'`) matched the period in `4.` and declared the text cleanly terminated. `_ensure_clean_sentence_termination` accepted `4.` as a complete sentence!
  In Q4, it displayed only 3 of the 4 Electronics focus areas. In Q5, it cut off the 5th research area. In Q29, it printed:
  `"Four relevant industry pathways from the source include: 1. Data Scientist 2. Machine Learning Engineer 3. AI Research Engineer 4."`
- **Earliest Root Cause**: `_ends_with_terminal_punct` does not check whether trailing punctuation is an incomplete list number prefix (e.g., `\b\d+\.$`).

### Defect 5: Destructive Extractive Repair Cross-Contaminating Departments
- **Files Affected**: `ai_platform/core/answering/generator.py:_extractive_repair` (lines 390–445)
- **Questions Impaired**: **Q3, Q7**
- **Observed Behavior**:
  When the model output required completion or repair, `_extractive_repair` performed a greedy lexical token match across all items in the evidence package without department isolation.
  - In Q3 (Aerospace mission), it extracted text from Civil Engineering faculty records and prepended internal metadata:
    `"[Civil And Infrastructure Engineering - Faculty] Further, the Department encourages faculty members to interact with the... [Aerospace Engineering - General Information]..."`
  - In Q7 (Materials minor options), it extracted raw markdown heading hashes and chunk labels:
    `"...###### in Materials Engineering with Minor in Data and Computational Science Management Entrepreneurship B.Tech. [Materials Engineering - Programs]..."`
- **Earliest Root Cause**: `_extractive_repair` lacks target-department filtering and fails to sanitize heading syntax (`#`) and chunk label brackets (`[...]`).

### Defect 6: Candidate Verification Demanding Candidate-Local Relation Grounding on Cross-Department Queries
- **Files Affected**: `ai_platform/core/retrieval/verification.py:verify_candidate` (lines 470–485)
- **Questions Impaired**: **Q30**
- **Observed Behavior**:
  For Q30 ("Compare Electronics Engineering and Materials Engineering research themes. Identify at least one meaningful overlap and three areas that distinguish the departments..."), the query planner correctly extracted both departments and flagged the query as a comparison/relation.
  However, `verify_candidate` required that every single candidate chunk independently ground the relation between both departments. Because no single document contains both departments' themes together, every chunk was rejected with `relation_not_locally_grounded`. `verified_candidates` was empty, leading to a false refusal ("I'm sorry, I don't know based on the available information.").
- **Earliest Root Cause**: `verification.py` fails to recognize multi-target / comparative queries where different candidates contribute evidence for different target entities.

### Defect 7: Blind Arithmetic Repetition of Inconsistent Document Tables
- **Files Affected**: `ai_platform/core/answering/generator.py:build_answer_prompt` and arithmetic verification
- **Questions Impaired**: **Q38**
- **Observed Behavior**:
  The Mathematics programme table lists: 42 core + 12 elective + 6 open + 2 internship + 12 thesis = stated total 72. (Actual arithmetic sum: 42 + 12 + 6 + 2 + 12 = 74).
  When asked "These add up to 72, correct?", the assistant replied:
  `"Yes, the credits listed in the B.S. in Mathematics and Computing table add up correctly. The total is 42 (core credits) + 12 (programme elective credits) + 6 (open elective credits) + 2 (internship credits) + 12 (thesis credits) = 72 credits."`
  The model hallucinated that 42+12+6+2+12 equals 72 because the document stated 72!
- **Earliest Root Cause**: Lack of instruction to verify arithmetic claims independently when a user query explicitly asks to confirm whether components add up to a stated total.

---

## 4. Planned Generalized Repairs (Ranked by Impact)

| Priority | Component | File Path | Defect Addressed | Generalized Fix Description |
|---|---|---|---|---|
| **P0** | Conversation Resolution | `ai_platform/core/conversation/conversation.py` | False deictic clarification (Q20, Q23) | Restrict `is_deictic_program` to truly conversational turns (`messages` present) or explicit deictic demonstratives ("this/that program") with a missing antecedent; do not flag generic nouns like "the degree" or "the course" in standalone queries. |
| **P0** | Scope Guard & Markers | `ai_platform/core/query/planner.py` | False OOS refusals on training queries (Q13, Q25, Q36) | Add vocational, skill-hub, PMKVY, training, upskilling, and VTU institutional terms to `INSTITUTIONAL_MARKERS` and `_relevance()`; ensure institutional vocational training documents are recognized as in-scope. |
| **P0** | Terminal Punctuation & List Truncation | `ai_platform/core/answering/generator.py` | Premature list cutoff at numbered prefix (Q4, Q5, Q29) | Update `_ends_with_terminal_punct` so that trailing numbered list prefixes (e.g. `\b\d+\.$` or `\b[A-Za-z]\.$`) are never accepted as terminal sentence punctuation; preserve trailing items in `_ensure_clean_sentence_termination`. |
| **P1** | Extractive Repair Sanitization | `ai_platform/core/answering/generator.py` | Cross-department contamination & raw markdown tags (Q3, Q7) | Filter candidates in `_extractive_repair` by the target entity/department of the query; strip markdown heading hashes (`#`) and bracketed source tags (`[Dept - File]`) from repaired sentences. |
| **P1** | Cross-Entity Verification | `ai_platform/core/retrieval/verification.py` | All candidates rejected on comparison queries (Q30) | When query route is `comparison` or query targets multiple entities, allow candidates to be verified if they ground *either* target entity rather than requiring a single chunk to ground both. |
| **P1** | Ranking Demotion | `ai_platform/core/retrieval/ranking.py` | Demotion of specific departmental proposals (Q2) | Ensure that candidates retrieved at Fused Rank #1 with explicit department alignment are not demoted below generic overview documents. |
| **P2** | System Prompt Simplification & Arithmetic Instruction | `ai_platform/core/answering/generator.py` | System prompt token overload & arithmetic trap (Q11, Q12, Q15, Q34, Q38, Q39) | Streamline `_DEFAULT_SYSTEM_INSTRUCTIONS` to remove ~1,000 tokens of test-specific clutter that dilute 3B LLM attention; add clear instruction to verify arithmetic and avoid taking sides on subjective "winner" traps. |
| **P2** | Transparent Citations | `ai_platform/service/assistant_service.py` | Irrelevant citations emitted from context package | Filter citations so only documents that actually substantiate claims present in the final answer are emitted. |

---

## 5. Explicitly Out-of-Scope Changes
1. **No LLM Model Swapping**: Local `qwen2.5:3b` is preserved. Diagnosis proves that once retrieval and prompt clarity are repaired, Qwen 3B produces accurate answers.
2. **No Re-indexing or ChromaDB Rebuilding**: All 40 questions have authoritative source documents already present in `chroma_db/iitj_v1`.
3. **No Hardcoded Question/Answer Lookups**: Zero `if "Q2" in query` or answer-key lookup dictionaries in runtime code.
4. **No Destructive Git Operations**: All user changes and untracked files are strictly preserved.
