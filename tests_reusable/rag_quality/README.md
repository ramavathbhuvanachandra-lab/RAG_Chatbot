# RAG Quality & Evidence Hardening Regression Tests (`tests_reusable/rag_quality/`)

This directory contains focused, behavior-driven regression and contract test suites for the canonical `ai_platform/` RAG engine. Tests are organized by **failure class, security invariants, and observable behavior**.

---

## 1. Architectural Constraints

- **Strict Imports:** All tests import from `ai_platform.*` only — never from `backend.*`.
- **Preserve Baselines:** Tests and runner scripts must never overwrite historical diagnostic baselines or benchmark logs.
- **Isolated Artifacts:** All diagnostic output and benchmark results are strictly written to `test_outputs/rag_quality/` or `test_outputs/recovery_baseline_eval/`.
- **Zero Evidence Bypasses:** Tests verify end-to-end evidence packaging through the typed graph pipeline; tests never bypass the evidence boundary to obtain an artificial PASS.

---

## 2. Test Suites & Contract Matrix

| Suite / Runner | Failure Class & Target Invariants | Key Scenarios Covered | Status |
|---|---|---|---|
| [`test_guard_and_evidence_hardening_regression.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/test_guard_and_evidence_hardening_regression.py) | Guard security hardening, prompt injection defense, multi-intent retention, APM procedural deadlines, synthetic institution contract | • Physical security & burglary attacks blocked (`picking locks`)<br>• PII/bank details blocked (`director phone / bank`)<br>• System credentials & router passwords blocked<br>• Examination paper leaks blocked<br>• Prompt injection / translation overrides blocked<br>• Academic fraud & SQL injection blocked<br>• Legitimate inquiries in scope (conduct rules, mess rebate, academic progression, transcripts, degree verification)<br>• Multi-intent retention (branch change & attendance, M.Tech eligibility & fellowship, room rent & qualifying degrees)<br>• Ph.D. APM procedural deadline ranking survival<br>• Synthetic institution profile validation & fallback | **21 PASS (100%)** |
| [`scripts/reproduce_recovery_scenarios.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/scripts/reproduce_recovery_scenarios.py) | Six canonical institutional recovery scenarios (end-to-end execution & trace logging) | • **Scenario A:** ITEP fee table for AY 2026–27 (exact token normalization)<br>• **Scenario B:** Short-term accommodation rent with 1 bedding set<br>• **Scenario C:** Tuition fee exemptions, hostel charges, mess advance<br>• **Scenario D:** Longer accommodation booking (16–30 days) and facilities<br>• **Scenario E:** Ph.D. admission cycle timeline & APM procedural schedule<br>• **Scenario F:** Security and Transport departmental responsibilities | **6/6 PASS (100%)** |
| [`test_answer_correctness.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/test_answer_correctness.py) | LLM semantic inversion of positive evidence & category threshold misattribution | • C2 (`msc10`): First-class qualification accepted for M.Sc.<br>• C5 (`phd43`): 4-year degree eligibility for Ph.D.<br>• Category thresholds: `60% GEN/OBC` vs `55% SC/ST`<br>• Planner constraint preservation across generation | **13 PASS (100%)** |
| [`test_multi_intent_evidence.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/test_multi_intent_evidence.py) | Planner misclassification & evidence routing | • C4: Document deadline cancellation policy (`adm14`)<br>• Multi-intent propagation & query decomposition | **6 PASS, 1 SKIPPED** |
| [`test_lsqc_contract.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/test_lsqc_contract.py) | Lossless Semantic Query Contract (LSQC) | • Extraction of categories, durations, numbers, qualifications, OR-branching | **10 PASS (100%)** |
| [`test_runner_safety.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/test_runner_safety.py) | Benchmark runner safety & baseline protection | • Safe `--output-dir`, collision refusal, pre-flight validation | **5 PASS (100%)** |
| [`run_semantic_eval.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_reusable/rag_quality/run_semantic_eval.py) | 20-question held-out semantic accuracy and reliability runner | • Scale transformations, alternative pathways, policy lookup, unanswerable/out-of-scope | **17/20 PASS (85% Semantic Correctness, 100% Retrieval Recall, 100% Clean Abstention)** |

---

## 3. Test Execution Commands

All commands are executed from the repository root using the project virtual environment:

```bash
# 1. Run guard and evidence hardening regression suite (21 tests):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/test_guard_and_evidence_hardening_regression.py -v

# 2. Run the 6 canonical recovery scenarios runner (logs traces to test_outputs/recovery_baseline_eval/):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/python scripts/reproduce_recovery_scenarios.py

# 3. Run all fast unit tests (no Ollama required, runs in <2s):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/ -v -m "not integration"

# 4. Run answer correctness unit tests (C2/C5):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/test_answer_correctness.py -v -m "not integration"

# 5. Run multi-intent / C4 unit tests:
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/test_multi_intent_evidence.py -v -m "not integration"

# 6. Run complete quality suite including live integration contracts (requires local Ollama):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/ tests_reusable/standalone/ -v
```

---

## 4. The Six Canonical Recovery Scenarios

The script [`scripts/reproduce_recovery_scenarios.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/scripts/reproduce_recovery_scenarios.py) verifies the end-to-end recovery of six critical institutional test cases. Traces and evaluation outputs are recorded in `test_outputs/recovery_baseline_eval/recovery_baseline_traces.json`:

1. **Scenario A (ITEP Fee Structure for AY 2026–27):**
   - *Query:* "What is the fee structure for the 4-Year Integrated Teacher Education Programme (ITEP) for the Academic Year 2026-27?"
   - *Key Invariant:* Normalized academic year token signature (`2026-27` matches `2026-2027` in grounding review), query-aware extraction ensures exact table reproduction without review reject.
2. **Scenario B (Short-Term Accommodation Rent with 1 Bedding Set):**
   - *Query:* "What is the room rent per day for short-term accommodation for students with 1 bedding set?"
   - *Key Invariant:* Multi-intent requested-attribute parser recognizes "rent" and "rates", retrieving INR 150/day and bedding rules accurately.
3. **Scenario C (Tuition Exemptions, Hostel Fees & Mess Advance):**
   - *Query:* "Explain the tuition fee exemptions for SC/ST students, along with hostel seat rent and mess advance."
   - *Key Invariant:* Multi-attribute coverage across tuition waiver categories, hostel fees, and dining advances without cross-intent confusion.
4. **Scenario D (Longer Accommodation Booking & Facilities):**
   - *Query:* "What are the rules and daily rates for longer accommodation bookings (16 to 30 days) and what facilities are included?"
   - *Key Invariant:* Preserves tier-based stay rules (16–30 days) and verified amenities.
5. **Scenario E (Postgraduate Admission-Cycle Timing & APM Schedule):**
   - *Query:* "When are the Annual Progress Monitoring (APM) meetings conducted for Ph.D. students joining in the first semester?"
   - *Key Invariant:* Procedural deadline evidence (`1st July to 30th September`) survives fusion, verification, and ranking.
6. **Scenario F (Institutional Security and Transport Responsibilities):**
   - *Query:* "What are the administrative responsibilities of the Security and Transport sections at IIT Jodhpur?"
   - *Key Invariant:* Dual-topic retrieval and balanced context packaging across administrative sections.

---

## 5. Defect Index & Historical Resolution

| Defect ID | Case ID | Question / Scenario | Root Cause | Resolution & Verification |
|---|---|---|---|---|
| **C2** | `msc10` | *"Is a first-class qualification accepted for M.Sc. admission?"* | Model generated negative response despite `"OR a first-class"` in evidence; grounding missed polarity inversion. | **VERIFIED FIXED** in `generator.py`, `grounding.py`, and `nodes.py` (answers "Yes", fully grounded). |
| **C5** | `phd06` / Q43 | *"Can a four-year bachelor's degree qualify someone for Ph.D. admission?"* | Model denied 4-year degree eligibility; grounding failed to catch negation. | **VERIFIED FIXED** via polarity detection + extractive repair. |
| **M.Tech First-Class** | `mtech08` | *"Is a first-class qualification accepted for regular M.Tech admission?"* | Same polarity inversion pattern as C2. | **VERIFIED FIXED** (answers "Yes, a first-class qualification is accepted"). |
| **C4** | `adm14` | *"What happens if an admitted student fails to submit required documents by the specified deadline?"* | Single administrative question misclassified as multi-intent; sub-queries diluted context. | **VERIFIED FIXED** in `planner.py` (single-intent policy pattern override; returns cancellation policy). |
| **Scope Halt** | `msc06`, `phd03` | Percentage criteria for M.Sc. / Ph.D. | Scope classifier falsely flagged institutional percentage criteria as out-of-scope. | **VERIFIED FIXED** in `planner.py` (scope markers expanded; reaches retrieval cleanly). |
| **AY Signature** | `gen_dev_001` | Academic year `2026-27` in grounding | Strict token mismatch between `2026-27` and `2026-2027` triggered false review status. | **VERIFIED FIXED** in `grounding.py` via `_academic_year_signature` normalization. |
| **Commercial Guard** | Commercial ticket queries | False positive refusal on legitimate queries with articles | Overly rigid regex failed on phrasing variations. | **VERIFIED FIXED** in `guard.py` (allows conversational articles/pronouns). |
| **Rent Attribute** | `gen_anchor_14` | Accommodation rent & rates parsing | `rent` and `rates?` missing from fee requested-attributes regex. | **VERIFIED FIXED** in `planner.py` (regex includes `rent` and `rates?`). |
