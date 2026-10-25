# RAG Quality Regression Tests

This directory contains focused, behaviour-driven regression tests for the `ai_platform/` RAG engine. Tests are organized by **failure class and observable behaviour**, not by which agent authored them.

---

## Architecture Constraints

- All tests import from `ai_platform.*` only — never from `backend.*`.
- Tests under this directory must not overwrite historical diagnostic baselines.
- Diagnostic output and benchmark results are strictly written to `test_outputs/rag_quality/`.
- Never weaken a test or bypass the evidence boundary to obtain an artificial PASS.

---

| File | Failure Class | Key Questions / Contracts Covered | Current Status |
|---|---|---|---|
| `test_answer_correctness.py` | LLM semantic inversion of positive evidence & category threshold misattribution | C2 (`msc10`), C5 (`phd43`), category thresholds (`60% GEN/OBC` vs `55% SC/ST`), planner constraint preservation | **13 PASS (100% — unit + live engine contracts)** |
| `test_multi_intent_evidence.py` | Planner misclassification & evidence routing | C4 (document deadline policy `adm14`), multi-intent propagation | **6 PASS, 1 SKIPPED (raw corpus check)** |
| `test_lsqc_contract.py` | Lossless Semantic Query Contract (LSQC) | Extraction of categories, durations, numbers, qualifications, OR-branching | **10 PASS (100%)** |
| `test_runner_safety.py` | Benchmark runner safety & baseline protection | Safe `--output-dir`, collision refusal, pre-flight validation | **5 PASS (100%)** |
| `run_semantic_eval.py` | 20-question held-out semantic accuracy and reliability runner | Scale transformations, alternative pathways, policy lookup, unanswerable/out-of-scope | **17/20 PASS (85.0% Semantic Correctness, 100% Retrieval Recall, 100% Clean Abstention)** |

---

## Test Execution Commands

All commands are run from repository root using the project virtual environment:

```bash
# 1. Run all fast unit tests (no Ollama required, runs in <1s):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/ -v -m "not integration"

# 2. Run runner safety suite specifically:
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/test_runner_safety.py -v

# 3. Run answer correctness unit tests (C2/C5):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/test_answer_correctness.py -v -m "not integration"

# 4. Run multi-intent / C4 unit tests:
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/test_multi_intent_evidence.py -v -m "not integration"

# 5. Run complete quality suite including live integration contracts (requires Ollama):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_reusable/rag_quality/ tests_reusable/standalone/ -v
```

---

## Pre-fix vs. Contract Tests

- **Faithful Paraphrase Tolerance tests**: Always pass to ensure polarity inversion detection does not become a crude keyword ban.
- **Inversion Detection tests**: Flag answers that deny qualifying routes when evidence contains them.
- **Live Engine Acceptance tests**: Confirm the end-to-end engine generates affirmative, grounded answers for C2, C5, and C4.

---

## Defect Index

| Defect ID | Case ID | Question | Root Cause | Status |
|---|---|---|---|---|
| **C2** | `msc10` | *"Is a first-class qualification accepted for M.Sc. admission?"* | Evidence contains `"OR a first-class"`. Model answers negative; grounding missed inversion. | **VERIFIED FIXED** in `generator.py` & `grounding.py` + `nodes.py` (answers "Yes", grounded). |
| **C5** | `phd06` / Q43 | *"Can a four-year bachelor's degree qualify someone for Ph.D. admission?"* | Evidence contains 4-year degree eligibility. Model denies it; grounding missed inversion. | **VERIFIED FIXED** via grounding polarity detection + extractive repair (answers affirmative). |
| **M.Tech First-Class** | `mtech08` | *"Is a first-class qualification accepted for regular M.Tech admission?"* | Same polarity inversion pattern as C2. | **VERIFIED FIXED** (answers "Yes, a first-class qualification is accepted"). |
| **C4** | `adm14` | *"What happens if an admitted student fails to submit required documents by the specified deadline?"* | Single administrative question misclassified as multi-intent; poor retrieval sub-queries. | **VERIFIED FIXED** in `planner.py` (single-intent policy pattern + deterministic override; answers with cancellation policy). |
| **Scope Halt** | `msc06`, `msc07`, `phd03`, `phd04` | Percentage criteria for M.Sc. / Ph.D. | Scope classifier falsely flagged institutional percentage criteria as out-of-scope. | **VERIFIED FIXED** in `planner.py` (markers expanded; all 4 reach retrieval and return exact percentages). |
