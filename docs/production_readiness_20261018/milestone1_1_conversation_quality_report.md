# Production Readiness Report — Frontend Milestone 1.1: Answer Completion, Conversation Resolution and Context-Aware Follow-Ups

**Date**: October 18, 2026  
**Author**: Senior Staff AI Engineer specializing in RAG systems, prompt engineering, conversational retrieval, and Streamlit application architecture  
**Target Repository**: IIT Jodhpur & Reusable Academic Core RAG Platform  
**Target Runtime**: Canonical `ai_platform/`, Streamlit 1.66.0, Python 3.14.2  
**Configured Models**: Local Ollama (`qwen2.5:3b`, `nomic-embed-text:latest`)  
**Status**: **QUALIFIED & COMPLETE**

---

## 1. Executive Summary

Frontend Milestone 1.1 diagnoses, hardens, and qualifies answer completion, conversational coreference resolution, and context-aware follow-up suggestions across the canonical RAG platform and the new modular Streamlit frontend (`frontend/`).

A live request previously produced a visibly truncated mid-sentence answer:
> *"The NOC should either state that the employer has no objection to"*

This milestone determined the exact root cause of this failure, implemented defensive generation controls, hardened the conversation coreference resolution pipeline to preserve query attributes (e.g., fees, stipends, leaves) and isolate topic transitions, and added a deterministic, zero-overhead follow-up suggestion capability integrated into the interactive chat UI.

### Headline Results

| Metric / Dimension | Outcome | Details |
| :--- | :---: | :--- |
| **Incomplete Answer Diagnosis** | **RESOLVED** | Identified `ANSWER_LLM_MAX_OUTPUT=256` cutoff with unchecked `done_reason: 'length'`. Increased token limit to 512, added continuation fallback and terminal punctuation enforcement. |
| **Prompt Architecture Audit** | **COMPLETE** | Audited all 7 prompt touchpoints across canonical platform; documented in [prompt_and_conversation_flow_audit.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/docs/architecture/prompt_and_conversation_flow_audit.md). |
| **Conversation Resolution** | **HARDENED** | Preserves antecedent attributes on category follow-ups (e.g. "What about sponsored students?" -> "tuition fee"), resolves explicit corrections, and cleanly isolates topic changes. |
| **Follow-Up Suggestions** | **DELIVERED** | Lightweight deterministic suggestion engine (`generate_follow_up_suggestions`), emitting <= 3 grounded questions. Zero extra LLM invocations (0.00s added latency). Safely suppresses suggestions on refusals/errors. |
| **UI Interactive Integration** | **CONNECTED** | Clicking suggestion triggers full canonical turn through `assistant_client.py`, retaining active session ID and bounded history window. |
| **Frontend Unit Test Suite** | **22 / 22 PASSED** | All 22 regression and unit tests in [test_milestone1_1_conversation_quality.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_frontend/test_milestone1_1_conversation_quality.py) pass 100%. |
| **Existing Frontend Baseline** | **30 / 30 PASSED** | Prior tests (`test_frontend_core.py`, `test_assistant_service.py`, `test_streamlit_canonical_runtime.py`) pass 100% with zero regressions. |
| **Live Ollama Qualification** | **8 / 8 PASSED (100%)** | Evaluated 8 diverse live conversational scenarios against local Ollama (`qwen2.5:3b`) without truncation, context leakage, or ungrounded assertions. |

---

## 2. Phase A — Incomplete Answer Generation: Root Cause & Resolution

### 2.1 Trace of Live Request Failure

We traced a single request from user input to final UI rendering:
1. **Prompt Construction**: `ai_platform/core/answering/generator.py::build_answer_prompt` compiled verified evidence and instructions.
2. **Model Invocation**: `ai_platform/runtime/llm.py::invoke_answer_llm` called local Ollama `http://localhost:11434/api/generate` with `qwen2.5:3b`.
3. **Termination Trigger**: In `ai_platform/runtime/llm.py`, `ANSWER_LLM_MAX_OUTPUT` was hardcoded to `256` tokens.
4. **Provider Behavior**: Ollama generated 256 tokens and halted with `done_reason: "length"`, stopping mid-sentence at:
   > *"The NOC should either state that the employer has no objection to"*
5. **Generator Inaction**: `generator.py` received the text string, ignored the `done_reason` metadata, and returned the truncated string in `AnswerGenerationResult`.
6. **Pass-Through**: `AssistantService` and `assistant_client.py` forwarded the text without length checks.
7. **Session Storage & UI**: Stored into Streamlit `st.session_state.conversations` and rendered verbatim to the user.

### 2.2 Root Cause Categorization

- **Model Output**: Cut off due to reaching the configured token limit (`done_reason: 'length'`).
- **Service Layer**: Ignored completion termination metadata and failed to inspect sentence termination boundaries.
- **Prompt Structure**: Previous system prompt did not instruct the model to close thoughts cleanly before running out of token budget.

### 2.3 The Smallest Correct Fix

1. **Token Budget Adjustment**: Increased `ANSWER_LLM_MAX_OUTPUT` in `ai_platform/runtime/llm.py` from 256 to 512 tokens. This provides adequate headroom for multi-part institutional policy answers without incurring unbounded latency.
2. **Termination Metadata Propagation**: Extended `AnswerGenerationResult` to carry `finish_reason: Optional[str] = None`.
3. **Continuation & Sentence Boundary Guard**:
   - Implemented `_ends_with_terminal_punct(text)` to verify proper punctuation (`.`, `?`, `!`, `"`).
   - If `finish_reason == "length"` and text ends mid-sentence, the generator attempts a single bounded continuation call (`invoke_answer_llm` with prompt requesting completion of the unfinished thought).
   - If continuation fails or is unavailable, `_ensure_clean_sentence_termination(text)` trims back to the last complete sentence boundary rather than displaying dangling fragments.
4. **Prompt Termination Directives**: Added explicit instructions in `_DEFAULT_SYSTEM_INSTRUCTIONS` requiring the model to finish all sentences cleanly and avoid initiating trailing clauses when near the token limit.

---

## 3. Phase B — Prompt Audit and Hardening

A comprehensive prompt audit was conducted and documented in [prompt_and_conversation_flow_audit.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/docs/architecture/prompt_and_conversation_flow_audit.md).

### 3.1 Answer Generation Prompt

- **File**: `ai_platform/core/answering/generator.py`
- **Improvements**:
  - Structured formatting: Mandates Markdown headings, bullet points, or numbered steps for multi-part questions.
  - Sentence termination: Enforces clean grammatical conclusion for every sentence.
  - Partial evidence discipline: Requires stating known facts cleanly and explicitly noting missing or unspecified institutional conditions rather than fabricating details.
  - Grounded citations: Strictly preserves existing bracketed citations `[DocID, Section]`.

### 3.2 Conversation Resolution & Coreference Rewriting

- **File**: `ai_platform/core/conversation/conversation.py`
- **Failure Modes Addressed**:
  1. *Category Follow-Ups*: When a user asks *"What about sponsored students?"* following *"What is the tuition fee for regular M.Tech. students?"*, earlier logic preserved `M.Tech.` but dropped the inquiry attribute (`tuition fee`), defaulting to general admission. Added `_extract_query_attribute` to capture attributes (`tuition fee`, `eligibility criteria`, `leave rules`, `stipend`) and synthesize the complete query: *"What is the tuition fee for sponsored students in M.Tech?"*.
  2. *Explicit Corrections*: Handled corrections such as *"No, I meant MBA, not M.Tech"* to cleanly update the active target entity.
  3. *Topic Change Isolation*: Prevented prior academic entities from persisting into new standalone topics (e.g., campus navigation, emergency queries).

---

## 4. Phase C & D — Context-Aware Suggested Follow-Up Questions

### 4.1 Design & Implementation Strategy

To keep the application highly responsive, suggestions are generated **deterministically without an extra LLM call**:
- **File**: `ai_platform/service/suggestions.py`
- **Function**: `generate_follow_up_suggestions(question, answer, evidence, active_entity, active_topic, institution_id)`
- **Behavior**:
  - Inspects the active entity (`M.Tech.`, `Ph.D.`, `MBA`), topic (`fee`, `leave`, `admission`), and retrieved section titles.
  - Emits at most 3 concise, highly relevant follow-up questions tailored to the active context.
  - Filters out suggestions matching the current question or prior history.
  - **Safe Suppression**: Returns `[]` on safe refusals, engine errors, empty evidence, or navigation queries.
  - **Latency Impact**: 0.00 seconds added (in-memory candidate generation).

### 4.2 Interactive UI Connection

In `frontend/components/messages.py` and `frontend/components/chat.py`:
- Suggestions are displayed as interactive buttons below the latest assistant turn.
- Clicking a suggestion sets session state and invokes `_process_user_turn(clicked_question, config)`, ensuring:
  1. Turn is submitted to the canonical `handle_chat_turn()` service.
  2. The active conversation ID and bounded history window are preserved.
  3. Evidence is retrieved and verified for the new query.
  4. Deduplication safeguards prevent multi-click races.
  5. Cross-session and cross-tenant boundaries remain completely isolated.

---

## 5. Phase E — Service Contract & Architectural Hygiene

- **Backward Compatibility**: `ChatTurnResponse` in `ai_platform/service/assistant_service.py` was extended with an optional field:
  ```python
  suggested_questions: list[str] = field(default_factory=list)
  ```
  Existing callers, benchmarks, and legacy tests continue to work without modification.
- **Frontend Boundary**: `frontend/clients/assistant_client.py` captures `suggested_questions` and packs it into `AssistantResponse`, which `frontend/core/models.py::ChatMessage` stores in session state.
- **Invariants Preserved**:
  - No legacy backend imports.
  - Zero database or Supabase dependencies.
  - No modification to `UI/` legacy directory.
  - No deletion or moving of files.

---

## 6. Phase F — Regression Test Verification

### 6.1 Milestone 1.1 Test Suite (`tests_frontend/test_milestone1_1_conversation_quality.py`)

A comprehensive suite of 22 focused tests was authored covering answer completion, conversation resolution, and follow-up suggestions:

```
tests_frontend/test_milestone1_1_conversation_quality.py ...................... [100%]
22 passed in 1.28s
```

Test coverage includes:
1. `test_terminal_punctuation_detection`: Verifies correct recognition of sentence endings.
2. `test_sentence_termination_cleanup`: Verifies trimming of mid-sentence dangling fragments.
3. `test_generator_bounded_continuation_on_length_cutoff`: Verifies single-pass continuation invocation when `done_reason == "length"`.
4. `test_generator_output_limit_configured_to_512`: Validates active token budget configuration.
5. `test_generator_short_complete_answer_not_continued`: Ensures complete short answers trigger no continuation.
6. `test_generator_prompt_structure_and_readability_guidelines`: Validates presence of Markdown headings and sentence termination rules in system prompt.
7. `test_conversation_category_followup_preserves_fee_attribute`: Confirms "What about sponsored students?" inherits "tuition fee" and "M.Tech".
8. `test_conversation_program_followup_preserves_leave_attribute`: Confirms pronoun "this programme" resolves to "Ph.D." and "leave rules".
9. `test_conversation_topic_change_clears_old_entity`: Confirms topic shifts do not inherit prior academic entities.
10. `test_conversation_explicit_correction`: Confirms "No, I meant MBA" correctly substitutes entity.
11. `test_conversation_bounded_history_window`: Confirms long histories are trimmed to bounded window.
12. `test_conversation_cross_session_isolation`: Confirms separate sessions maintain complete entity isolation.
13. `test_conversation_institution_isolation`: Confirms cross-institution queries do not leak context.
14. `test_suggestions_count_at_most_three`: Confirms suggestion count never exceeds 3.
15. `test_suggestions_relevance_to_entity_and_topic`: Confirms suggestions match active entity and topic.
16. `test_suggestions_suppressed_on_safe_refusal`: Confirms zero suggestions emitted on safety refusal.
17. `test_suggestions_suppressed_on_empty_evidence`: Confirms zero suggestions emitted on ungrounded/empty evidence.
18. `test_suggestions_distinct_from_query`: Confirms suggestions do not repeat the current question.
19. `test_suggestions_deterministic_zero_llm_calls`: Confirms zero LLM invocations occur during suggestion generation.
20. `test_assistant_service_contract_suggested_questions`: Validates backward-compatible service response contract.
21. `test_frontend_client_preserves_suggestions`: Validates frontend client adaptation of suggestion metadata.
22. `test_session_manager_stores_and_isolates_suggestions`: Confirms independent session state isolation for suggestions.

### 6.2 Pre-Existing Test Suites

All existing unit and integration test suites pass with zero regressions:
- `tests_frontend/test_frontend_core.py`: **17 / 17 PASSED**
- `tests_reusable/test_assistant_service.py`: **9 / 9 PASSED**
- `tests_reusable/test_streamlit_canonical_runtime.py`: **4 / 4 PASSED**
- **Total Unit & Integration Pass Rate**: **52 / 52 PASSED (100%)**

---

## 7. Phase G — Real-Model Qualification Results

Evaluation was executed against local Ollama (`qwen2.5:3b`) using the automated qualification runner:
`scripts/run_frontend_milestone1_1_qualification.py`.

### 7.1 Evaluation Matrix

| Case | Scenario | Resolved Query | Result | Verification & Punctuation |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Long multi-part question risking truncation | Standalone multi-clause query | **PASS** | Complete answer ending on clean sentence boundary without mid-sentence truncation. |
| **2** | Ordinary short factual question | `What is the Ph.D. application processing fee for General category applicants?` | **PASS** | Returns exact factual answer (Rs. 300) with proper terminal punctuation. |
| **3** | Base question + context follow-up | `What is the tuition fee for sponsored students in M.Tech?` | **PASS** | Resolved query attribute (`tuition fee`) and entity (`M.Tech.`) into explicit search query. |
| **4** | Second follow-up referring to earlier programme | `How many days of leave can scholars in PhD programme take?` | **PASS** | Correctly resolved pronoun to `Ph.D.` and preserved leave topic. |
| **5** | Follow-up after multi-intent answer | `Are there fee waivers for SC or ST students in M.Tech?` | **PASS** | Resolved "that program" to `M.Tech.` and answered safely. |
| **6** | Genuine topic change in same session | `Where is the Lecture Hall Complex located on campus?` | **PASS** | Clean topic change; standalone structured navigation returned with 0 suggestions. |
| **7** | Same vague follow-up in new session | `What about sponsored students?` | **PASS** | Clean session boundary; no antecedent leakage from previous chat. |
| **8** | Unsupported question safe abstention | `What is the ranking of IIT Jodhpur in the 2050 QS World University Rankings?` | **PASS** | Safely abstained without fabricating facts and emitted 0 follow-up suggestions. |

**Overall Real-Model Score**: **8 / 8 PASSED (100%)**  
**Machine-Readable Artifacts**:
- `test_outputs/frontend_milestone1_1/qualification_results.json`
- `test_outputs/frontend_milestone1_1/qualification_summary.md`

---

## 8. Remaining Risks and Future Work

1. **Local Model Generation Speed**: On CPU-bound environments, running full multi-turn generation with `qwen2.5:3b` averages ~15-25s per turn. GPU acceleration or quantization tuning would improve interactive responsiveness.
2. **Dynamic Suggestion Personalization**: While deterministic suggestions have zero latency overhead and guaranteed grounding, an optional offline model-based suggestion ranker could be evaluated in Milestone 2 for deeper conversational branch prediction.
