# Frontend & Conversation Quality Test Suite (`tests_frontend/`)

This directory contains automated unit, integration, and conversation-quality tests for the modular Streamlit frontend (`frontend/`) and its presentation-to-service contracts.

---

## 1. Overview & Test Objectives

The tests in this suite ensure that the presentation layer remains completely decoupled from backend retrieval and LLM mechanics while providing robust session isolation, history windowing, and safe handling of assistant responses.

Key guarantees verified:
- **Zero Backend Leakage:** The frontend never imports from the legacy `backend/` directory or introduces un-abstracted external dependencies.
- **Service Decoupling:** Interactions pass strictly through [`frontend.clients.assistant_client.send_chat_turn()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py) to [`ai_platform.service.assistant_service.handle_chat_turn()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/service/assistant_service.py).
- **Session & Tenant Isolation:** Distinct conversation threads maintain isolated histories without cross-talk; tenant configurations validate strictly against schemas.
- **Fast Deterministic Execution:** All tests run in <5 seconds without requiring an active Ollama instance or vector database.

---

## 2. Test Modules

| Test File | Focus Area | Key Contracts & Invariants Covered | Current Status |
|---|---|---|---|
| [`test_frontend_core.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_frontend/test_frontend_core.py) | Frontend Core & Session Mechanics | • Tenant configuration loading & schema validation (`load_institution_ui_config`)<br>• Optional capability fallbacks (e.g., absent map / missing emergency contacts)<br>• Tenant/plugin identity mismatch protection<br>• Assistant client contract adaptation & safe refusal handling<br>• Session manager lifecycle: unique session IDs, switching conversations<br>• History windowing (`get_bounded_history` bounded to `MAX_HISTORY_MESSAGES = 10`)<br>• Architectural hygiene: AST audit confirming zero legacy `backend.*` imports | **17 PASS (100%)** |
| [`test_milestone1_1_conversation_quality.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_frontend/test_milestone1_1_conversation_quality.py) | Conversation Quality & Answering | • Answer completeness & clean sentence termination (`_ensure_clean_sentence_termination`)<br>• Truncation detection & trailing conjunction repair<br>• Coreference resolution & category inheritance across conversational turns<br>• Topic change isolation & user correction handling in multi-turn context<br>• Suggested follow-up questions: rule-based, deterministic, zero extra LLM calls<br>• Suggestion suppression on safe refusals or errors | **22 PASS (100%)** |

---

## 3. Test Execution Commands

Run all tests from the repository root using the project virtual environment:

```bash
# Run all frontend and conversation-quality tests:
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_frontend/ -v

# Run frontend core mechanics tests only:
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_frontend/test_frontend_core.py -v

# Run conversation quality and follow-up tests only:
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_frontend/test_milestone1_1_conversation_quality.py -v

# Run joint frontend + Phase 1 diagnostic contract suite (43 total tests):
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_frontend/ tests/diagnostics/test_phase1_contracts.py -v
```

---

## 4. Test Scope & Distinction from End-to-End Acceptance

- **Automated Contract Tests:** These tests validate presentation logic, state manipulation, and simulated assistant turn interactions using in-memory mocks. They run fast and deterministically.
- **End-to-End Acceptance:** Interactive Streamlit browser verification (e.g., rendering widgets in a live browser, verifying real-time streaming, CSS layout on mobile viewports) is a separate testing phase conducted with `streamlit run app.py` and live Ollama runtime. Passing these unit/contract tests confirms presentation contract integrity, but does not substitute for interactive UI acceptance.
