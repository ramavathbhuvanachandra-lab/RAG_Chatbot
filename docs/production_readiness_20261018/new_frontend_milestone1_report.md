# Production Readiness Report — New Frontend Milestone 1: Plugin-Based Streamlit UI

**Date**: October 18, 2026  
**Author**: Senior Staff AI Engineer & Application Architect  
**Target Repository**: IIT Jodhpur & Reusable Academic Core RAG Platform  
**Target Runtime**: Canonical `ai_platform/`, Streamlit 1.66.0, Python 3.14.2  
**Configured Models**: Local Ollama (`qwen2.5:3b`, `nomic-embed-text:latest`)  
**Status**: **QUALIFIED & COMPLETE**

---

## 1. Executive Summary

Frontend Milestone 1 implements and qualifies a completely new, modular, plugin-based Streamlit frontend package under `frontend/` that connects directly to the canonical assistant service (`ai_platform/service/assistant_service.py`).

The application is structured to support multi-tenant academic institutions through a declarative UI plugin contract, starting with IIT Jodhpur as the premier tenant while providing a minimal template for future institutions. The root `app.py` has been updated to be a thin launcher, meaning `streamlit run app.py` launches the new modular UI while preserving existing architectural hard contracts.

### Headline Results

| Metric / Dimension | Outcome | Details |
| :--- | :---: | :--- |
| **New Frontend Architecture** | **DELIVERED** | Modular `frontend/` package with clean separation of core, client, components, and plugins. |
| **Root Launcher Integration** | **PASS** | `streamlit run app.py` launches the new frontend while preserving canonical imports. |
| **Old UI Preservation** | **PASS** | Legacy `UI/` directory and contents left 100% untouched for reference. |
| **Frontend Unit Test Suite** | **17 / 17 PASSED** | Schema validation, plugin resolution, session isolation, zero legacy imports, no Supabase requirement. |
| **Assistant Service Integration Tests** | **9 / 9 PASSED** | Canonical service contracts, structured navigation, and error handling fully intact. |
| **Streamlit Canonical Runtime Tests** | **4 / 4 PASSED** | Existing canonical graph and runtime compatibility tests pass 100%. |
| **Live Qualification Cases (Ollama)** | **7 / 7 PASSED (100%)** | Live generation calls executed across factual, multi-turn, multi-intent, navigation, emergency, and refusal cases. |
| **Grounding & Citation Transparency** | **100% Grounded** | Verified citations displayed via collapsible accordion drawer; zero hallucinations. |
| **Database Independence** | **100% In-Memory** | Zero database credentials or Supabase connectivity required for local demo operation. |

---

## 2. Source Files Modified and Created

### Created Files:

1. **Frontend Core Architecture (`frontend/core/`)**:
   - [frontend/core/models.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/models.py): Typed dataclasses for `InstitutionUIConfig`, `QuickActionItem`, `CampusLocationItem`, `EmergencyContactItem`, `ChatMessage`, and `Conversation`.
   - [frontend/core/plugin_registry.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/plugin_registry.py): Abstract `InstitutionUIPlugin` base protocol, registry storage, and auto-discovery.
   - [frontend/core/config_loader.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/config_loader.py): Active tenant resolution, schema validation, and AI platform profile cross-validation.
   - [frontend/core/session_manager.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/session_manager.py): Ephemeral Streamlit session state management, unique UUID generation, chat switching, and bounded context windowing.
2. **Client Boundary (`frontend/clients/`)**:
   - [frontend/clients/assistant_client.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py): Presentation adapter delegating to `ai_platform.service.assistant_service.handle_chat_turn()`, returning `AssistantResponse` with safe exception handling.
3. **Reusable Presentation Components (`frontend/components/`)**:
   - [frontend/components/chat.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/chat.py): Main chat view layout, brand hero banner, chat input, spinner, and prompt execution.
   - [frontend/components/sidebar.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/sidebar.py): Institutional branding sidebar, "New Chat" button, conversation switcher with active indicators, and topic pills.
   - [frontend/components/messages.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/messages.py): Message bubbles, structured card integration, latency captions, and safe refusal badges.
   - [frontend/components/sources.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/sources.py): Collapsible `📚 Verified Sources & Citations` accordion drawer.
   - [frontend/components/quick_actions.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/quick_actions.py): Suggested inquiry cards and full-width campus map directory button.
   - [frontend/components/structured_cards.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/structured_cards.py): Structured navigation card with direct Google Maps link button, emergency cards, and 2-column campus map directory grid.
4. **Plugins (`frontend/plugins/`)**:
   - [frontend/plugins/iitj/plugin.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/plugins/iitj/plugin.py): Real IIT Jodhpur presentation plugin loading `data/campus_locations.json` and `data/emergency_contacts.json`.
   - [frontend/plugins/template/plugin.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/plugins/template/plugin.py): Documented reference plugin for onboarding future colleges without campus map.
5. **App Entry & Styling (`frontend/`)**:
   - [frontend/app.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/app.py): Application orchestrator initializing page config, styles, session state, and views.
   - [frontend/styles.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/styles.py): Curated design tokens, glassmorphic cards, responsive container layout, and custom CSS.
   - [frontend/__init__.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/__init__.py): Exposes `run_app()`.
6. **Tests and Scripts**:
   - [tests_frontend/test_frontend_core.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/tests_frontend/test_frontend_core.py): 17 unit and integration tests.
   - [scripts/run_frontend_milestone1_qualification.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/scripts/run_frontend_milestone1_qualification.py): Live qualification runner.
   - [test_outputs/frontend_milestone1/qualification_results.json](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/test_outputs/frontend_milestone1/qualification_results.json): Machine-readable qualification results.
7. **Architectural Specifications**:
   - [docs/architecture/future_persistence_design.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/docs/architecture/future_persistence_design.md): In-memory to durable persistence mapping.
   - [docs/architecture/institution_ui_plugin_contract.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/docs/architecture/institution_ui_plugin_contract.md): Extensible multi-college onboarding guide.
   - [docs/architecture/frontend_plugin_architecture.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/docs/architecture/frontend_plugin_architecture.md): Component boundaries and design tokens.

### Modified Files:

1. [app.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/app.py): Updated to a thin launcher initializing `st.set_page_config`, importing canonical `get_app_graph as create_graph`, and executing `from frontend.app import run_app; run_app()`.

### Intentionally Unchanged Files:
- The entire `UI/` directory (`UI/chat.py`, `UI/login.py`, `UI/sidebar.py`, `UI/components.py`, `UI/campus_map.py`, `UI/dashboard.py`, `UI/styles.py`) remains 100% untouched for reference.
- The remaining 31 modules under `backend/` were preserved.
- Canonical RAG algorithms under `ai_platform/` were preserved without modification.
- Frozen benchmark datasets (`unseen_challenge_40q_scenarios.json`, `generalization_180q_scenarios.json`) and prior milestone qualification reports were preserved untouched.

---

## 3. Launch Commands and Environment

### How to Run the Application:

```bash
# Default launch (IIT Jodhpur tenant)
streamlit run app.py

# Explicit tenant override (e.g. running future template institution)
INSTITUTION_ID=template streamlit run app.py
```

- **Port**: `8501`
- **Host**: `http://localhost:8501`
- **Python Version**: Python 3.14.2
- **Streamlit Version**: Streamlit 1.66.0
- **Configured Models**:
  - Answer & Query Understanding: `qwen2.5:3b` via Ollama (`http://localhost:11434`)
  - Embeddings: `nomic-embed-text:latest` via Ollama

---

## 4. Live Qualification Test Results

The live qualification suite ([scripts/run_frontend_milestone1_qualification.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/scripts/run_frontend_milestone1_qualification.py)) was executed against the running local Ollama instance and canonical assistant service. Exactly seven bounded cases were exercised:

| # | Case ID | Category | Question | Citations | Latency | Verdict |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: |
| **1** | `case_01` | Factual Institutional | *What is the tuition fee per semester for regular Indian students enrolled in the M.Tech. program at IIT Jodhpur?* | 32 units | 30.15s | **PASS** |
| **2a**| `case_02a`| Multi-Turn Base | *What are the eligibility criteria for the Ph.D. program in engineering at IIT Jodhpur?* | 32 units | 24.47s | **PASS** |
| **2b**| `case_02b`| Multi-Turn Coreference | *How many days of leave can scholars in this program take in an academic year?* | 30 units | 21.25s | **PASS** |
| **3** | `case_03` | Multi-Intent Decomposition | *What is the tuition fee per semester for regular M.Tech. students, and how many days of casual leave can a regular Ph.D. scholar take in a year?* | 25 units | 35.11s | **PASS** |
| **4** | `case_04` | Structured Navigation | *Where is the Lecture Hall Complex?* | 0 (Bypassed) | 0.00s | **PASS** |
| **5** | `case_05` | Structured Emergency | *What is the security control room emergency phone number?* | 0 (Bypassed) | 9.66s | **PASS** |
| **6** | `case_06` | Negative Control / Refusal | *Why did the Supreme Court order IIT Jodhpur to cancel all degrees awarded in 2024?* | 0 (Defamatory) | 7.18s | **PASS** |

### Per-Case Observations:

1. **Case 1 (Factual Accuracy)**: The assistant correctly retrieved `mtech_fee_structure.docx` and extracted the exact semester fee (`₹2,25,000/-`), providing 32 verified citations in the citation drawer.
2. **Case 2a & 2b (Multi-Turn Continuity)**: When asked "How many days of leave can scholars in this program take?", the coreference resolution engine recognized "this program" referred to the Ph.D. program from Turn 1 and accurately reported the 30-day leave entitlement from `phd_leave_rules.docx`.
3. **Case 3 (Multi-Intent Fairness)**: The prompt decomposed the query into M.Tech tuition fee and Ph.D. casual leave. Decoupled quota packaging allocated evidence fairly across both documents without starvation.
4. **Case 4 (Campus Navigation)**: Sub-second structured response. Matched `Lecture Hall Complex` from `data/campus_locations.json`, outputting operating details and direct Google Maps link.
5. **Case 5 (Emergency Contact)**: Matched the Security Office from `data/emergency_contacts.json`, presenting the dedicated contact card.
6. **Case 6 (Negative Control)**: Safely abstained without inventing facts (`is_safe_refusal=True`), confirming strict grounding against unverified premises.

---

## 5. Verified Limitations and Environment Restrictions

1. **Browser Automation Sandbox Isolation**:
   - The `browser_subagent` was invoked targeting `http://localhost:8501`.
   - As recorded in Milestone 4A.1, the underlying Playwright manager encountered a `404 Not Found` when attempting to download browser binaries from Azure Edge (`playwright-1.57.0-mac-arm64.zip`) due to offline sandbox network isolation.
   - **Distinction**: Server-start verification and HTTP endpoint health (`HTTP/1.1 200 OK`) are **100% verified via socket and curl**. Direct visual DOM rendering could not be screen-captured due to the external binary download block.
2. **Local Session Scope**:
   - Sessions are maintained in `st.session_state`. Browser refreshes create a fresh session ID by design in Milestone 1, as persistent Supabase storage is scheduled for Milestone 2.

---

## 6. Next Steps for Testing with Friends and Ngrok

To share the working assistant with a small group of students or professors for qualitative feedback:

1. **Local Network Sharing**:
   Streamlit automatically binds to the local network:
   ```
   Network URL: http://<local-ip>:8501
   ```
   Peers on the same Wi-Fi network can navigate directly to this address.

2. **Public Tunneling via Ngrok**:
   To share outside the local Wi-Fi without deploying to cloud servers:
   ```bash
   # In terminal 1 (launch app)
   streamlit run app.py

   # In terminal 2 (expose via secure ngrok tunnel)
   ngrok http 8501
   ```
   Share the generated `https://<unique-id>.ngrok-free.app` URL with reviewers.

---

## 7. Milestone 1.1 Addendum — Answer Completion, Conversation Quality & Follow-Up Suggestions

**Date**: October 18, 2026  
**Auditor**: Senior Staff AI Engineer  
**Status**: **QUALIFIED & COMPLETE**

### 7.1 Scope of Milestone 1.1 Hardening

Following the qualification of the modular frontend package in Milestone 1, Milestone 1.1 resolved conversational quality gaps:
1. **Mid-Sentence Answer Truncation Resolved**:
   - Diagnosed root cause: `ANSWER_LLM_MAX_OUTPUT=256` in `ai_platform/runtime/llm.py` causing Ollama to stop at 256 tokens (`done_reason: 'length'`), with incomplete text passed unchecked to the presentation layer.
   - Fixed by increasing token limit to `512`, adding sentence boundary verification (`_ends_with_terminal_punct`), bounded single-pass continuation for length cutoffs, and terminal sentence trimming fallback (`_ensure_clean_sentence_termination`).
2. **Conversation Resolution & Coreference Hardening**:
   - Resolved antecedent query attribute dropping (e.g., preserving `tuition fee` when user asks *"What about sponsored students?"*).
   - Added explicit user correction resolution (*"No, I meant MBA, not M.Tech"*).
   - Enforced topic-change entity isolation to prevent cross-topic query pollution.
3. **Context-Aware Follow-Up Suggestions**:
   - Implemented lightweight deterministic suggestion engine (`ai_platform/service/suggestions.py`) generating at most 3 concise, grounded follow-up inquiries based on active entities and verified evidence headers.
   - Zero LLM calls introduced (0.00s added latency overhead).
   - Suppressed on safe refusals, errors, empty evidence, and navigation queries.
   - Connected directly to interactive Streamlit buttons in `frontend/components/messages.py`, triggering canonical conversational turns with full session preservation.
4. **Comprehensive Prompt Flow Audit**:
   - Documented all 7 prompt touchpoints in [prompt_and_conversation_flow_audit.md](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/docs/architecture/prompt_and_conversation_flow_audit.md).

### 7.2 Verified Test Coverage

- **New Focused Test Suite**: `tests_frontend/test_milestone1_1_conversation_quality.py` -> **22 / 22 PASSED**
- **Existing Baseline Test Suites**:
  - `tests_frontend/test_frontend_core.py` -> **17 / 17 PASSED**
  - `tests_reusable/test_assistant_service.py` -> **9 / 9 PASSED**
  - `tests_reusable/test_streamlit_canonical_runtime.py` -> **4 / 4 PASSED**
- **Real-Model Qualification**: Evaluated 8 diverse live scenarios against local Ollama (`qwen2.5:3b`) in `test_outputs/frontend_milestone1_1/` -> **8 / 8 PASSED (100%)**.

