# Modular Streamlit Frontend (`frontend/`)

The `frontend/` package provides the presentation layer for the IIT Jodhpur Institutional RAG Chatbot. It connects users to the canonical AI platform via an institutional UI built on Streamlit.

---

## 1. Responsibilities & Boundaries

- **Presentation Only:** The frontend renders messages, sidebar navigation, campus map locations, quick inquiries, and citations. It contains **no** retrieval, prompt engineering, embedding, vector database, or LLM generation logic.
- **Service Decoupling:** User requests are passed to the canonical AI platform via [`frontend.clients.assistant_client.send_chat_turn()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py), which delegates to [`ai_platform.service.assistant_service.handle_chat_turn()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/service/assistant_service.py).
- **Multi-Tenant Theming:** UI styling, logos, titles, and quick prompts are loaded dynamically per institution using [`frontend.core.config_loader`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/config_loader.py).

---

## 2. Package Architecture

```text
frontend/
├── __init__.py
├── app.py                      # Main frontend application entry point (run_app)
├── styles.py                   # Custom CSS styling tokens & theme injection
├── clients/
│   └── assistant_client.py     # Presentation-to-service client adapter
├── components/
│   ├── chat.py                 # Chat container, input bar, and turn processor
│   ├── messages.py             # Message history rendering and citation display
│   ├── quick_actions.py        # One-click question suggestion chips
│   ├── sidebar.py              # Sidebar with clear chat, campus map toggle, info
│   └── structured_cards.py     # Structured UI cards for navigation & emergency
├── core/
│   ├── config_loader.py        # Loads tenant UI configuration (defaults to iitj)
│   ├── models.py               # Data models: InstitutionUIConfig, ChatMessage, Conversation
│   └── session_manager.py      # Streamlit session state and history management
└── plugins/                    # Extensible UI widgets and plugin hooks
```

---

## 3. Runtime Flow

1. **Launcher:** `streamlit run app.py` executes the root launcher, which calls [`run_app()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/app.py#L21).
2. **Configuration & Session:** [`run_app()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/app.py#L21) retrieves the active institution ID (default: `"iitj"`), loads [`InstitutionUIConfig`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/models.py), injects curated CSS styles from [`styles.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/styles.py), and initializes session state in [`session_manager.py`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/session_manager.py).
3. **Rendering:**
   - [`render_sidebar()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/sidebar.py): Displays institution branding, session controls, and toggle switches for the campus interactive map.
   - [`render_chat_view()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/components/chat.py): Displays previous messages, quick-action chips for new conversations, and the primary chat input box.
4. **Turn Execution:** When the user enters a question or clicks a suggestion:
   - History is captured from session state.
   - User message is appended to the message list.
   - [`send_chat_turn()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py) forwards the question, history, session ID, and institution ID to [`handle_chat_turn()`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/service/assistant_service.py).
   - The returned [`AssistantResponse`](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py#L23) (containing answer, structured cards, citations, and follow-up suggestions) is stored and rendered.

---

## 4. Citation & Evidence Transparency

- **Source Separation:** At most 4 authoritative citations are rendered in collapsible UI cards below supported answers.
- **Abstention Integrity:** In accordance with platform policy, if an answer is a safe refusal or out-of-scope abstention, citation cards are suppressed to prevent misleading attribution.
- **Retrieval Separation:** Displayed UI citations are distinct from the larger candidate pool evaluated by the retrieval and ranking layers.

---

## 5. Launch Command

From the repository root:

```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/streamlit run app.py
```

---

## 6. Testing

The frontend is covered by automated unit and conversation-quality tests in `tests_frontend/`:

```bash
PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/iitj_rag/bin/pytest tests_frontend/ -v
```
