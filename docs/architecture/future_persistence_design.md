# Architectural Blueprint — Future Persistence and Multi-Tenant Storage Design

**Date**: October 18, 2026  
**Author**: Senior Staff AI Engineer & Application Architect  
**Target Repository**: IIT Jodhpur & Reusable Academic Core RAG Platform  
**Scope**: Temporary UI Session State to Durable Multi-Tenant Persistence Boundary  
**Status**: Architecture Design Document (Zero Production Database Activation in Milestone 1)

---

## 1. Executive Overview

Milestone 1 establishes a clean, decoupled frontend architecture that operates in a stateless, local-development mode without requiring Supabase credentials, phone OTP validation, or persistent database tables. All active conversations, messages, and session IDs reside temporarily in Streamlit's in-memory `st.session_state`.

This document specifies the target architecture for transitioning from temporary in-memory sessions to durable, multi-tenant persistence. It defines:
1. The boundary between ephemeral UI session state and the future persistence service.
2. The mapping from local conversation IDs (`chat_<uuid>`) to durable database entities.
3. The schema structures from `backend/*_db.py` eligible for reuse.
4. Tenant isolation invariants (`institution_id` partition keys).
5. Unverified schema decisions and required migration prerequisites.

---

## 2. In-Memory Session vs. Durable Persistence Boundary

The Streamlit UI application interacts exclusively through high-level abstractions:
- The UI layer calls [frontend/core/session_manager.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/session_manager.py) to create threads and append messages.
- The service client [frontend/clients/assistant_client.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/clients/assistant_client.py) passes `session_id` and `institution_id` to [ai_platform/service/assistant_service.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/service/assistant_service.py).

```mermaid
graph TD
    subgraph FrontendUI ["Frontend Presentation Layer"]
        UI["components/chat.py"]
        SessMgr["core/session_manager.py"]
        Client["clients/assistant_client.py"]
    end

    subgraph ServiceBoundary ["Canonical Assistant Service"]
        Service["ai_platform/service/assistant_service.py"]
        Engine["ai_platform/runtime/engine.py"]
    end

    subgraph FuturePersistence ["Future Application Data Layer (Phase 2)"]
        Store["application/persistence/session_store.py"]
        DBClient["application/database/client.py"]
        Supabase[(Supabase PostgreSQL)]
    end

    UI -->|append user/assistant turn| SessMgr
    SessMgr -.->|Future: write-through cache| Store
    UI -->|send turn| Client
    Client -->|ChatTurnRequest(session_id, tenant)| Service
    Service --> Engine
    Store --> DBClient --> Supabase
```

### Invariants:
1. **Zero Database Coupling in Frontend**: Neither `frontend/app.py` nor any component under `frontend/components/` may import database drivers (`supabase`, `psycopg2`, `sqlalchemy`) or execute SQL/REST mutations directly.
2. **Best-Effort Asynchronous Persistence**: Database write failures or latency spikes must never block the interactive Streamlit user turn. Writes to durable storage will follow a safe, best-effort write-through pattern.
3. **Graceful Fallback**: If the database is unreachable, the application continues to operate seamlessly using the current in-memory `st.session_state` without throwing unhandled exceptions to the user.

---

## 3. Session and Entity Mapping

In Milestone 1, each conversation thread is identified by a unique, locally generated UUID string:
```python
conv_id = f"chat_{uuid.uuid4().hex[:10]}"
```

When durable persistence is activated, this identifier maps directly into the existing database schema:

| In-Memory UI Entity | Durable Database Table | Schema Fields | Key Mapping / Role |
| :--- | :--- | :--- | :--- |
| **`st.session_state.conversations[conv_id]`** | `sessions` | `id (UUID)`, `user_id (UUID)`, `title (TEXT)`, `created_at`, `institution_id (TEXT)` | Primary key `sessions.id` matches the UI conversation ID. |
| **`ChatMessage` (user turn)** | `messages` | `id (UUID)`, `session_id (UUID)`, `user_id (UUID)`, `role='user'`, `content (TEXT)`, `created_at` | Scoped to `session_id`. Stored sequentially. |
| **`ChatMessage` (assistant turn)** | `messages` | `id (UUID)`, `session_id (UUID)`, `role='assistant'`, `content (TEXT)`, `response_time (NUMERIC)`, `sources (JSONB)` | Preserves verified citations, structured navigation blocks, and latency metrics. |
| **User Identity** | `users` | `id (UUID)`, `phone_number (TEXT)`, `name (TEXT)`, `institution_id (TEXT)` | In local demo mode, defaults to `ANON_<client_uuid>`. In production, set via authentication. |
| **User Feedback (Future)** | `message_feedback` | `id (UUID)`, `message_id (UUID)`, `rating (INT)`, `comment (TEXT)` | Bound to specific assistant `message_id`. |

---

## 4. Reusable Backend Table Helpers

The repository contains pre-existing database access modules under `backend/`:
- `backend/user_db.py`: `get_or_create_user(name, phone_number)`
- `backend/session_db.py`: `create_session(user_id, title)`, `get_user_sessions(user_id)`
- `backend/message_db.py`: `save_message(session_id, user_id, role, content, response_time)`
- `backend/feedback_db.py`: `save_feedback(message_id, rating, comment)`

### Migration Strategy to `application/`:
When persistence is implemented, these files will be relocated into an `application/` package (e.g., `application/persistence/` and `application/database/`) rather than imported from `backend/`. This ensures that `backend/` can be retired completely in Milestone 4B2.

---

## 5. Multi-Tenant Isolation & Access Control Invariants

To safely support multiple institutions (IIT Jodhpur, plus future onboarded colleges) sharing the same platform database, strict multi-tenant isolation rules must be enforced:

1. **Mandatory Tenant Foreign Key**:
   Every persistent record (`users`, `sessions`, `messages`) must include an `institution_id` column.
2. **Row Level Security (RLS)**:
   In Supabase / PostgreSQL, Row Level Security policies must enforce:
   ```sql
   ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
   CREATE POLICY tenant_isolation_sessions ON sessions
       USING (institution_id = current_setting('app.current_tenant', true));
   ```
3. **No Cross-Tenant Query Leaks**:
   API and database query wrappers must never omit `institution_id` in `WHERE` clauses:
   ```python
   # Correct multi-tenant query pattern
   client.table("sessions") \
         .select("*") \
         .eq("user_id", user_id) \
         .eq("institution_id", active_tenant_id) \
         .execute()
   ```
4. **Tenant Switch State Invalidation**:
   As implemented in [frontend/core/session_manager.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/session_manager.py), whenever the active `institution_id` is modified, all previous conversation threads and cached histories are immediately segregated or cleared.

---

## 6. Unverified Schema Decisions & Pre-Activation Checklist

Prior to activating production database persistence, the following technical prerequisites must be verified against the staging Supabase instance:

- [ ] **Verify `institution_id` Column Presence**: Confirm that the existing staging `sessions` and `messages` tables contain an `institution_id` column. If missing, generate an explicit SQL migration (`ALTER TABLE sessions ADD COLUMN institution_id VARCHAR(32) NOT NULL DEFAULT 'iitj'`).
- [ ] **Sources JSONB Column**: Confirm `messages` table supports a `sources JSONB` column to persist verified citation excerpts, source document names, and headings.
- [ ] **Composite Indices**: Create composite indices on `(institution_id, user_id)` and `(session_id, created_at)` to guarantee sub-50ms message history retrieval.
- [ ] **Connection Pooling**: Configure PgBouncer connection pooling for Streamlit multi-user deployments to prevent socket exhaustion during concurrent sessions.
