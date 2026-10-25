# Architectural Specification — Institution UI Plugin and Onboarding Contract

**Date**: October 18, 2026  
**Author**: Senior Staff AI Engineer & Application Architect  
**Target Repository**: IIT Jodhpur & Reusable Academic Core RAG Platform  
**Scope**: Multi-College Extensibility, UI Plugin Protocol, and Institutional Onboarding  
**Status**: Formal Architectural Contract

---

## 1. Overview and Core Philosophy

The primary objective of the new frontend architecture is to enable **zero-duplication multi-institution onboarding**. When a new academic institution is added to the platform:
- **No Shared UI Code Is Cloned**: All colleges share the exact same Streamlit application, sidebar components, chat composers, message renderers, and citation accordions.
- **No RAG Engine Duplication**: The core retrieval, ranking, reciprocal rank fusion (RRF), evidence boundary verification, and answer generation engines remain strictly centralized under [ai_platform/core/](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/core/).
- **Decoupled Responsibilities**:
  1. **UI Presentation Plugin** (`frontend/plugins/<tenant>/`): Governs branding, welcome copy, suggested quick actions, and optional features (campus map, emergency cards).
  2. **AI Institutional Profile & Policies** (`ai_platform/institutions/<tenant>/`): Governs official acronyms, scope boundaries, program aliases, and structured lookup handlers.
  3. **Knowledge Corpus & Vector Index** (`data/<tenant>/` or vectorstore): Governs official PDFs, DOCX ordinances, and embeddings.

---

## 2. The Separation of Concerns Boundary

A critical security and architectural rule governs the system:
> **A logo or UI theme configuration is NEVER an authorization to access institutional knowledge.**

```mermaid
graph TD
    subgraph UIPlugin ["1. UI Presentation Plugin (frontend/plugins/)"]
        Logo["Logo Icon & Colors"]
        Copy["Welcome Copy & Taglines"]
        QA["Quick Action Suggested Inquiries"]
        OptionalMap["has_campus_map: bool"]
    end

    subgraph AIProfile ["2. Institutional AI Profile (ai_platform/institutions/)"]
        Profile["InstitutionProfile (data_path, collection)"]
        Scope["Scope Policy & Out-of-Scope Gates"]
        Lexical["Acronyms & Aliases"]
        Handlers["STRUCTURED_HANDLERS (navigation, emergency)"]
    end

    subgraph DataCorpus ["3. Knowledge Corpus & Vectorstore"]
        Docs["Official Academic Ordinances & Fee Tables"]
        Chroma["ChromaDB Collection (nomic-embed-text)"]
    end

    UIPlugin -->|Configure Presentation| StreamlitApp["frontend/app.py"]
    StreamlitApp -->|ChatTurnRequest(institution_id)| AssistantService["ai_platform/service/assistant_service.py"]
    AssistantService -->|Load Policy & Handlers| AIProfile
    AssistantService -->|Retrieve Verified Evidence| DataCorpus
```

---

## 3. The UI Plugin Interface Definition

Every institution provides a lightweight plugin implementing [frontend/core/plugin_registry.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/plugin_registry.py):

```python
from frontend.core.models import InstitutionUIConfig
from frontend.core.plugin_registry import InstitutionUIPlugin

class NewCollegeUIPlugin(InstitutionUIPlugin):
    @property
    def institution_id(self) -> str:
        return "new_college"

    def get_config(self) -> InstitutionUIConfig:
        return InstitutionUIConfig(...)
```

### Typed Configuration Fields (`InstitutionUIConfig`):

| Parameter | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `institution_id` | `str` | **Yes** | Lowercase alphanumeric identifier matching `[a-z0-9_\-]+`. Must match the AI platform profile ID. |
| `display_name` | `str` | **Yes** | Human-readable name (e.g. `"IIT Jodhpur"`). |
| `tagline` | `str` | Optional | Subtitle displayed in header banner and sidebar. |
| `logo_icon` | `str` | Optional | Emoji or icon string (default: `"🎓"`). |
| `primary_color` | `str` | Optional | Hex CSS color for primary elements (e.g. `"#0F294A"`). |
| `accent_color` | `str` | Optional | Hex CSS color for interactive hover states (e.g. `"#2563EB"`). |
| `welcome_message` | `str` | Optional | Initial assistant greeting message rendered in new chats. |
| `supported_topics` | `tuple[str, ...]` | Optional | Bullet points rendered in the sidebar "ℹ️ About" accordion. |
| `quick_actions` | `tuple[QuickActionItem, ...]` | Optional | Suggested prompt cards shown on the initial conversation turn. |
| `has_campus_map` | `bool` | Optional | If `True`, enables the interactive 2-column location directory. |
| `campus_map_locations` | `tuple[CampusLocationItem, ...]` | Optional | Locations with Google Maps links and operating hours. |
| `has_emergency_cards` | `bool` | Optional | If `True`, enables structured emergency contact rendering. |
| `emergency_contacts` | `tuple[EmergencyContactItem, ...]` | Optional | Emergency contacts with phone numbers. |
| `footer_text` | `str` | Optional | Attribution text at bottom of sidebar. |

---

## 4. Concrete Example: IIT Jodhpur (`iitj`)

IIT Jodhpur represents a fully featured institution enabling all optional capabilities:
- **Branding**: Navy (`#0F294A`) and Cobalt Blue (`#2563EB`).
- **Campus Map**: `has_campus_map=True`, loaded dynamically from [data/campus_locations.json](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/data/campus_locations.json).
- **Emergency Cards**: `has_emergency_cards=True`, loaded from [data/emergency_contacts.json](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/data/emergency_contacts.json).
- **Quick Actions**: 4 preset inquiries for Admissions, Academics, Hostels, and Emergency.

Implementation is isolated in:
[frontend/plugins/iitj/plugin.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/plugins/iitj/plugin.py)

---

## 5. Minimal Example: Future College Without Campus Map (`template`)

A future institution that does not have campus map coordinates or custom emergency numbers can omit those features simply by setting flags to `False`:

```python
class TemplateUIPlugin(InstitutionUIPlugin):
    @property
    def institution_id(self) -> str:
        return "template"

    def get_config(self) -> InstitutionUIConfig:
        return InstitutionUIConfig(
            institution_id="template",
            display_name="Example Institute of Technology",
            tagline="Academic & Campus AI Assistant",
            logo_icon="🏛️",
            has_campus_map=False,       # Map button is automatically hidden!
            has_emergency_cards=False,  # Emergency cards gracefully omitted
            supported_topics=("Admissions", "Academic Regulations"),
            quick_actions=(
                QuickActionItem(
                    id="qa_admissions",
                    label="Admissions",
                    prompt="What are the admission requirements?",
                    icon="🎓",
                ),
            ),
        )
```

The shared UI automatically adapts:
- The `🗺️ Campus Map & Location Directory` button is **hidden**.
- No campus map layout code is executed.
- Regular RAG question answering remains 100% active.

---

## 6. Tenant Resolution and Security Contract

1. **Resolution Source**:
   The active tenant is resolved from the environment variable `INSTITUTION_ID` (default: `"iitj"`).
2. **Cross-Tenant Validation**:
   When [frontend/core/config_loader.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/frontend/core/config_loader.py) loads the UI configuration, it cross-checks with [ai_platform/institutions/loader.py](file:///Users/bhuvanachandra/Desktop/RAG_Chatbot/ai_platform/institutions/loader.py):
   ```python
   profile = load_institution_profile(tenant_id)
   if profile.institution_id.lower() != config.institution_id.lower():
       raise ValueError("Tenant identity mismatch!")
   ```
3. **Turn Execution Boundary**:
   The frontend client packages `institution_id` into the `ChatTurnRequest`:
   ```python
   req = ChatTurnRequest(question=prompt, chat_history=history, institution_id="iitj")
   ```
   The assistant service guarantees that retrieval, vector collections, scope policies, and fallback messages are strictly partitioned by that verified `institution_id`.

---

## 7. Institutional Onboarding Checklist

When onboarding another college, follow this step-by-step verification checklist:

- [ ] **Step 1: Ingest Corpus**: Chunk and index the new institution's official documents into a distinct Chroma collection (`<institution_id>_v1`).
- [ ] **Step 2: Define AI Profile**: Create `ai_platform/institutions/<institution_id>/profile.py` with `InstitutionProfile` declaring collection, fallback message, and supported languages.
- [ ] **Step 3: Define Scope Policies**: Add `scope_policy.json` and acronym mappings for the institution under `ai_platform/institutions/<institution_id>/data/`.
- [ ] **Step 4: Create UI Plugin**: Create `frontend/plugins/<institution_id>/plugin.py` implementing `InstitutionUIPlugin`.
- [ ] **Step 5: Register Plugin**: Register the plugin in `frontend/core/plugin_registry.py`.
- [ ] **Step 6: Run Verification Tests**:
  - Run `python3 -m unittest tests_frontend/test_frontend_core.py`.
  - Run `python3 -m unittest tests_reusable/production_readiness/test_synthetic_institution.py`.
- [ ] **Step 7: Launch Local Tenant Demo**:
  ```bash
  INSTITUTION_ID=<institution_id> streamlit run app.py
  ```
