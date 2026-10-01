# Institution Change Map

## Purpose

This repository is a complete IIT Jodhpur chatbot product.

The RAG/application machinery is intentionally kept mostly shared.
Institution-specific behavior is isolated as much as practical so the
repository can later be copied for another institution.

The goal is NOT to make every file universally reusable.

The goal is:

1. Build IIT Jodhpur correctly.
2. Keep institution-specific behavior identifiable.
3. Copy the complete repository for a new college.
4. Change only the institution-dependent areas.
5. Avoid rewriting the RAG engine for every college.

---

# 1. DO NOT NORMALLY CHANGE

These are shared RAG/application components.

Examples:

- retrieval orchestration
- dense retrieval
- BM25
- RRF
- reranking
- evidence assembly
- evidence sufficiency
- evidence coverage
- grounding
- claim verification
- conversation resolution
- multi-intent processing
- LangGraph orchestration
- persistence fallback behavior
- generic LLM interface
- evaluation infrastructure

Current locations include:

backend/
    answer_claim_audit.py
    answer_grounding_scope.py
    answer_grounding.py
    answer_guard.py
    claim_context_filter.py
    claim_scope.py
    contradiction_handler.py
    conversation_resolver.py
    conversation_session.py
    evidence_coverage.py
    evidence_groups.py
    evidence_scope_gate.py
    evidence_temporal_numeric.py
    evidence.py
    final_evidence_scope.py
    graph.py
    local_context.py
    multi_intent.py
    multi_intent_answer.py
    multi_intent_situation.py
    nodes.py
    retrieval_diversity.py
    retrieval_query_planner.py
    retriever.py
    situation_decision.py
    situation_retrieval_control.py
    situation_retrieval_integration.py
    situational_answer.py
    state.py
    supported_claim_preservation.py
    utils.py

IMPORTANT:
A file being listed here does not mean it can never be modified.
It means it should not be changed merely because a new college was added.

---

# 2. INSTITUTION-SPECIFIC CHANGE SURFACE

This is the primary area to inspect when creating a new college deployment.

Current IIT Jodhpur-specific areas include:

backend/institutions/iitj/

data/data_iitj/

data/campus_locations.json
data/emergency_contacts.json

Likely institution-dependent backend behavior includes:

backend/campus_navigation.py
backend/emergency.py
backend/prompts.py
backend/student_situation.py
backend/assistant_router.py

These files must be reviewed before copying to another institution.

---

# 3. CONFIGURATION / DEPLOYMENT

These files are not necessarily institution-specific, but they may expose
institution-dependent settings:

backend/config.py
backend/llm.py
backend/embedding.py
backend/ingestion.py
backend/vectorstore.py

Prefer configuration/profile changes over code changes.

Institution facts must remain in the institution knowledge base,
not inside generic configuration.

---

# 4. IIT JODHPUR DATA

The current IIT Jodhpur RAG corpus is stored under:

data/data_iitj/

This includes the organized institutional knowledge used by the chatbot.

When deploying another college:

- replace the institutional corpus
- rebuild the vectorstore
- verify metadata/source information
- run the institution regression suite

Do not copy IIT Jodhpur factual content into the new college.

---

# 5. UI / BRANDING

The frontend is part of the complete product.

Current UI:

UI/

When deploying another college, review:

- institution name
- logo/branding
- welcome text
- assistant name
- navigation/campus UI
- colors/theme where institution-specific
- contact information
- institution-specific labels

Do not modify RAG logic merely to change UI branding.

---

# 6. DEPLOYMENT

The repository also contains deployment infrastructure:

Dockerfile
requirements.txt
requirements_gpu.txt

Deployment settings may change between institutions or environments.

Keep deployment-specific changes separate from RAG logic whenever possible.

---

# 7. TESTING RULE

Existing legacy tests in:

tests/

must not be casually rewritten or deleted.

New reusable/IITJ architecture tests belong under:

tests_reusable/

Test outputs belong under:

test_outputs/

Every significant production failure should become a regression test.

---

# 8. NEW COLLEGE WORKFLOW

When creating a new college chatbot:

1. Copy the complete working repository.

2. Create a new institution profile.

3. Replace the institution corpus.

4. Rebuild the vectorstore.

5. Review the institution-specific backend change surface.

6. Update institution-specific prompts/behavior where required.

7. Update structured institution data such as campus/contact information.

8. Update UI branding.

9. Update deployment configuration.

10. Run the full institution regression suite.

11. Fix institution-specific failures first.

12. Only modify shared core logic when the behavior is genuinely generic
    and the change is proven beneficial across institutions.

---

# 9. DESIGN PRINCIPLE

Do not ask:

"How can every single file become reusable?"

Ask:

"Which parts are genuinely common across institutions, and which parts
belong to this institution?"

Correctness comes before abstraction.

The IIT Jodhpur chatbot is the first complete product.

Reusability is achieved by isolating what changes, not by prematurely
generalizing everything.
