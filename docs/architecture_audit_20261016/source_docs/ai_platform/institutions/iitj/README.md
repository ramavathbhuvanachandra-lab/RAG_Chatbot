IIT Jodhpur Institution Adapter
This package is the IIT Jodhpur implementation of the reusable AI platform's
institution boundary.
Boundary
The package contains institution-specific knowledge and configuration only:
- deployment identity and corpus/vector-store paths;
- program/entity/topic/attribute/qualifier vocabulary;
- abbreviations, aliases and phrases;
- claim-scope dimensions;
- source-category metadata used as a weak ranking prior;
- navigation concepts.
The package does not own retrieval, RRF, ranking, verification, evidence
qualification, answer generation, grounding or guarding. Those mechanisms
belong to ai_platform/core.
Layout
ai_platform/institutions/iitj/
├── __init__.py
├── profile.py
├── semantic_registry.py
├── lexical.py
├── source_policy.py
├── scope_policy.py
├── navigation.py
├── campus_navigation.py      # existing; keep unchanged in this migration
├── emergency.py              # existing; keep unchanged in this migration
├── prompts.py                # existing; keep unchanged in this migration
└── data/
    ├── aliases.json
    ├── acronyms.json
    ├── phrases.json
    ├── semantic_terms.json
    ├── source_policy.json
    ├── scope_policy.json
    └── navigation.json
data/ is configuration, not the factual document corpus. The current IITJ
corpus remains under the repository-level data/data_iitj/ path and the
existing Chroma collection remains iitj_v1.
Design rule
Do not add rules such as if query == ... here. Add institution vocabulary
and configuration. The reusable core decides how those signals affect
retrieval, verification and evidence selection.