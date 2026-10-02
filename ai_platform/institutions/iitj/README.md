RAG Retrieval Hardening Bundle
What this changes
This bundle is a correctness consolidation of the existing reusable RAG architecture. It does not introduce another retrieval stack.

The core invariant is:

understand -> retrieve broadly -> verify exact target/scope -> rank -> package only compatible evidence -> answer from that evidence

Changes included
backend/core/nodes.py

promotes uniquely detected institution entities/programs into the canonical query target when safe

excludes generic topic labels from protected-entity enforcement

preserves protected target/entity terms in retrieval variants

limits final ranked evidence to a bounded set before context expansion

limits context expansion fan-out

passes required entity constraints into the answer generator

backend/core/candidate_verification.py

protected entity grounding is required for resolved entities

lexical rescue cannot bypass a protected entity mismatch

generic verifier no longer relies on a fixed engineering-degree equivalence vocabulary

backend/core/retrieval/ranking.py

target/entity alignment is explicit

generic core no longer contains institution/program-name source markers

backend/core/retrieval_contracts.py

canonical local source identity collapses equivalent absolute/relative data/ paths without hardcoding an institution

backend/core/query/understanding.py

program-specific backstop vocabulary removed; institution registry supplies program detection

request type now participates in strict retrieval requirements

backend/core/answering/generator.py

answer generation receives required entities and known program names

model output introducing an unrequested program is rejected/repaired rather than accepted as grounded

tests_reusable/final_rag_core/test_retrieval_target_integrity.py

wrong-program rejection

exact-program acceptance

wrong-department rejection

answer-level unrequested-program rejection

Important limitation
The files are syntax-validated and the deterministic target-integrity tests pass in isolation. They have not been run against the user's live Mac Ollama/Chroma runtime in this environment. The live Node/E2E gate is therefore still required.

Recommended live sequence
Back up the current working tree.

Overlay the bundled files at their same paths.

Run:

python -m py_compile \
  backend/core/candidate_verification.py \
  backend/core/nodes.py \
  backend/core/retrieval_contracts.py \
  backend/core/query/models.py \
  backend/core/query/understanding.py \
  backend/core/retrieval/ranking.py \
  backend/core/answering/generator.py

PYTHONPATH=. python tests_reusable/final_rag_core/test_retrieval_target_integrity.py
PYTHONPATH=. python tests_reusable/final_rag_core/test_core_evidence_flow.py

# Then run the existing real Node-2 / deep diagnostic.
Do not rebuild Chroma solely because these files changed; retrieval semantics are being changed, not the embedding model or stored vectors. Rebuild only if your existing ingestion/index state is stale or the live test proves an index/data problem.

