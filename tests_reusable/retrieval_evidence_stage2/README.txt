RETRIEVAL / RANKING / VERIFICATION / EVIDENCE FORENSICS — STAGE 2

Purpose
-------
Diagnose the existing production RAG retrieval path before changing thresholds or
rewriting the retriever.

Architecture measured
=====================
Query contract
  -> bounded Dense + BM25 + lexical recall
  -> weighted RRF
  -> candidate verification
  -> ranking
  -> verified-anchor evidence boundary
  -> evidence sufficiency
  -> evidence coverage
  -> claim audit
  -> evidence packaging / final LLM context

Important
=========
This bundle changes NO production retrieval, ranking, verification, evidence,
or UI logic. It adds an observable forensic runner and small isolated contract tests.
That is intentional: we need evidence before making production changes.

Test set
========
48 realistic cases:
- 12 easy
- 12 medium
- 12 hard
- 12 adversarial/system-break cases

The adversarial set includes prompt injection attempts, cross-program contamination,
coverage dishonesty, malformed input, and three random/garbage-style inputs.

Diagnosis labels
================
GOOD
SAFE_REFUSAL
QUERY_CONTRACT
RETRIEVAL_RECALL
RETRIEVAL_OR_ALIGNMENT
VERIFICATION_DROP
EVIDENCE_BOUNDARY_DROP
PACKAGING_OR_ASSESSMENT
INSUFFICIENT_OR_UNRESOLVED
ADVERSARIAL_POLICY_FAILURE
ERROR

Each case records
=================
- query/frame contract
- retrieval query variants
- Dense/BM25/lexical list sizes
- RRF weights and top fused candidates
- verification decisions and rejection reasons
- ranked candidates and scores
- lexical fallback usage
- evidence groups and evidence boundary trace
- evidence sufficiency / coverage
- claim audit
- final evidence context size + preview
- per-stage latency

Production changes
==================
None.

Next step after the run
=======================
Use the first-failure diagnosis counts and individual traces to make the smallest
real production change. Do not lower verification thresholds globally without a
trace showing that a correct candidate is being rejected incorrectly.
