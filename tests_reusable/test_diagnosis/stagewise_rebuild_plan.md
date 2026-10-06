V1 Retrieval Architecture — Stagewise Rebuild Plan
Current decision
Do not rebuild the whole pipeline at once.
The current 15-case run gives enough evidence to freeze the healthy parts and isolate the destructive gate:
Stage	Current signal	Decision
1. Raw retrieval	Dense 11/15; BM25 15/15	Keep BM25/RRF stable; isolate dense weakness later
2. Verification	6 verified / 1 uncertain / 8 rejected for the benchmark's relevant candidate	Active investigation + first rebuild candidate
3. Ranking	14/15 broad relevant preservation, but same-key identity is not yet fully proven	Freeze until Stage 2 passes
4. Evidence boundary	Only 3/15 relevant evidence survived in the current run	Investigate after Stage 3; do not tune it yet
5. Assessment / audit	Downstream of missing evidence	Freeze
6. Packaging / final context	3/15 package-ready / non-empty	Freeze; it is downstream for now


Most important rule
A stage is not declared healthy because it does not throw an exception.
A stage is healthy only when the same candidate identity can be traced through the boundary with the expected evidence intact.
Stage 1 — Freeze and isolate
Keep:
- BM25
- RRF
- existing query benchmark
Only isolate the dense lane. Do not rewrite hybrid fusion while it retains the expected candidate on the current benchmark.
Pass condition for the frozen Stage 1 boundary:
BM25 -> RRF keeps the expected candidate across at least 15 cases.
Stage 2 — Verification forensic rebuild
This is the immediate workstream.
For every relevant fused candidate record:
1. candidate key
2. source
3. candidate text
4. query frame target
5. requested attribute/facet
6. required entities
7. target-grounded result
8. attribute-grounded result
9. relation-grounded result
10. scope result
11. semantic compatibility
12. exact rejection reasons
The current rejection trace repeatedly shows required_target_not_grounded even when scope_compatible=true. That is the first hypothesis to test, not a reason to blindly lower thresholds.
Stage-2 rebuild principles
- Preserve explicit conflict rejection.
- Preserve protected-entity rejection.
- Preserve target/attribute/relation anti-mixing protection.
- Remove duplicated or contradictory rescue paths.
- Make target grounding depend on the structured query frame in one canonical place.
- Never allow ranking to resurrect rejected_candidates.
- Keep lexical rescue bounded and explainable.
Stage-2 exit gate
At least 15 adversarial cases.
Target:
- relevant same-key verification survival >= 14/15
- no explicit-conflict candidate accepted
- no protected-entity mismatch accepted
- every rejected candidate has a reason that is independently inspectable
If this fails, return to Stage 2. Do not touch Stage 3.
Stage 3 — Ranking rebuild
Only after Stage 2 passes.
Input must be exactly verified_candidates (plus the explicitly defined uncertainty policy, if any).
Test:
- same-key preservation from verification to ranking
- top-1 correctness
- top-k relevance
- score monotonicity
- duplicate/diversity behavior
- no rejected candidate resurrection
At least 15 cases again.
If Stage 3 fails, first verify Stage 2 identity tracking. Only rebuild Stage 3 once Stage 2 is proven clean.
Stage 4 — Evidence boundary
Then inspect:
ranked candidates -> evidence groups -> local context -> scope filter
The current data already shows a potentially severe failure pattern: evidence grouping can create extra candidates and scope filtering can remove all candidates. This must be broken into separate measurements.
Required invariants:
- grouping must not silently replace the ranked evidence
- context expansion must be provenance-linked
- scope filtering must not remove the sole verified answer without a traceable reason
- no unrelated institution/program scope may enter final evidence
Stage 5 — Evidence assessment / claim audit
Only after Stage 4 provides valid evidence.
Verify:
- evidence status
- evidence score
- coverage status
- claim support
- unsupported/conflicting claim handling
Do not use Stage 5 to compensate for missing Stage-4 evidence.
Stage 6 — Packaging / final context
Final boundary:
verified evidence -> audited evidence -> packaged context -> answer
Pass conditions:
- non-empty context whenever sufficient evidence exists
- traceability preserved
- package cannot be ready when evidence candidates are empty
- no hidden fallback to pre-evidence ranked candidates
Loop rule
If a stage fails:
freeze downstream -> inspect immediate upstream identity -> rebuild only the failed stage -> rerun >=15 cases -> proceed
Never make a downstream stage more permissive to hide an upstream failure.