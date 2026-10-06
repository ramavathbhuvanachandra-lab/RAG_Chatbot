# AI Platform V1 — Production Diagnosis Suite

This suite tests **only `ai_platform/core`**. It does not import, execute, or modify `backend/`.

## Diagnostic path

`question` → conversation → query understanding → multi-intent → hybrid retrieval → RRF/fusion → verification+ranking → evidence context → evidence sufficiency → coverage → claim audit → evidence package → optional answer LLM.

The runner is designed for diagnosis, not for making the code pass. It records the state after every real graph node and identifies the first stage where useful evidence disappears or wrong evidence becomes dominant.

## Run the first 6 questions

```bash
PYTHONPATH=. python -u tests_reusable/test_diagnosis/run_diagnosis.py --limit 6
```

## Run all 12

```bash
PYTHONPATH=. python -u tests_reusable/test_diagnosis/run_diagnosis.py --limit 12
```

## Run one case

```bash
PYTHONPATH=. python -u tests_reusable/test_diagnosis/run_diagnosis.py --case mtech_regular_eligibility
```

## Include the answer LLM

Start with 6 cases: `--with-llm`. This may be slow. The first pass should normally be without it so retrieval/evidence failures are isolated from generation latency.

## Reports

- `reports/retrieval_diagnosis.md` — human-readable diagnosis
- `reports/retrieval_diagnosis.json` — complete machine-readable trace

The benchmark uses fixed diagnostic markers from the IIT Jodhpur material previously supplied for this project. Marker hits are diagnostic heuristics; the stage trace is the primary evidence.
