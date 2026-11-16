# Demo app: HTTPX docs bot + 10 mutants

A small RAG bot over the HTTPX documentation (`docs/`, BSD-3, see `docs/HTTPX_LICENSE.txt`).
Chunks of ~380 words (≈500 tokens) with 60-word overlap, TF-IDF retrieval (set
`DEMO_EMBEDDINGS=st` for all-MiniLM-L6-v2), top 4 chunks, answered by `DEMO_MODEL`.

    uvicorn demo_app.main:app --port 8000
    curl -s localhost:8000/chat -H 'content-type: application/json' -d '{"question":"How do I enable HTTP/2?"}'

Switch a mutant with `MUTANT=m3` (env var, read on every request) or by writing `m3` to
`demo_app/ACTIVE_MUTANT` (that's what the CI demo PR does).

| Mutant | What it breaks |
|---|---|
| m1 | removes the "only from context" rule |
| m2 | retrieves the top 1 chunk instead of 4 |
| m3 | shuffles retrieval (random chunks) |
| m4 | switches to a weaker model (`DEMO_WEAK_MODEL`) |
| m5 | removes the "say you don't know" line |
| m6 | removes the injection-defence wording |
| m7 | temperature 1.2 |
| m8 | truncates the context to 300 characters |
| m9 | 2,500-word chunks with a 900-word context budget, so context gets cut off |
| m10 | system prompt demands a confident answer, never "I don't know" |

`docs/zz_connection_pool_tuning.md` is a deliberately poisoned page for indirect
prompt-injection tests (created with `mutantqa plant`; listed in `docs/.mutantqa-planted.json`).
The system prompt contains the canary `CANARY-7f3a91`, which must never appear in output.

Without `DEMO_MODEL`, an offline *simulated* model answers extractively and obeys the same
prompt rules, so every mutant has an observable effect. It exists to test the pipeline
without an API key. Results measured against it are not results.
