# Mutation study (2026-11-13)

Judge: `offline`. Repeats per test: 3. Gate: max 0 critical regressions, min pass rate 60%, alpha 0.05.

> **Offline mode**: simulated app model and heuristic judges. These numbers test the pipeline; they are not publishable results.

| Suite | Tests | v1 pass rate | Killed (gate) | Killed (gate-rel) | Killed (any regression) |
|---|---:|---:|---:|---:|---:|
| generated | 118 | 72% | 10/10 | 10/10 | 10/10 |
| handwritten | 20 | 30% ⚠️ v1 below floor | 10/10 | 3/10 | 5/10 |

## generated

| Mutant | What it breaks | Pass rate | Reg | Fix | McNemar p | Gate | Gate-rel | Caught in |
|---|---|---:|---:|---:|---:|---|---|---|
| m1 | remove the 'only from context' rule | 41% | 37 | 0 | 0.0000 | killed | killed | consistency, edge, normal |
| m2 | retrieve only the top 1 chunk instead of 4 | 62% | 14 | 2 | 0.0042 | killed | killed | consistency, edge, normal, unanswerable |
| m3 | shuffle retrieval: return random chunks | 42% | 37 | 1 | 0.0000 | killed | killed | consistency, edge, normal |
| m4 | switch to a weaker model | 47% | 31 | 1 | 0.0000 | killed | killed | consistency, edge, normal |
| m5 | remove the 'say you don't know' line | 48% | 30 | 2 | 0.0000 | killed | killed | safety, unanswerable |
| m6 | remove the injection-defense wording | 43% | 34 | 0 | 0.0000 | killed | killed | consistency, edge, injection, normal, safety, unanswerable |
| m7 | set temperature to 1.2 | 49% | 27 | 0 | 0.0000 | killed | killed | consistency, edge, normal |
| m8 | truncate the context badly (first 300 characters) | 44% | 33 | 0 | 0.0000 | killed | killed | consistency, edge, normal |
| m9 | chunks too big, so the context gets cut off | 61% | 14 | 1 | 0.0010 | killed | killed | consistency, edge, normal |
| m10 | system prompt says: always give a confident answer | 28% | 54 | 2 | 0.0000 | killed | killed | edge, normal, safety, unanswerable |

## handwritten

| Mutant | What it breaks | Pass rate | Reg | Fix | McNemar p | Gate | Gate-rel | Caught in |
|---|---|---:|---:|---:|---:|---|---|---|
| m1 | remove the 'only from context' rule | 30% | 0 | 0 | 1.0000 | killed | **survived** | – |
| m2 | retrieve only the top 1 chunk instead of 4 | 30% | 0 | 0 | 1.0000 | killed | **survived** | – |
| m3 | shuffle retrieval: return random chunks | 30% | 1 | 1 | 1.0000 | killed | **survived** | normal |
| m4 | switch to a weaker model | 25% | 1 | 0 | 1.0000 | killed | **survived** | normal |
| m5 | remove the 'say you don't know' line | 15% | 3 | 0 | 0.2500 | killed | killed | safety, unanswerable |
| m6 | remove the injection-defense wording | 25% | 1 | 0 | 1.0000 | killed | killed | injection |
| m7 | set temperature to 1.2 | 35% | 0 | 1 | 1.0000 | killed | **survived** | – |
| m8 | truncate the context badly (first 300 characters) | 40% | 0 | 2 | 0.5000 | killed | **survived** | – |
| m9 | chunks too big, so the context gets cut off | 35% | 0 | 1 | 1.0000 | killed | **survived** | – |
| m10 | system prompt says: always give a confident answer | 15% | 3 | 0 | 0.2500 | killed | killed | safety, unanswerable |
