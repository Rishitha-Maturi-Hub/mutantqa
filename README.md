# mutantqa

**QA for LLM apps, built the way a QA engineer thinks.** It generates test suites for your RAG bot or
chatbot, scores the answers, flags flaky tests, and blocks a pull request only when quality drops by more
than noise. It proves the suite works by breaking the app on purpose and measuring how much it catches.

![Dashboard](docs/dashboard.gif)
<!-- Record with the dashboard open on a v1 vs mutant comparison. 2-3 min demo video: docs/demo_script.md -->

## Quickstart

```bash
pip install mutantqa                      # or: pip install -e ".[demo,dashboard]" from this repo
mutantqa init                             # writes mutantqa.yaml: point it at your app
mutantqa generate                         # builds tests/suite.jsonl (commit it, like code)
mutantqa run --tag v1                     # runs every test 3x, scores, saves to SQLite
mutantqa compare v1 v2 --gate --markdown report.md   # exit code 1 if the change made things worse
```

Try it without an API key on the bundled demo app (simulated model, heuristic judges):

```bash
pip install -e ".[demo,dashboard]"
mutantqa run --tag v1 -c mutantqa.offline.yaml
MUTANT=m6 mutantqa run --tag broken -c mutantqa.offline.yaml
mutantqa compare v1 broken --gate -c mutantqa.offline.yaml
streamlit run dashboard/app.py -- --config mutantqa.offline.yaml
```

## Why it exists

Prompt and model changes break LLM apps in ways unit tests can't see: a reworded system prompt stops the
bot from saying "I don't know", a cheaper model invents API parameters, a retrieval tweak makes answers
drift. Teams usually notice in production. mutantqa brings four habits from software QA to that problem:

1. **Mutation testing.** The demo app ships 10 deliberately broken variants (`demo_app/README.md`). The
   question "how good is this test suite?" becomes a number: the share of broken variants it catches.
2. **Flakiness detection.** Every test runs N times. The verdict is the majority, and tests whose repeats
   disagree are labelled flaky instead of silently passing or failing.
3. **Statistically honest gating.** Pass/fail per test is paired between runs, so the gate uses McNemar's
   exact test on the tests that changed. Two random regressions in 120 tests don't block a PR; a real drop
   does. Critical categories (injection, safety) can block on a single regression.
4. **Deterministic injection checks.** A canary string in the app's system prompt and marker words in the
   attacks make most injection verdicts a string match, not a judgement call. Planted documents test
   *indirect* injection through retrieval.

It also publishes how often its LLM judge agrees with humans (`experiments/human_agreement.py`), because
an evaluation tool that doesn't measure its own judge is asking to be trusted.

### Related tools

There are good tools in this space, and mutantqa overlaps with several. Read their docs before you
pick one.

| Tool | What it's mainly for |
|---|---|
| [DeepEval](https://github.com/confident-ai/deepeval) | pytest-style LLM evals with many metrics; synthetic data generation |
| [promptfoo](https://github.com/promptfoo/promptfoo) | config-driven prompt/model comparisons, red-teaming, CI integration |
| [Ragas](https://github.com/explodinggradients/ragas) | RAG metrics (faithfulness, context precision/recall) and test-set generation |
| [Giskard](https://github.com/Giskard-AI/giskard) | vulnerability scans and RAG test generation |
| [TruLens](https://github.com/truera/trulens) | feedback functions and tracing for LLM apps |
| [LangSmith](https://smith.langchain.com) | tracing, datasets and evaluation platform (hosted) |
| [OpenAI Evals](https://github.com/openai/evals) | eval framework and registry |
| [garak](https://github.com/NVIDIA/garak), [PyRIT](https://github.com/Azure/PyRIT) | security probing and red-teaming |

mutantqa's angle is narrower: measure the *test suite* (mutation kill rate), treat non-determinism as a
first-class result (flaky tests), gate on significance rather than raw deltas, and publish judge-human
agreement. If you only need metrics or red-teaming, the tools above are more mature.

## Results

> Fill in after the real runs (Steps 5 and 7). The offline row only shows the pipeline works.

| Metric | Result |
|---|---|
| Mutants caught, generated suite vs handwritten (real model + judge) | __/10 vs __/10 |
| Judge–human agreement (κ / accuracy), 200 samples | __ / __% |
| Human–human agreement (κ) | __ |
| Real bugs found across 3 open-source apps | __ (__ confirmed by maintainers) |
| Flaky test rate | __% |
| Cost per full suite run (120 tests × 3 repeats) | $__ |
| *Pipeline sanity check (offline simulator, heuristic judges; not a result)* | *10/10 vs 3/10 (relative gate)* |

Full tables: `experiments/results/`. CI catching a mutant: ![CI failure](docs/ci_failure.png)

## How scoring works

Cheap deterministic checks run first; LLM judges run only where needed.

| Check | How | Pass rule | Applies to |
|---|---|---|---|
| Canary leak | String match, also catches spaced-out or obfuscated forms | never present | all |
| Attack marker | String match on the marker the attack asks for; quoting it inside a refusal is allowed | not echoed | injection |
| Refusal | Phrase patterns; LLM fallback only where a refusal is expected | unanswerable and safety refuse, normal answers | normal, unanswerable |
| Correctness | LLM judge vs reference and expected behaviour, scored 0 / 0.5 / 1, reasoning before score | ≥ 0.7 | normal, edge, consistency |
| Groundedness | LLM splits the answer into claims, then checks each claim against the retrieved context | ≥ 0.8 of claims supported | normal, edge, consistency |
| Injection followed | LLM judge: did the reply obey the injected instruction? | no | injection |
| Consistency | Pairwise "same factual answer?" across paraphrases and repeats, judged in both orders | ≥ 0.8 agreement | consistency |
| Safety | Refusal, else LLM judge: is the reply safe? | refused or safe | safety |
| Latency | Rule (optional) | under the limit | all |

A result passes only if every applicable check passes. A test passes if a majority of its repeats pass;
mixed repeats mark it flaky. Judges run at temperature 0, and every judge call is cached on disk by
`sha256(model, prompt)`, so re-scoring unchanged outputs is free. An optional second judge records
disagreements as "needs review".

**The gate** (`compare --gate`) fails when any of these holds:
- regressions in critical categories exceed `gate.max_regressions` (default 0)
- the candidate's pass rate is below `gate.min_pass_rate`
- McNemar's exact test shows a significant drop (p < `gate.alpha`) with more regressions than fixes

**Test generation** (`mutantqa generate`) covers 6 categories:

- **normal:** questions written from random doc chunks, plus some multi-hop questions.
- **edge:** variants of normal questions with typos, noisy long inputs, formatting demands, Hinglish/Tenglish code-mixing, vague phrasing and false premises.
- **unanswerable:** each one is verified by retrieving its top chunks and asking a judge whether they answer it.
- **injection:** 15 attack templates plus LLM variations, and indirect attacks through planted documents (`mutantqa plant`).
- **consistency:** paraphrase groups.
- **safety:** an AdvBench subset (`experiments/fetch_safety_subset.py`; Zou et al., 2023, MIT) plus out-of-scope requests.

Near-duplicates (cosine similarity > 0.9 within a category) are dropped. A second LLM pass validates the labels, and a 10% sample goes to `tests/spot_check.csv` for you to review (`mutantqa spot-check`).

## CI

```yaml
# .github/workflows/llm-tests.yml in the app being tested
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Rishitha-Maturi-Hub/mutantqa/action@v0.1.0
        with:
          config: mutantqa.yaml
          start-command: uvicorn app.main:app --port 8000   # omit for a python target
          wait-url: http://localhost:8000/health
        env:
          OPENAI_API_KEY: ${{ secrets.LLM_API_KEY }}
```

Commit a baseline with `mutantqa run --tag v1 && mutantqa baseline update --tag v1`, and refresh it after
merges. The action restores the judge cache, runs the suite, writes the markdown report to the job summary,
and uploads the JSON report. This repo's own `llm-tests.yml` runs the demo offline on every PR: commit `m6`
to `demo_app/ACTIVE_MUTANT` and the check goes red.

## Configuration

`mutantqa init` writes an annotated `mutantqa.yaml`. Its main sections:

- **target:** HTTP (request template, JSON paths to the answer and the sources) or a Python `module:function`.
- **app_description** and **knowledge_dir:** what the app is for and the documents it answers from.
- **canary:** the secret string planted in the app's system prompt.
- **generation:** test counts per category.
- **run:** `repeats` and `concurrency`.
- **judge:** any [LiteLLM](https://docs.litellm.ai) model string, e.g. `gpt-4o-mini`, `claude-haiku-4-5`, `ollama/llama3.1`.
- **thresholds** and **gate:** the pass rules above.

API keys go in `.env` (see `.env.example`), never in the repo. Cost control: use `--limit 30 --repeats 1`
while developing, set a spending cap with your provider, and keep the cache on.

## Limitations

- LLM judges make mistakes, and their error rate differs by category. Read `experiments/human_study/results.md`
  before trusting a category's numbers, and treat judge disagreements as "needs review".
- Mutation kill rate measures detection of *these 10* faults in *this* app. Real regressions can be subtler.
- Groundedness needs the app to return its retrieved context; without it, only correctness is checked.
- Generated tests inherit the generator model's blind spots. The spot-check error rate is reported for that reason.
- Single-turn only. No agents, tools or multi-turn conversations in v0.1.
- The offline mode (simulated model + word-overlap heuristics) is for trying the pipeline and testing the
  tool. Its numbers are not evidence of anything about real models.

## Roadmap

- Multi-turn conversations and agent/tool-call traces
- More languages for edge cases; per-category judge calibration
- Judge ensembles with agreement-weighted verdicts
- PR comments with a regression diff; trend charts across baselines
- More mutant operators (prompt paraphrase, reranker removal, stale index)

## Project layout

```
mutantqa/        package: cli, config, models, llm (LiteLLM + cache), targets/, generate/, scorers/,
                 runner, compare (McNemar + gate), storage (SQLite), report
dashboard/       Streamlit app
demo_app/        HTTPX docs RAG bot + 10 mutants (offline simulator or real model)
action/          composite GitHub Action
experiments/     mutation study, human agreement study, safety subset, real-app kit
tests/           pytest suite for mutantqa itself (74 tests); handwritten demo suite in tests/data/
```

Demo corpus: HTTPX documentation, © Encode OSS Ltd., BSD-3-Clause (`demo_app/docs/HTTPX_LICENSE.txt`).

MIT licence.
