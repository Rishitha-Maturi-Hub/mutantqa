# What's built and what's left

Built and tested (74 pytest tests: offline end-to-end, plus every LLM judge and generator path against a scripted fake LLM):
Steps 1–6 (demo app + 10 mutants, adapters, runner, storage, generator, scorers, compare/McNemar
gate, CLI, GitHub Action, dashboard, packaging), plus the tooling for Steps 7–8 (mutation study,
human-agreement study, AdvBench fetcher, real-app kit, write-up outline).

Everything below needs a real API key or human time, so it's yours to do.

## Nov 14–20: make it real
- [ ] Rename if you prefer (`mutantqa` was free on PyPI and GitHub on Nov 13):
      `grep -rl mutantqa . | xargs sed -i 's/mutantqa/NEWNAME/g'` and rename the `mutantqa/` folder.
- [ ] Push to GitHub, put your name in LICENSE and pyproject.toml, and replace YOUR_USER in the README.
- [ ] `cp .env.example .env`, add a key, and set a monthly spending limit with the provider.
- [ ] `export DEMO_MODEL=gpt-4o-mini` (or another model), then `mutantqa run --tag hw --suite tests/data/handwritten_suite.jsonl`
      on the real config. Read the outputs; this is the "20 handwritten tests end to end" checkpoint.
- [ ] Read every module until you can explain it. Interviewers will ask.

## Nov 21–27: real suite
- [ ] `python experiments/fetch_safety_subset.py`, then `mutantqa generate`.
- [ ] Fill `tests/spot_check.csv` and run `mutantqa spot-check tests/spot_check.csv`. Record the error rate.
- [ ] Extend `tests/data/handwritten_suite.jsonl` to 40 normal questions for the baseline comparison.

## Nov 28–Dec 4: headline number
- [ ] `python experiments/mutation_study.py --baseline-suite tests/data/handwritten_suite.jsonl`
      (first with `--repeats 1 --mutants m1,m5` to check the cost).
- [ ] If v1 is below `min_pass_rate`, fix the app or the thresholds before trusting the "gate" column.

## Dec 5–11: ship
- [ ] `mutantqa run --tag v1 && mutantqa baseline update --tag v1`, then commit `.mutantqa/baseline.json`.
- [ ] Add the `LLM_API_KEY` secret. Open a demo PR that writes `m6` to `demo_app/ACTIVE_MUTANT`,
      and screenshot the red check to `docs/ci_failure.png`.
- [ ] Record `docs/dashboard.gif` and the demo video (`docs/demo_script.md`).
- [ ] Publish: `python -m build && twine upload -r testpypi dist/*`, test the install, then PyPI. Tag v0.1.0.

## Dec 12–Jan 10: evidence
- [ ] 3 real apps (`experiments/real_apps/README.md`). Verify bugs by hand and report them politely.
- [ ] Human study: `python experiments/human_agreement.py sample --run v1`, label with a friend, then
      run `merge` and `analyze`.
- [ ] Fill the README results table.

## Known gaps worth knowing before interviews
- Offline mode is a simulator for testing the pipeline. Never quote its numbers as results.
- The dashboard was smoke-tested with Streamlit's AppTest, but no one has looked at it in a browser yet.
  Check the layout locally.
- `mutation_study.py` needs a python target (it switches mutants through an env var in-process).
- Retrieval in the demo uses TF-IDF by default. `DEMO_EMBEDDINGS=st` (with `pip install mutantqa[embeddings]`)
  switches to all-MiniLM-L6-v2, which the guide suggests. Chroma was not used; a numpy index is simpler
  to explain at this size.
