# Write-up outline (blog post or 4–6 page PDF)

1. **Problem.** Why prompt and model changes cause silent regressions. One concrete example from the demo.
2. **Related tools.** Same list as the README, described fairly. Be explicit about where they already
   cover what you do.
3. **System design.** Architecture diagram; generation (6 categories, verification, dedupe, spot-check
   error rate); scoring order (rules, then judges); majority vote and flakiness; McNemar gate.
4. **Experiments.**
   - Mutation study: kill rate for the generated vs handwritten suite, a per-mutant table, which
     categories caught what, and which mutants survived and why.
   - Judge–human agreement: 200 items, human–human κ (task difficulty), judge accuracy/κ/fail
     precision and recall, per category; examples of judge errors.
   - Real apps: setup, confirmed bugs, maintainer responses.
   - Cost and runtime per suite run.
5. **Limitations.** Judge error by category, mutants vs real faults, generated-test blind spots, single-turn only.
6. **Future work.** The README roadmap, prioritised by what the experiments showed.

Write every number from `experiments/results/` and `experiments/human_study/results.md`. Never round
in your favour, and report n next to every percentage.
