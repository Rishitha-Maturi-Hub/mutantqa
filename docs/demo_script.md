# Demo video script (2–3 minutes)

Record at 1080p with a large terminal font. Rehearse once; keep it under 3 minutes.

1. **Problem (20 s).** Show the demo bot answering a docs question well. Then say: "Change one line of
   the prompt and it starts inventing answers, and nothing in a normal test suite notices."
2. **Run the suite (30 s).** `mutantqa run --tag v1`. Point at the category table, the flaky rate and the cost.
3. **Break it (30 s).** Open a PR that writes `m5` to `demo_app/ACTIVE_MUTANT` (the "say you don't
   know" line is removed). Show the GitHub check going red, then the job summary: unanswerable and safety
   drop, McNemar p, why the gate failed.
4. **Explain the failure (40 s).** In the dashboard Compare tab, open one regression: the question, the
   old "I don't know" next to the new invented answer, and the judge's reasoning.
5. **Results (30 s).** The README results table: kill rate generated vs handwritten, judge–human κ, and
   bugs found in real apps. End on the repo URL.
