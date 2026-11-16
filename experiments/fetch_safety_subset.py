"""Download a random subset of AdvBench harmful behaviours for the safety category.

Source: Zou, Wang, Kolter, Fredrikson (2023), "Universal and Transferable Adversarial
Attacks on Aligned Language Models", https://github.com/llm-attacks/llm-attacks (MIT).
Cite it in your README and write-up.

    python experiments/fetch_safety_subset.py --n 30
writes tests/data/advbench_subset.csv (column: goal), which generation.safety_prompts_file
points at. We use the *prompts* only, to check that the app refuses.
"""
import argparse
import csv
import io
import random
from pathlib import Path

import httpx

URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="tests/data/advbench_subset.csv")
    args = ap.parse_args()
    text = httpx.get(URL, timeout=30, follow_redirects=True).raise_for_status().text
    goals = [row["goal"] for row in csv.DictReader(io.StringIO(text)) if row.get("goal")]
    subset = random.Random(args.seed).sample(goals, min(args.n, len(goals)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["goal"])
        w.writerows([g] for g in subset)
    print(f"Wrote {len(subset)} of {len(goals)} AdvBench prompts to {out} (seed {args.seed}).")


if __name__ == "__main__":
    main()
