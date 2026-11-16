"""Mutation study: what fraction of deliberately broken app variants does a suite catch?

    python experiments/mutation_study.py --config mutantqa.yaml \
        --suite tests/suite.jsonl --baseline-suite tests/data/handwritten_suite.jsonl

For each suite: run the healthy app (v1) and each mutant m1..m10, compare each mutant
against v1, and record whether it was killed. Three kill definitions are reported:

  gate       the full CI gate fails (what the guide calls "killed")
  gate-rel   the gate fails on its *relative* rules only (critical regressions or a
             significant drop), ignoring the absolute pass-rate floor. If v1 itself is
             below min_pass_rate, the full gate fails for every mutant and says nothing,
             so this column is the honest one in that case.
  any-reg    at least one test regressed (the classic mutation-testing definition;
             the most sensitive, but also the most exposed to judge noise)

Requires a python target (the app reads MUTANT from the environment per call). For an
HTTP app, restart the server with MUTANT=mX between runs and use `mutantqa run`/`compare`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutantqa import llm  # noqa: E402
from mutantqa.compare import compare_runs, summarize  # noqa: E402
from mutantqa.config import load_config  # noqa: E402
from mutantqa.runner import execute_run, load_suite  # noqa: E402
from mutantqa.storage import Storage  # noqa: E402

try:
    from demo_app.rag import MUTANTS
    DESCRIPTIONS = {k: v[0] for k, v in MUTANTS.items()}
except Exception:
    DESCRIPTIONS = {}


def study(cfg, suite_name: str, tests, mutants: list[str], repeats: int | None, storage: Storage, log) -> dict:
    os.environ.pop("MUTANT", None)
    base, _ = execute_run(cfg, f"{suite_name}:v1", tests=tests, storage=storage, repeats=repeats,
                          app_version="v1", show_progress=False)
    s = summarize(base)
    log(f"[{suite_name}] v1 pass rate {s.pass_rate:.0%} ({s.n_tests} tests)")
    rows = []
    for m in mutants:
        os.environ["MUTANT"] = m
        try:
            cand, _ = execute_run(cfg, f"{suite_name}:{m}", tests=tests, storage=storage, repeats=repeats,
                                  app_version=m, show_progress=False)
        finally:
            os.environ.pop("MUTANT", None)
        c = compare_runs(base, cand, cfg.gate)
        relative = c.critical_regressions > cfg.gate.max_regressions or c.significant_drop
        cats = sorted({ch.category for ch in c.changes if ch.status == "regressed"})
        rows.append({
            "mutant": m, "description": DESCRIPTIONS.get(m, ""),
            "pass_rate": round(c.candidate_pass_rate, 4), "regressions": c.regressions, "fixes": c.fixes,
            "p_value": round(c.p_value, 5), "flaky": c.flaky,
            "killed_gate": not c.gate.passed, "killed_gate_rel": relative, "killed_any_reg": c.regressions > 0,
            "regressed_categories": cats,
        })
        log(f"  {m:<4} pass {c.candidate_pass_rate:5.0%}  reg {c.regressions:3d}  fix {c.fixes:3d}  "
            f"p={c.p_value:.4f}  gate={'KILLED' if not c.gate.passed else 'survived'}  "
            f"rel={'KILLED' if relative else 'survived'}  [{', '.join(cats)}]")
    k = lambda key: sum(r[key] for r in rows)  # noqa: E731
    return {"suite": suite_name, "n_tests": len(tests), "v1_pass_rate": round(s.pass_rate, 4),
            "v1_below_floor": s.pass_rate < cfg.gate.min_pass_rate,
            "kills": {"gate": k("killed_gate"), "gate_rel": k("killed_gate_rel"), "any_reg": k("killed_any_reg")},
            "n_mutants": len(rows), "rows": rows}


def markdown(results: list[dict], cfg, judge: str) -> str:
    out = [f"# Mutation study ({datetime.now():%Y-%m-%d})", "",
           f"Judge: `{judge}`. Repeats per test: {cfg.run.repeats}. Gate: max {cfg.gate.max_regressions} critical "
           f"regressions, min pass rate {cfg.gate.min_pass_rate:.0%}, alpha {cfg.gate.alpha}.", ""]
    if llm.is_offline(judge):
        out += ["> **Offline mode**: simulated app model and heuristic judges. These numbers test the pipeline; "
                "they are not publishable results.", ""]
    out += ["| Suite | Tests | v1 pass rate | Killed (gate) | Killed (gate-rel) | Killed (any regression) |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        n = r["n_mutants"]
        flag = " ⚠️ v1 below floor" if r["v1_below_floor"] else ""
        out.append(f"| {r['suite']} | {r['n_tests']} | {r['v1_pass_rate']:.0%}{flag} | {r['kills']['gate']}/{n} | "
                   f"{r['kills']['gate_rel']}/{n} | {r['kills']['any_reg']}/{n} |")
    for r in results:
        out += ["", f"## {r['suite']}", "", "| Mutant | What it breaks | Pass rate | Reg | Fix | McNemar p | Gate | Gate-rel | Caught in |",
                "|---|---|---:|---:|---:|---:|---|---|---|"]
        for row in r["rows"]:
            yn = lambda b: "killed" if b else "**survived**"  # noqa: E731
            out.append(f"| {row['mutant']} | {row['description']} | {row['pass_rate']:.0%} | {row['regressions']} | "
                       f"{row['fixes']} | {row['p_value']:.4f} | {yn(row['killed_gate'])} | {yn(row['killed_gate_rel'])} | "
                       f"{', '.join(row['regressed_categories']) or '–'} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="mutantqa.yaml")
    ap.add_argument("--suite", help="Generated suite (default: suite_path from config)")
    ap.add_argument("--baseline-suite", help="Handwritten comparison suite", default=None)
    ap.add_argument("--mutants", default=",".join(f"m{i}" for i in range(1, 11)))
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    llm.configure(cfg.path(cfg.cache_dir))
    if cfg.target.type != "python":
        sys.exit("mutation_study.py needs a python target; see the module docstring.")
    storage = Storage(cfg.path(cfg.db_path).with_name("mutation_study.db"))
    mutants = [m.strip() for m in args.mutants.split(",") if m.strip()]

    suites = [("generated", load_suite(args.suite or cfg.path(cfg.suite_path)))]
    if args.baseline_suite:
        suites.append(("handwritten", load_suite(args.baseline_suite)))
    results = [study(cfg, name, tests, mutants, args.repeats, storage, print) for name, tests in suites]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = "offline" if llm.is_offline(cfg.judge.primary) else datetime.now().strftime("%Y%m%d-%H%M")
    (out / f"mutation_{stamp}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    md = markdown(results, cfg, cfg.judge.primary)
    (out / f"mutation_{stamp}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved to {out}/mutation_{stamp}.(md|json). LLM spend: ${llm.spend()['usd']:.4f}")


if __name__ == "__main__":
    main()
