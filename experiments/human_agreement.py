"""Human agreement study: how often does the automatic judge agree with people?

Workflow (about 6-8 hours of labelling for 200 items):

  1. Sample 200 (test, output) pairs from a run, stratified across categories:
       python experiments/human_agreement.py sample --config mutantqa.yaml --run v1 --n 200
     This writes, in experiments/human_study/:
       labels_rater_a.csv, labels_rater_b.csv   blind sheets (no judge scores in them)
       judge_key.csv                            the judge's verdicts; don't open it until step 4
  2. You and a friend each fill the `pass` column (1/0) of your own sheet independently,
     following experiments/labeling_guideline.md.
  3. Merge and settle disagreements together:
       python experiments/human_agreement.py merge
     Fill the `final` column of adjudication.csv for the rows marked DISAGREE (agreed rows are prefilled).
  4. Analyse:
       python experiments/human_agreement.py analyze
     -> human_study/results.md with Cohen's kappa (human-human and judge-human), accuracy, and
        precision/recall for *detecting failures*, overall and per category.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("experiments/human_study")
SHEET_COLS = ["item_id", "category", "input", "expected_behavior", "reference_answer", "output",
              "context_excerpt", "pass", "note"]


def sample(args):
    from mutantqa.config import load_config
    from mutantqa.storage import Storage, resolve_run

    cfg = load_config(args.config)
    bundle = resolve_run(args.run, Storage(cfg.path(cfg.db_path)), cfg.path(cfg.baseline_path))
    tests = {t.id: t for t in bundle.tests}
    by_cat: dict[str, list] = defaultdict(list)
    for r in bundle.results:
        if r.repeat == 0 and r.passed is not None and r.error is None and r.test_id in tests:
            by_cat[tests[r.test_id].category].append(r)
    rng = random.Random(args.seed)
    total = sum(len(v) for v in by_cat.values())
    if total < args.n:
        print(f"Only {total} scored outputs available; sampling all of them.")
    # Proportional allocation with a floor, so small categories still get enough items to say something.
    floor = min(15, args.n // max(1, len(by_cat)))
    quota = {c: min(len(rs), max(floor, round(args.n * len(rs) / max(1, total)))) for c, rs in by_cat.items()}
    while sum(quota.values()) > args.n:
        biggest = max(quota, key=quota.get)
        quota[biggest] -= 1
    picked = [r for c, rs in by_cat.items() for r in rng.sample(rs, quota[c])]
    rng.shuffle(picked)  # don't let category order prime the raters

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in picked:
        t = tests[r.test_id]
        rows.append({"item_id": f"{r.test_id}#{r.repeat}", "category": t.category, "input": t.input,
                     "expected_behavior": t.expected_behavior, "reference_answer": t.reference_answer or "",
                     "output": r.output, "context_excerpt": " ||| ".join(c[:400] for c in r.retrieved_context[:3]),
                     "pass": "", "note": ""})
    for rater in ("a", "b"):
        with (OUT / f"labels_rater_{rater}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLS)
            w.writeheader()
            w.writerows(rows)
    with (OUT / "judge_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "category", "judge_pass", "failed_checks"])
        for r in picked:
            w.writerow([f"{r.test_id}#{r.repeat}", tests[r.test_id].category, int(bool(r.passed)),
                        ";".join(r.failed_checks)])
    print(f"Sampled {len(picked)} items: " + ", ".join(f"{c} {q}" for c, q in quota.items()))
    print(f"Sheets in {OUT}/. Label independently, then run `merge`.")


def _read_labels(path: Path) -> dict[str, int]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("pass") or "").strip()
            if v in ("0", "1"):
                out[row["item_id"]] = int(v)
    return out


def merge(args):
    a, b = _read_labels(OUT / "labels_rater_a.csv"), _read_labels(OUT / "labels_rater_b.csv")
    with (OUT / "labels_rater_a.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r["item_id"] for r in rows if r["item_id"] not in a or r["item_id"] not in b]
    if missing:
        print(f"{len(missing)} items are unlabelled by at least one rater, e.g. {missing[:3]}. Finish labelling first.")
        sys.exit(1)
    n_dis = 0
    with (OUT / "adjudication.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "category", "rater_a", "rater_b", "status", "final", "input", "output"])
        for r in rows:
            i = r["item_id"]
            agree = a[i] == b[i]
            n_dis += not agree
            w.writerow([i, r["category"], a[i], b[i], "agree" if agree else "DISAGREE", a[i] if agree else "",
                        r["input"], r["output"]])
    print(f"{n_dis} disagreements of {len(rows)}. Fill `final` for DISAGREE rows in {OUT / 'adjudication.csv'}, "
          f"then run `analyze`.")


def metrics(human: list[int], judge: list[int]) -> dict:
    """Agreement stats. 'Positive' = FAIL, because the job of the judge is to catch failures."""
    from sklearn.metrics import cohen_kappa_score

    n = len(human)
    tp = sum(1 for h, j in zip(human, judge) if h == 0 and j == 0)
    fp = sum(1 for h, j in zip(human, judge) if h == 1 and j == 0)
    fn = sum(1 for h, j in zip(human, judge) if h == 0 and j == 1)
    acc = sum(h == j for h, j in zip(human, judge)) / n if n else float("nan")
    kappa = cohen_kappa_score(human, judge) if len(set(human) | set(judge)) > 1 else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"n": n, "accuracy": acc, "kappa": kappa, "fail_precision": prec, "fail_recall": rec,
            "human_fail_rate": sum(1 for h in human if h == 0) / n if n else float("nan"),
            "tp": tp, "fp": fp, "fn": fn}


def analyze(args):
    from sklearn.metrics import cohen_kappa_score

    a, b = _read_labels(OUT / "labels_rater_a.csv"), _read_labels(OUT / "labels_rater_b.csv")
    final, cats = {}, {}
    with (OUT / "adjudication.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cats[row["item_id"]] = row["category"]
            if row["final"].strip() in ("0", "1"):
                final[row["item_id"]] = int(row["final"])
    judge = {}
    with (OUT / "judge_key.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            judge[row["item_id"]] = int(row["judge_pass"])
    ids = [i for i in cats if i in final and i in judge and i in a and i in b]
    if len(ids) < len(cats):
        print(f"Warning: {len(cats) - len(ids)} items lack a final label and are excluded.")

    def fmt(x, pct=False):
        return "–" if x != x else (f"{x:.0%}" if pct else f"{x:.2f}")  # x != x is NaN

    hh = cohen_kappa_score([a[i] for i in ids], [b[i] for i in ids])
    rows = [("all", metrics([final[i] for i in ids], [judge[i] for i in ids]),
             hh, sum(a[i] == b[i] for i in ids) / len(ids))]
    for c in sorted(set(cats[i] for i in ids)):
        sub = [i for i in ids if cats[i] == c]
        ha, hb = [a[i] for i in sub], [b[i] for i in sub]
        k = cohen_kappa_score(ha, hb) if len(set(ha) | set(hb)) > 1 else float("nan")
        rows.append((c, metrics([final[i] for i in sub], [judge[i] for i in sub]), k,
                     sum(x == y for x, y in zip(ha, hb)) / len(sub)))

    lines = ["# Judge vs human agreement", "",
             f"{len(ids)} items labelled independently by two people, disagreements settled together. "
             "Precision/recall treat **fail** as the positive class (the judge's job is to catch failures).", "",
             "| Slice | n | Human–human κ | Human–human agree | Judge–human κ | Judge accuracy | Fail precision | Fail recall | Human fail rate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m, k, agree in rows:
        lines.append(f"| {name} | {m['n']} | {fmt(k)} | {fmt(agree, True)} | {fmt(m['kappa'])} | "
                     f"{fmt(m['accuracy'], True)} | {fmt(m['fail_precision'], True)} | {fmt(m['fail_recall'], True)} | "
                     f"{fmt(m['human_fail_rate'], True)} |")
    lines += ["", "κ guide (Landis & Koch): <0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, "
              ">0.80 almost perfect. κ is unstable for small slices or when one label dominates; report n alongside it.",
              "", "## Judge errors to read", ""]
    misses = [i for i in ids if final[i] == 0 and judge[i] == 1]
    false_alarms = [i for i in ids if final[i] == 1 and judge[i] == 0]
    lines.append(f"- Missed failures (judge passed, humans failed): {len(misses)}: " + ", ".join(misses[:15]))
    lines.append(f"- False alarms (judge failed, humans passed): {len(false_alarms)}: " + ", ".join(false_alarms[:15]))
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--config", default="mutantqa.yaml")
    s.add_argument("--run", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=13)
    sub.add_parser("merge")
    sub.add_parser("analyze")
    args = ap.parse_args()
    {"sample": sample, "merge": merge, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
