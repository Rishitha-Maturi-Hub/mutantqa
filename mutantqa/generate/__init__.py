"""Builds the whole suite: generate per category -> dedupe -> validate -> spot-check sample."""
from __future__ import annotations

import csv
import random
from pathlib import Path

from ..config import Config
from ..models import TestCase
from .common import build_context
from .consistency import generate_consistency
from .dedupe import dedupe
from .edge import generate_edge
from .injection import generate_injection, planted_document
from .normal import generate_normal
from .safety import generate_safety
from .unanswerable import generate_unanswerable
from .validate import validate


def generate_suite(cfg: Config, only: list[str] | None = None, log=print) -> tuple[list[TestCase], dict]:
    ctx = build_context(cfg)
    want = {c: n for c, n in cfg.generation.per_category.items() if not only or c in only}
    if ctx.offline:
        log("[note] generation model is 'offline': using templates. Set generation.model or judge.primary "
            "to a real model for publishable suites.")

    tests: list[TestCase] = []
    # Edge and consistency tests are built from normal questions, so make extra normals
    # to draw on even if the normal category itself isn't requested.
    need_sources = max(want.get("edge", 0), (want.get("consistency", 0) + 3) // 4)
    normal = generate_normal(ctx, max(want.get("normal", 0), min(need_sources, 20)) if want else 0)
    log(f"normal:       {len(normal)}")
    if "normal" in want:
        tests += normal[: want["normal"]]
    steps = [
        ("edge", lambda n: generate_edge(ctx, n, normal)),
        ("unanswerable", lambda n: generate_unanswerable(ctx, n)),
        ("injection", lambda n: generate_injection(ctx, n)),
        ("consistency", lambda n: generate_consistency(ctx, n, normal)),
        ("safety", lambda n: generate_safety(ctx, n)),
    ]
    for cat, fn in steps:
        if cat in want:
            got = fn(want[cat])
            log(f"{cat + ':':<14}{len(got)}")
            tests += got

    tests, n_dup = dedupe(tests, cfg.generation.dedupe_threshold)
    tests, rejected = validate(ctx, tests)
    stats = {
        "total": len(tests),
        "duplicates_dropped": n_dup,
        "rejected_by_validator": len(rejected),
        "generation_cost_usd": round(ctx.cost_usd, 4),
        "per_category": {c: sum(1 for t in tests if t.category == c) for c in want},
        "planted_docs": list(ctx.planted),
    }
    return tests, stats


def write_spot_check(tests: list[TestCase], path: str | Path, fraction: float = 0.1, seed: int = 0) -> int:
    """Sample ~10% of tests (at least one per category) into a CSV for manual review.

    Fill the `ok` column with 1/0; `mutantqa spot-check` then reports the error rate,
    which belongs in your README and write-up.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[TestCase]] = {}
    for t in tests:
        by_cat.setdefault(t.category, []).append(t)
    sample = []
    for ts in by_cat.values():
        k = max(1, round(len(ts) * fraction))
        sample += rng.sample(ts, min(k, len(ts)))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "input", "reference_answer", "expected_behavior", "ok", "note"])
        for t in sample:
            w.writerow([t.id, t.category, t.input, t.reference_answer or "", t.expected_behavior, "", ""])
    return len(sample)


__all__ = ["generate_suite", "write_spot_check", "planted_document"]
