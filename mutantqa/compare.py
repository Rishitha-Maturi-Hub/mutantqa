"""Compare a baseline run with a candidate run and decide whether to block the change.

Statistics: each test gives a *paired* observation (pass/fail before, pass/fail after),
so the right test is McNemar's, which looks only at the discordant pairs:
    b = tests that went pass -> fail,  c = tests that went fail -> pass.
Under "nothing changed", b and c are equally likely, so b ~ Binomial(b + c, 0.5).
We use the exact (binomial) form, which is correct for the small counts typical here;
the chi-square approximation is unreliable when b + c < 25. This matches
statsmodels.stats.contingency_tables.mcnemar(table, exact=True).
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel
from scipy.stats import binomtest

from .config import GateConfig
from .models import CATEGORIES, RunBundle, TestResult
from .scorers import verdicts


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


class TestChange(BaseModel):
    __test__ = False

    test_id: str
    category: str
    input: str
    status: str  # fixed | regressed | still_passing | still_failing | added | removed
    flaky: bool = False  # candidate repeats disagreed
    baseline_flaky: bool = False
    old_output: str | None = None
    new_output: str | None = None
    failed_checks: dict[str, str] = {}  # check name -> judge reasoning, for the candidate

    @property
    def label(self) -> str:
        return "flaky" if self.flaky and self.status.startswith("still") else self.status


class CategoryStat(BaseModel):
    category: str
    n: int
    baseline_pass_rate: float | None
    candidate_pass_rate: float | None


class GateResult(BaseModel):
    passed: bool
    reasons: list[str]


class Comparison(BaseModel):
    baseline_id: str
    candidate_id: str
    n_paired: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    regressions: int
    fixes: int
    critical_regressions: int
    flaky: int
    p_value: float
    significant_drop: bool
    categories: list[CategoryStat]
    changes: list[TestChange]
    gate: GateResult

    def by_status(self, status: str) -> list[TestChange]:
        return [c for c in self.changes if c.status == status]


def _first_output(results: list[TestResult], test_id: str) -> TestResult | None:
    rs = [r for r in results if r.test_id == test_id]
    return min(rs, key=lambda r: r.repeat) if rs else None


def _failing_reasons(results: list[TestResult], test_id: str) -> dict[str, str]:
    """Reasons from the first failing repeat, restricted to checks that failed there."""
    for r in sorted((r for r in results if r.test_id == test_id), key=lambda r: r.repeat):
        if r.passed is False:
            return {k: r.reasons.get(k, "") for k in r.failed_checks} or dict(r.reasons)
    return {}


def compare_runs(baseline: RunBundle, candidate: RunBundle, gate: GateConfig) -> Comparison:
    vb = verdicts(baseline.tests, baseline.results)
    vc = verdicts(candidate.tests, candidate.results)
    tests = {t.id: t for t in baseline.tests} | {t.id: t for t in candidate.tests}

    changes: list[TestChange] = []
    for tid, t in tests.items():
        b, c = vb.get(tid), vc.get(tid)
        if b and c:
            status = {(True, True): "still_passing", (False, False): "still_failing",
                      (True, False): "regressed", (False, True): "fixed"}[(b.passed, c.passed)]
        elif c:
            status = "added"
        elif b:
            status = "removed"
        else:
            continue
        old, new = _first_output(baseline.results, tid), _first_output(candidate.results, tid)
        changes.append(TestChange(
            test_id=tid, category=t.category, input=t.input, status=status,
            flaky=bool(c and c.flaky), baseline_flaky=bool(b and b.flaky),
            old_output=old.output if old else None, new_output=new.output if new else None,
            failed_checks=_failing_reasons(candidate.results, tid) if c and not c.passed else {},
        ))

    paired = [ch for ch in changes if ch.status in ("still_passing", "still_failing", "regressed", "fixed")]
    regressions = [ch for ch in paired if ch.status == "regressed"]
    fixes = [ch for ch in paired if ch.status == "fixed"]
    crit = [ch for ch in regressions if ch.category in gate.critical_categories]
    p = mcnemar_exact(len(regressions), len(fixes))

    def rate(vs: dict, ids) -> float | None:
        vals = [vs[i].passed for i in ids if i in vs]
        return sum(vals) / len(vals) if vals else None

    cats = []
    for cat in CATEGORIES:
        ids = [tid for tid, t in tests.items() if t.category == cat]
        if ids:
            cats.append(CategoryStat(category=cat, n=len(ids), baseline_pass_rate=rate(vb, ids),
                                     candidate_pass_rate=rate(vc, ids)))

    base_rate = rate(vb, list(vb)) or 0.0
    cand_rate = rate(vc, list(vc)) or 0.0
    significant_drop = p < gate.alpha and len(regressions) > len(fixes)

    reasons = []
    if len(crit) > gate.max_regressions:
        reasons.append(f"{len(crit)} regression(s) in critical categories "
                       f"({', '.join(gate.critical_categories)}); allowed: {gate.max_regressions}.")
    if gate.max_total_regressions is not None and len(regressions) > gate.max_total_regressions:
        reasons.append(f"{len(regressions)} total regressions; allowed: {gate.max_total_regressions}.")
    if cand_rate < gate.min_pass_rate:
        reasons.append(f"Pass rate {cand_rate:.1%} is below the minimum {gate.min_pass_rate:.0%}.")
    if significant_drop:
        reasons.append(f"Statistically significant drop: {len(regressions)} regressions vs {len(fixes)} fixes "
                       f"(McNemar exact {'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}'}, alpha {gate.alpha}).")

    return Comparison(
        baseline_id=baseline.run.id, candidate_id=candidate.run.id, n_paired=len(paired),
        baseline_pass_rate=base_rate, candidate_pass_rate=cand_rate, regressions=len(regressions),
        fixes=len(fixes), critical_regressions=len(crit), flaky=sum(1 for v in vc.values() if v.flaky),
        p_value=p, significant_drop=significant_drop, categories=cats, changes=changes,
        gate=GateResult(passed=not reasons, reasons=reasons),
    )


# ------------------------------------------------------------------ single-run summary


class RunSummary(BaseModel):
    run_id: str
    n_tests: int
    pass_rate: float
    flaky_rate: float
    per_category: dict[str, tuple[int, int]]  # category -> (passed, total)
    latency_p50_ms: float
    latency_p95_ms: float
    app_cost_usd: float
    errors: int


def summarize(bundle: RunBundle) -> RunSummary:
    vs = verdicts(bundle.tests, bundle.results)
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for v in vs.values():
        per[v.category][1] += 1
        per[v.category][0] += int(v.passed)
    lat = sorted(r.latency_ms for r in bundle.results if r.error is None) or [0.0]
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    n = len(vs) or 1
    return RunSummary(
        run_id=bundle.run.id, n_tests=len(vs), pass_rate=sum(v.passed for v in vs.values()) / n,
        flaky_rate=sum(v.flaky for v in vs.values()) / n,
        per_category={c: (per[c][0], per[c][1]) for c in CATEGORIES if c in per},
        latency_p50_ms=statistics.median(lat), latency_p95_ms=p95,
        app_cost_usd=sum(r.cost_usd for r in bundle.results), errors=sum(1 for r in bundle.results if r.error),
    )
