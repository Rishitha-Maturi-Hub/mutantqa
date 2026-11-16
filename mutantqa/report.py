"""Reporters: terminal (rich), markdown (CI job summary / PR comment), JSON (dashboard)."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .compare import Comparison, RunSummary


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def fmt_p(p: float) -> str:
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = (b - a) * 100
    return "±0" if abs(d) < 0.05 else f"{d:+.0f} pts"


def _clip(s: str | None, n: int = 300) -> str:
    if s is None:
        return "–"
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ------------------------------------------------------------------ terminal


def print_summary(s: RunSummary, console: Console | None = None) -> None:
    console = console or Console()
    t = Table(title=f"Run {s.run_id}: {s.pass_rate:.0%} of {s.n_tests} tests passed")
    t.add_column("Category")
    t.add_column("Passed", justify="right")
    t.add_column("Rate", justify="right")
    for cat, (p, n) in s.per_category.items():
        t.add_row(cat, f"{p}/{n}", _pct(p / n if n else None))
    console.print(t)
    console.print(
        f"Flaky: {s.flaky_rate:.0%}   Latency p50/p95: {s.latency_p50_ms:.0f}/{s.latency_p95_ms:.0f} ms   "
        f"App cost: ${s.app_cost_usd:.4f}   Errors: {s.errors}"
    )


def print_comparison(c: Comparison, console: Console | None = None, max_items: int = 10) -> None:
    console = console or Console()
    verdict = "[bold green]PASS[/]" if c.gate.passed else "[bold red]FAIL[/]"
    console.print(f"\nGate: {verdict}   {c.baseline_id} → {c.candidate_id}   "
                  f"pass rate {c.baseline_pass_rate:.0%} → {c.candidate_pass_rate:.0%}   "
                  f"regressions {c.regressions}, fixes {c.fixes}, flaky {c.flaky}, McNemar {fmt_p(c.p_value)}")
    for r in c.gate.reasons:
        console.print(f"  [red]•[/] {r}")
    t = Table(show_edge=False)
    for col in ("Category", "n", c.baseline_id, c.candidate_id, "Change"):
        t.add_column(col, justify="left" if col == "Category" else "right")
    for cs in c.categories:
        t.add_row(cs.category, str(cs.n), _pct(cs.baseline_pass_rate), _pct(cs.candidate_pass_rate),
                  _delta(cs.baseline_pass_rate, cs.candidate_pass_rate))
    console.print(t)
    regs = c.by_status("regressed")
    if regs:
        console.print(f"\n[bold]Regressions[/] (showing {min(len(regs), max_items)} of {len(regs)})")
        for ch in regs[:max_items]:
            console.print(f"[red]✗[/] [{ch.category}] {ch.test_id}: {_clip(ch.input, 120)}")
            for name, why in list(ch.failed_checks.items())[:2]:
                console.print(f"    {name}: {_clip(why, 160)}")


# ------------------------------------------------------------------ markdown


def comparison_markdown(c: Comparison, max_items: int = 15) -> str:
    icon = "✅" if c.gate.passed else "❌"
    lines = [
        f"## {icon} mutantqa: {'passed' if c.gate.passed else 'failed'} ({c.baseline_id} → {c.candidate_id})",
        "",
        f"Pass rate **{c.baseline_pass_rate:.1%} → {c.candidate_pass_rate:.1%}** across {c.n_paired} paired tests. "
        f"{c.regressions} regressed, {c.fixes} fixed, {c.flaky} flaky. McNemar exact {fmt_p(c.p_value)}.",
        "",
    ]
    if c.gate.reasons:
        lines += ["**Why the gate failed**", ""] + [f"- {r}" for r in c.gate.reasons] + [""]
    lines += ["| Category | n | " + c.baseline_id + " | " + c.candidate_id + " | Change |",
              "|---|---:|---:|---:|---:|"]
    for cs in c.categories:
        lines.append(f"| {cs.category} | {cs.n} | {_pct(cs.baseline_pass_rate)} | {_pct(cs.candidate_pass_rate)} "
                     f"| {_delta(cs.baseline_pass_rate, cs.candidate_pass_rate)} |")
    regs = c.by_status("regressed")
    if regs:
        lines += ["", f"### Regressions ({len(regs)})", ""]
        for ch in regs[:max_items]:
            lines += [
                "<details>",
                f"<summary><b>[{ch.category}]</b> {_clip(ch.input, 110)}</summary>",
                "",
                f"**Test** `{ch.test_id}`{' (flaky)' if ch.flaky else ''}",
                "",
                f"**Before:** {_clip(ch.old_output, 500)}",
                "",
                f"**After:** {_clip(ch.new_output, 500)}",
                "",
            ]
            for name, why in ch.failed_checks.items():
                lines.append(f"- `{name}`: {_clip(why, 300)}")
            lines += ["", "</details>"]
        if len(regs) > max_items:
            lines.append(f"\n…and {len(regs) - max_items} more. See the JSON report or dashboard.")
    flaky = [ch for ch in c.changes if ch.flaky]
    if flaky:
        lines += ["", f"### Flaky tests ({len(flaky)})", ""]
        lines += [f"- [{ch.category}] `{ch.test_id}` {_clip(ch.input, 100)}" for ch in flaky[:max_items]]
    return "\n".join(lines) + "\n"


def write_markdown(c: Comparison, path: str | Path) -> None:
    Path(path).write_text(comparison_markdown(c), encoding="utf-8")


def write_json(c: Comparison, path: str | Path) -> None:
    data = c.model_dump(mode="json")
    for ch, raw in zip(c.changes, data["changes"]):
        raw["label"] = ch.label
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
