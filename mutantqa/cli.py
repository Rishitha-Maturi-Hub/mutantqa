"""Command-line interface.

    mutantqa init
    mutantqa generate
    mutantqa run --tag v1
    mutantqa compare v1 v2 --gate --markdown report.md     # exit code 1 if the gate fails
    mutantqa report --tag v2
    mutantqa baseline update --tag v1

The non-zero exit code from `compare --gate` is what makes a CI step fail.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__, llm
from .config import DEFAULT_CONFIG_PATH, STARTER_CONFIG, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="QA for LLM apps.")
baseline_app = typer.Typer(no_args_is_help=True, help="Manage the committed baseline run.")
app.add_typer(baseline_app, name="baseline")
console = Console()

ConfigOpt = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to mutantqa.yaml")


def _load(config: str):
    try:
        cfg = load_config(config)
    except Exception as exc:
        console.print(f"[red]Config error:[/] {exc}")
        raise typer.Exit(2) from None
    llm.configure(cfg.path(cfg.cache_dir))
    return cfg


@app.command()
def version():
    """Print the version."""
    console.print(__version__)


@app.command()
def init(path: str = typer.Option(DEFAULT_CONFIG_PATH, help="Where to write the config"),
         force: bool = typer.Option(False, help="Overwrite an existing config")):
    """Write a starter mutantqa.yaml."""
    p = Path(path)
    if p.exists() and not force:
        console.print(f"[yellow]{p} already exists.[/] Use --force to overwrite.")
        raise typer.Exit(1)
    p.write_text(STARTER_CONFIG, encoding="utf-8")
    gi = Path(".gitignore")
    lines = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    add = [x for x in (".env", ".mutantqa/cache/", ".mutantqa/*.db") if x not in lines]
    if add:
        gi.write_text("\n".join(lines + add) + "\n", encoding="utf-8")
    console.print(f"Wrote {p}. Edit the target section, then run [bold]mutantqa generate[/].")


@app.command()
def generate(
    config: str = ConfigOpt,
    only: Optional[list[str]] = typer.Option(None, "--only", help="Only these categories (repeatable)"),
    append: bool = typer.Option(False, help="Append to the existing suite instead of replacing it"),
    spot_check: bool = typer.Option(True, help="Also write a 10%% sample for manual review"),
):
    """Generate the test suite (tests/suite.jsonl)."""
    from .generate import generate_suite, write_spot_check
    from .runner import load_suite, save_suite

    cfg = _load(config)
    tests, stats = generate_suite(cfg, only=only, log=console.print)
    path = cfg.path(cfg.suite_path)
    if append and path.exists():
        tests = load_suite(path) + tests
    save_suite(tests, path)
    console.print(f"\n[green]Wrote {len(tests)} tests[/] to {path}")
    console.print(json.dumps(stats, indent=1))
    if spot_check:
        sc = path.with_name("spot_check.csv")
        n = write_spot_check(tests, sc)
        console.print(f"Spot-check sample of {n} tests in {sc}. Fill the 'ok' column, then run "
                      f"[bold]mutantqa spot-check {sc}[/].")


@app.command("spot-check")
def spot_check_cmd(path: str):
    """Report the error rate from a filled-in spot-check CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("ok", "").strip() in ("0", "1")]
    if not rows:
        console.print("No rows with ok=0/1 yet.")
        raise typer.Exit(1)
    bad = [r for r in rows if r["ok"].strip() == "0"]
    console.print(f"Reviewed {len(rows)} tests: {len(bad)} invalid ({len(bad) / len(rows):.1%} error rate).")
    for r in bad:
        console.print(f"  [{r['category']}] {r['id']}: {r.get('note', '')}")


@app.command()
def run(
    tag: str = typer.Option(..., "--tag", "-t", help="Name for this run, e.g. v1 or pr-42"),
    config: str = ConfigOpt,
    repeats: Optional[int] = typer.Option(None, help="Override run.repeats"),
    limit: Optional[int] = typer.Option(None, help="Only the first N tests (cheap dev loop)"),
    app_version: Optional[str] = typer.Option(None, help="Defaults to the current git commit"),
    app_config: Optional[str] = typer.Option(None, help="File with the app's prompt/model settings, hashed into the run"),
    suite: Optional[str] = typer.Option(None, help="Use this suite file instead of suite_path"),
):
    """Run the suite against the target, score it and save the results."""
    from .compare import summarize
    from .report import print_summary
    from .runner import execute_run, load_suite

    cfg = _load(config)
    extra = Path(app_config).read_text(encoding="utf-8") if app_config else ""
    llm.reset_spend()
    tests = load_suite(suite) if suite else None
    bundle, stats = execute_run(cfg, tag, tests=tests, app_version=app_version, app_config_extra=extra,
                                repeats=repeats, limit=limit)
    print_summary(summarize(bundle), console)
    s = llm.spend()
    console.print(f"Judge cost: ${stats['judge_cost_usd']:.4f} ({s['calls']} LLM calls, {s['cache_hits']} cache hits)")
    if stats["errors"]:
        console.print(f"[yellow]{stats['errors']} of {stats['calls']} target calls failed.[/] "
                      "Check that the app is running and the target config is right.")
    if stats["errors"] == stats["calls"]:
        raise typer.Exit(2) from None


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline run id ('baseline' = the committed baseline file)"),
    candidate: str = typer.Argument(..., help="Candidate run id"),
    config: str = ConfigOpt,
    gate: bool = typer.Option(False, "--gate", help="Exit 1 if the gate fails"),
    markdown: Optional[str] = typer.Option(None, "--markdown", help="Write a markdown report here"),
    json_out: Optional[str] = typer.Option(None, "--json", help="Write a JSON report here"),
):
    """Compare two runs: regressions, fixes, flaky tests, McNemar's test, gate decision."""
    from .compare import compare_runs
    from .report import print_comparison, write_json, write_markdown
    from .storage import Storage, resolve_run

    cfg = _load(config)
    st = Storage(cfg.path(cfg.db_path))
    try:
        b = resolve_run(baseline, st, cfg.path(cfg.baseline_path))
        c = resolve_run(candidate, st, cfg.path(cfg.baseline_path))
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from None
    result = compare_runs(b, c, cfg.gate)
    print_comparison(result, console)
    if markdown:
        write_markdown(result, markdown)
    if json_out:
        write_json(result, json_out)
    if gate and not result.gate.passed:
        raise typer.Exit(1)


@app.command()
def report(tag: str = typer.Option(..., "--tag", "-t"), config: str = ConfigOpt):
    """Terminal summary of one run."""
    from .compare import summarize
    from .report import print_summary
    from .storage import Storage, resolve_run

    cfg = _load(config)
    st = Storage(cfg.path(cfg.db_path))
    try:
        bundle = resolve_run(tag, st, cfg.path(cfg.baseline_path))
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from None
    print_summary(summarize(bundle), console)


@app.command()
def runs(config: str = ConfigOpt):
    """List stored runs."""
    from .storage import Storage

    cfg = _load(config)
    for r in Storage(cfg.path(cfg.db_path)).list_runs():
        console.print(f"{r.id:<16} {r.created_at:%Y-%m-%d %H:%M}  app={r.app_version}  "
                      f"config={r.config_hash}  suite={r.suite_version}")


@baseline_app.command("update")
def baseline_update(tag: str = typer.Option(..., "--tag", "-t", help="Run to promote to baseline"),
                    config: str = ConfigOpt,
                    keep_context: bool = typer.Option(False, help="Keep retrieved chunks (much bigger file)")):
    """Write a run to the baseline file (commit it so CI can compare against it)."""
    from .storage import Storage, write_bundle

    cfg = _load(config)
    st = Storage(cfg.path(cfg.db_path))
    if not st.has_run(tag):
        console.print(f"[red]No run '{tag}'.[/]")
        raise typer.Exit(2) from None
    bundle = st.load_bundle(tag)
    bundle.run.id = "baseline"
    for r in bundle.results:
        r.run_id = "baseline"
        if not keep_context:
            r.retrieved_context = []  # comparisons don't need it; keeps the committed file small
    write_bundle(bundle, cfg.path(cfg.baseline_path))
    console.print(f"Baseline updated from run '{tag}' -> {cfg.path(cfg.baseline_path)}")


@app.command()
def plant(
    topic: str = typer.Option(..., help="What the planted page is 'about', e.g. 'connection pool tuning'"),
    out: str = typer.Option(..., help="Where to write it, inside YOUR OWN copy of the corpus"),
    marker: Optional[str] = typer.Option(None, help="Marker word; random if omitted"),
):
    """Write a document containing a hidden instruction, for indirect injection tests.

    The file must go directly inside knowledge_dir; it is registered in that folder's
    .mutantqa-planted.json manifest."""
    from .generate import planted_document
    from .generate.common import PLANTED_MANIFEST

    text, m = planted_document(topic, marker)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text, encoding="utf-8")
    manifest_path = outp.parent / PLANTED_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[outp.name] = {"marker": m, "topic": topic}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    console.print(f"Wrote {out} with marker {m}. Re-index the app, then run `mutantqa generate --only injection --append`.")


if __name__ == "__main__":
    app()
