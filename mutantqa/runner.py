"""Runs every test N times against the target, concurrently, then scores the outputs.

Concurrency is bounded by an asyncio.Semaphore: enough to be fast, not so much that we
trip the app's (or the provider's) rate limits. Raw outputs are saved as they arrive,
so a crash at test 110 of 120 doesn't throw away the first 109 paid calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import llm
from .config import Config
from .models import Run, RunBundle, TestCase, TestResult
from .scorers import score_results
from .storage import Storage
from .targets import Target, make_target


def load_suite(path: str | Path) -> list[TestCase]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No test suite at {p}. Run `mutantqa generate` first.")
    tests = [TestCase.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [t.id for t in tests]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate test ids in {p}")
    return tests


def save_suite(tests: list[TestCase], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(t.model_dump_json(exclude_defaults=False) + "\n" for t in tests), encoding="utf-8")


def suite_version(tests: list[TestCase]) -> str:
    blob = json.dumps([t.model_dump(mode="json") for t in tests], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def git_commit(cwd: str | Path = ".") -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


async def collect_outputs(
    target: Target,
    tests: list[TestCase],
    run_id: str,
    repeats: int,
    concurrency: int,
    storage: Storage | None = None,
    show_progress: bool = True,
) -> list[TestResult]:
    sem = asyncio.Semaphore(concurrency)
    results: list[TestResult] = []
    pending: list[TestResult] = []

    async def one(test: TestCase, rep: int) -> TestResult:
        async with sem:
            try:
                resp = await target.ask(test.input)
                return TestResult(run_id=run_id, test_id=test.id, repeat=rep, output=resp.output,
                                  retrieved_context=resp.context, latency_ms=resp.latency_ms, cost_usd=resp.cost_usd)
            except Exception as exc:  # record, don't crash the whole run
                return TestResult(run_id=run_id, test_id=test.id, repeat=rep, output="", latency_ms=0.0,
                                  error=f"{type(exc).__name__}: {exc}")

    jobs = [one(t, r) for t in tests for r in range(repeats)]
    progress = Progress(TextColumn("[bold]running[/]"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                        disable=not show_progress, transient=True)
    with progress:
        task = progress.add_task("run", total=len(jobs))
        for fut in asyncio.as_completed(jobs):
            res = await fut
            results.append(res)
            pending.append(res)
            progress.advance(task)
            if storage and len(pending) >= 20:
                storage.save_results(pending)
                pending = []
    if storage and pending:
        storage.save_results(pending)
    await target.aclose()
    results.sort(key=lambda r: (r.test_id, r.repeat))
    return results


def execute_run(
    cfg: Config,
    tag: str,
    tests: list[TestCase] | None = None,
    storage: Storage | None = None,
    target: Target | None = None,
    app_version: str | None = None,
    app_config_extra: str = "",
    repeats: int | None = None,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[RunBundle, dict]:
    """Run + score + save. Returns the bundle and a small stats dict (cost, errors)."""
    llm.configure(cfg.path(cfg.cache_dir))
    tests = tests if tests is not None else load_suite(cfg.path(cfg.suite_path))
    if limit:
        tests = tests[:limit]
    storage = storage or Storage(cfg.path(cfg.db_path))
    run = Run(id=tag, app_version=app_version or git_commit(cfg.base_dir),
              config_hash=cfg.config_hash(app_config_extra), suite_version=suite_version(tests))
    storage.start_run(run, tests)
    target = target or make_target(cfg.target)

    results = asyncio.run(collect_outputs(target, tests, tag, repeats or cfg.run.repeats, cfg.run.concurrency,
                                          storage, show_progress))
    judge_cost = score_results(tests, results, cfg)
    storage.save_results(results)

    stats = {
        "app_cost_usd": sum(r.cost_usd for r in results),
        "judge_cost_usd": judge_cost,
        "errors": sum(1 for r in results if r.error),
        "calls": len(results),
    }
    return RunBundle(run=run, tests=tests, results=results), stats
