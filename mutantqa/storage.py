"""SQLite storage. Three tables, scores and reasons as JSON columns. Deliberately simple.

The suite is snapshotted per run (test_cases has run_id in its key) because the suite
changes over time, and a comparison must know each test's category *as it was run*.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Run, RunBundle, TestCase, TestResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    app_version TEXT,
    config_hash TEXT,
    suite_version TEXT
);
CREATE TABLE IF NOT EXISTS test_cases (
    run_id TEXT NOT NULL,
    id TEXT NOT NULL,
    category TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    test_id TEXT NOT NULL,
    repeat INTEGER NOT NULL,
    output TEXT,
    retrieved_context TEXT,
    latency_ms REAL,
    cost_usd REAL,
    scores TEXT,
    reasons TEXT,
    passed INTEGER,
    failed_checks TEXT,
    error TEXT,
    PRIMARY KEY (run_id, test_id, repeat)
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
"""


class Storage:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ----------------------------------------------------------- writes

    def start_run(self, run: Run, tests: list[TestCase]) -> None:
        """Create (or overwrite) a run and snapshot its suite."""
        with self.conn:
            self._delete(run.id)
            self.conn.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)",
                (run.id, run.created_at.isoformat(), run.app_version, run.config_hash, run.suite_version),
            )
            self.conn.executemany(
                "INSERT INTO test_cases VALUES (?,?,?,?)",
                [(run.id, t.id, t.category, t.model_dump_json()) for t in tests],
            )

    def save_results(self, results: list[TestResult]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        r.run_id, r.test_id, r.repeat, r.output, json.dumps(r.retrieved_context),
                        r.latency_ms, r.cost_usd, json.dumps(r.scores), json.dumps(r.reasons),
                        None if r.passed is None else int(r.passed), json.dumps(r.failed_checks), r.error,
                    )
                    for r in results
                ],
            )

    def import_bundle(self, bundle: RunBundle) -> None:
        self.start_run(bundle.run, bundle.tests)
        self.save_results(bundle.results)

    def delete_run(self, run_id: str) -> None:
        with self.conn:
            self._delete(run_id)

    def _delete(self, run_id: str) -> None:
        for table, col in (("results", "run_id"), ("test_cases", "run_id"), ("runs", "id")):
            self.conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (run_id,))

    # ----------------------------------------------------------- reads

    def has_run(self, run_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone() is not None

    def list_runs(self) -> list[Run]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created_at").fetchall()
        return [Run(**dict(r)) for r in rows]

    def load_bundle(self, run_id: str) -> RunBundle:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run '{run_id}' in {self.path}")
        tests = [
            TestCase.model_validate_json(r["data"])
            for r in self.conn.execute("SELECT data FROM test_cases WHERE run_id=? ORDER BY rowid", (run_id,))
        ]
        results = [
            TestResult(
                run_id=r["run_id"], test_id=r["test_id"], repeat=r["repeat"], output=r["output"] or "",
                retrieved_context=json.loads(r["retrieved_context"] or "[]"), latency_ms=r["latency_ms"] or 0.0,
                cost_usd=r["cost_usd"] or 0.0, scores=json.loads(r["scores"] or "{}"),
                reasons=json.loads(r["reasons"] or "{}"),
                passed=None if r["passed"] is None else bool(r["passed"]),
                failed_checks=json.loads(r["failed_checks"] or "[]"), error=r["error"],
            )
            for r in self.conn.execute(
                "SELECT * FROM results WHERE run_id=? ORDER BY test_id, repeat", (run_id,)
            )
        ]
        return RunBundle(run=Run(**dict(row)), tests=tests, results=results)

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------- JSON bundles (baselines)


def write_bundle(bundle: RunBundle, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(bundle.model_dump_json(indent=1), encoding="utf-8")


def read_bundle(path: str | Path) -> RunBundle:
    return RunBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))


def resolve_run(name: str, storage: Storage, baseline_path: str | Path) -> RunBundle:
    """Find a run by name: the database first, then a JSON file.

    `baseline` falls back to the committed baseline file, which is how CI compares a
    fresh run against the last merged state without a shared database.
    """
    if storage.has_run(name):
        return storage.load_bundle(name)
    if name == "baseline" and Path(baseline_path).exists():
        return read_bundle(baseline_path)
    if name.endswith(".json") and Path(name).exists():
        return read_bundle(name)
    raise KeyError(f"run '{name}' not found in the database, and no baseline file at {baseline_path}")
