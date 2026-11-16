import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("MUTANTQA_EMBEDDINGS", "tfidf")  # deterministic, no model download
os.environ.pop("MUTANT", None)

from mutantqa.config import Config, TargetConfig  # noqa: E402
from mutantqa.models import Run, RunBundle, TestCase, TestResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offline_cfg(tmp_path) -> Config:
    return Config(
        target=TargetConfig(type="python", function="demo_app.rag:answer"),
        app_description="A support bot for the HTTPX Python library.",
        knowledge_dir=str(ROOT / "demo_app" / "docs"),
        canary="CANARY-7f3a91",
        suite_path=str(tmp_path / "suite.jsonl"),
        db_path=str(tmp_path / "db.sqlite"),
        cache_dir=str(tmp_path / "cache"),
        baseline_path=str(tmp_path / "baseline.json"),
        base_dir=tmp_path,
    )


@pytest.fixture
def demo_project(tmp_path, monkeypatch):
    """A throwaway copy of the repo's demo setup, with cwd set to it."""
    for item in ("demo_app", "mutantqa.offline.yaml"):
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, tmp_path / item, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy(src, tmp_path / item)
    (tmp_path / "tests" / "data").mkdir(parents=True)
    shutil.copy(ROOT / "tests" / "data" / "handwritten_suite.jsonl", tmp_path / "tests" / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def make_bundle(run_id: str, outcomes: dict[str, list[bool]], categories: dict[str, str] | None = None) -> RunBundle:
    """Build a RunBundle from {test_id: [pass/fail per repeat]}."""
    categories = categories or {}
    tests = [TestCase(id=tid, category=categories.get(tid, "normal"), input=f"q {tid}", expected_behavior="x")
             for tid in outcomes]
    results = [TestResult(run_id=run_id, test_id=tid, repeat=i, output=f"out {tid} {i}", latency_ms=1.0, passed=p,
                          failed_checks=[] if p else ["correctness"], reasons={"correctness": "why"})
               for tid, ps in outcomes.items() for i, p in enumerate(ps)]
    return RunBundle(run=Run(id=run_id), tests=tests, results=results)
