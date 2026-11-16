"""Core data model. Everything that crosses a module boundary is one of these."""
from __future__ import annotations

import datetime
import uuid
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["normal", "edge", "unanswerable", "injection", "consistency", "safety"]
CATEGORIES: tuple[str, ...] = ("normal", "edge", "unanswerable", "injection", "consistency", "safety")


class TestCase(BaseModel):
    __test__ = False  # stop pytest from trying to collect this class

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    category: Category
    input: str
    expected_behavior: str  # plain English: what a correct reply does
    reference_answer: str | None = None
    source_doc: str | None = None  # which document the question came from
    paraphrase_group: str | None = None
    tags: list[str] = []

    def tag_value(self, key: str) -> str | None:
        """Tags can carry `key:value` metadata, e.g. `marker:HACKED` for injection tests."""
        prefix = f"{key}:"
        for t in self.tags:
            if t.startswith(prefix):
                return t[len(prefix):]
        return None


class TestResult(BaseModel):
    __test__ = False

    run_id: str
    test_id: str
    repeat: int  # 0, 1, 2 ...
    output: str
    retrieved_context: list[str] = []
    latency_ms: float
    cost_usd: float = 0.0
    scores: dict[str, float] = {}  # {"correctness": 1.0, "groundedness": 0.8}
    reasons: dict[str, str] = {}  # judge explanations
    passed: bool | None = None
    failed_checks: list[str] = []  # names of the checks that failed
    error: str | None = None  # set when the target call itself failed


class Run(BaseModel):
    id: str  # e.g. "v1", "pr-42"
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    app_version: str = "unknown"  # git commit of the app under test
    config_hash: str = ""  # hash of model + prompt + settings
    suite_version: str = ""  # hash of the suite file


class TestVerdict(BaseModel):
    """A test's final verdict, aggregated across its repeats (majority vote)."""

    __test__ = False

    test_id: str
    category: str
    passed: bool
    flaky: bool  # repeats disagreed
    pass_count: int
    repeats: int


class RunBundle(BaseModel):
    """A run plus everything needed to compare it, independent of the database.

    This is also the on-disk format of `.mutantqa/baseline.json`, so CI can compare
    a fresh run against a baseline committed to the repo.
    """

    run: Run
    tests: list[TestCase]
    results: list[TestResult]
