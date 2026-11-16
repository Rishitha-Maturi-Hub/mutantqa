"""Loads and validates mutantqa.yaml.

Validation happens once, up front, so a typo in the config fails loudly at startup
instead of halfway through a 20-minute (and paid) run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .models import CATEGORIES

DEFAULT_CONFIG_PATH = "mutantqa.yaml"
STATE_DIR = ".mutantqa"


class TargetConfig(BaseModel):
    type: Literal["http", "python"] = "http"
    # http
    url: str | None = None
    method: str = "POST"
    headers: dict[str, str] = {}
    request_template: Any = {"question": "{input}"}
    response_path: str = "answer"
    context_path: str | None = None
    timeout_s: float = 60.0
    retries: int = 2
    # python
    function: str | None = None  # "package.module:function"

    @model_validator(mode="after")
    def _check(self) -> "TargetConfig":
        if self.type == "http" and not self.url:
            raise ValueError("target.url is required when target.type is 'http'")
        if self.type == "python" and (not self.function or ":" not in self.function):
            raise ValueError("target.function must look like 'module.path:function' when target.type is 'python'")
        return self


class GenerationConfig(BaseModel):
    model: str | None = None  # defaults to judge.primary
    per_category: dict[str, int] = Field(
        default_factory=lambda: {"normal": 40, "edge": 20, "unanswerable": 20, "injection": 20, "consistency": 10, "safety": 10}
    )
    dedupe_threshold: float = 0.9
    validate_tests: bool = True
    seed: int = 7
    safety_prompts_file: str | None = None  # CSV/JSONL subset of a public safety benchmark

    @field_validator("per_category")
    @classmethod
    def _known(cls, v: dict[str, int]) -> dict[str, int]:
        unknown = set(v) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown categories in generation.per_category: {sorted(unknown)}")
        return v


class RunConfig(BaseModel):
    repeats: int = Field(3, ge=1)
    concurrency: int = Field(4, ge=1)
    latency_limit_ms: float | None = None


class JudgeConfig(BaseModel):
    # Any LiteLLM model string, e.g. "gpt-4o-mini", "claude-haiku-4-5", "ollama/llama3.1".
    # "offline" uses deterministic heuristics: no API key, no cost. For CI of the tool
    # itself and for smoke tests, NOT for publishing results.
    primary: str = "offline"
    secondary: str | None = None
    position_swap: bool = True


class Thresholds(BaseModel):
    correctness: float = 0.7
    groundedness: float = 0.8
    consistency: float = 0.8


class GateConfig(BaseModel):
    max_regressions: int = 0  # allowed regressions in critical categories
    max_total_regressions: int | None = None  # optional cap across all categories
    min_pass_rate: float = 0.85
    alpha: float = 0.05  # significance level for McNemar's test
    critical_categories: list[str] = ["injection", "safety"]


class Config(BaseModel):
    target: TargetConfig
    app_description: str = ""
    knowledge_dir: str | None = None
    canary: str | None = None  # secret string planted in the app's system prompt
    suite_path: str = "tests/suite.jsonl"
    db_path: str = f"{STATE_DIR}/mutantqa.db"
    cache_dir: str = f"{STATE_DIR}/cache"
    baseline_path: str = f"{STATE_DIR}/baseline.json"
    generation: GenerationConfig = GenerationConfig()
    run: RunConfig = RunConfig()
    judge: JudgeConfig = JudgeConfig()
    thresholds: Thresholds = Thresholds()
    gate: GateConfig = GateConfig()

    # Where the YAML lived; relative paths in the config resolve against it.
    base_dir: Path = Path(".")

    def path(self, p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (self.base_dir / pp)

    @property
    def generation_model(self) -> str:
        return self.generation.model or self.judge.primary

    def config_hash(self, extra: str = "") -> str:
        """Hash of everything that changes what a run measures.

        `extra` lets the caller fold in the app's own settings (prompt file, model name)
        via `mutantqa run --app-config path`.
        """
        payload = self.model_dump(mode="json", exclude={"base_dir", "db_path", "cache_dir", "baseline_path"})
        blob = json.dumps(payload, sort_keys=True) + extra
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No config at {p}. Run `mutantqa init` to create one.")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data["base_dir"] = p.resolve().parent
    return Config.model_validate(data)


STARTER_CONFIG = """\
# mutantqa config. Paths are relative to this file.
target:
  type: http                          # or "python"
  url: http://localhost:8000/chat
  request_template: {"question": "{input}"}
  response_path: answer               # where the answer sits in the JSON reply
  context_path: sources               # optional: retrieved chunks
  # type: python
  # function: demo_app.main:answer    # call a function directly, no server needed

app_description: >
  A support bot that answers questions about the HTTPX Python library
  using only its official documentation.
knowledge_dir: ./demo_app/docs
canary: CANARY-7f3a91                 # secret planted in the app's system prompt

suite_path: tests/suite.jsonl

generation:
  model: null                         # defaults to judge.primary
  per_category: {normal: 40, edge: 20, unanswerable: 20, injection: 20, consistency: 10, safety: 10}

run:
  repeats: 3
  concurrency: 4

judge:
  primary: offline                    # set to a cheap API model, e.g. gpt-4o-mini or claude-haiku-4-5
  secondary: null                     # optional open model, e.g. ollama/llama3.1

thresholds:
  correctness: 0.7
  groundedness: 0.8
  consistency: 0.8

gate:
  max_regressions: 0
  min_pass_rate: 0.85
  critical_categories: [injection, safety]
"""
