"""End-to-end: the CLI against the demo app, offline. This is the CI checkpoint for the
whole pipeline: generate -> run -> compare -> gate exit codes -> baseline."""
import json

from typer.testing import CliRunner

from mutantqa.cli import app

runner = CliRunner()
CFG = ["--config", "mutantqa.offline.yaml"]


def invoke(*args, env=None):
    return runner.invoke(app, list(args), env=env or {}, catch_exceptions=False)


def test_full_pipeline(demo_project, monkeypatch):
    r = invoke("init", "--path", "fresh.yaml")
    assert r.exit_code == 0 and (demo_project / "fresh.yaml").exists()
    assert ".env" in (demo_project / ".gitignore").read_text(encoding="utf-8")

    suite = "tests/data/handwritten_suite.jsonl"
    monkeypatch.delenv("MUTANT", raising=False)
    r = invoke("run", "--tag", "v1", "--suite", suite, "--repeats", "1", *CFG)
    assert r.exit_code == 0, r.output

    # Same app again: the gate must pass against itself (relative rules only; this tiny
    # handwritten suite is scored by the offline heuristics, so lower the floor).
    cfg_text = (demo_project / "mutantqa.offline.yaml").read_text(encoding="utf-8").replace("min_pass_rate: 0.6", "min_pass_rate: 0.2")
    (demo_project / "mutantqa.offline.yaml").write_text(cfg_text, encoding="utf-8")
    r = invoke("run", "--tag", "v1-again", "--suite", suite, "--repeats", "1", *CFG)
    assert invoke("compare", "v1", "v1-again", "--gate", *CFG).exit_code == 0

    # m6 removes the injection defence: a critical regression, so the gate must fail.
    monkeypatch.setenv("MUTANT", "m6")
    r = invoke("run", "--tag", "m6", "--suite", suite, "--repeats", "1", *CFG)
    monkeypatch.delenv("MUTANT")
    r = invoke("compare", "v1", "m6", "--gate", "--markdown", "report.md", "--json", "report.json", *CFG)
    assert r.exit_code == 1
    assert "injection" in (demo_project / "report.md").read_text(encoding="utf-8")
    assert json.loads((demo_project / "report.json").read_text(encoding="utf-8"))["gate"]["passed"] is False

    # Baseline file flow used by CI.
    assert invoke("baseline", "update", "--tag", "v1", *CFG).exit_code == 0
    assert invoke("compare", "baseline", "v1-again", "--gate", *CFG).exit_code == 0
    assert invoke("report", "--tag", "baseline", *CFG).exit_code == 0
    assert invoke("compare", "nope", "v1", *CFG).exit_code == 2


def test_plant_registers_manifest(demo_project):
    r = invoke("plant", "--topic", "retry budgets", "--out", "demo_app/docs/zz_retry.md", "--marker", "PWNED-1111")
    assert r.exit_code == 0
    manifest = json.loads((demo_project / "demo_app/docs/.mutantqa-planted.json").read_text(encoding="utf-8"))
    assert manifest["zz_retry.md"] == {"marker": "PWNED-1111", "topic": "retry budgets"}
