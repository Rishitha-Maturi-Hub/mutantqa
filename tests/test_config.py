import pytest
from pydantic import ValidationError

from mutantqa.config import STARTER_CONFIG, Config, load_config


def test_starter_config_is_valid(tmp_path):
    p = tmp_path / "mutantqa.yaml"
    p.write_text(STARTER_CONFIG, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.target.type == "http"
    assert cfg.run.repeats == 3
    assert cfg.gate.critical_categories == ["injection", "safety"]
    assert cfg.base_dir == tmp_path


def test_repo_configs_are_valid():
    from conftest import ROOT

    for name in ("mutantqa.yaml", "mutantqa.offline.yaml"):
        load_config(ROOT / name)


def test_http_target_requires_url():
    with pytest.raises(ValidationError, match="url"):
        Config.model_validate({"target": {"type": "http"}})


def test_python_target_requires_module_function():
    with pytest.raises(ValidationError, match="module.path:function"):
        Config.model_validate({"target": {"type": "python", "function": "nocolon"}})


def test_unknown_category_rejected():
    with pytest.raises(ValidationError, match="unknown categories"):
        Config.model_validate({"target": {"type": "python", "function": "a:b"},
                               "generation": {"per_category": {"normal": 1, "jokes": 2}}})


def test_config_hash_tracks_settings(offline_cfg):
    h1 = offline_cfg.config_hash()
    offline_cfg.thresholds.correctness = 0.9
    assert offline_cfg.config_hash() != h1
    assert offline_cfg.config_hash("prompt v2") != offline_cfg.config_hash("prompt v1")


def test_missing_config_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="mutantqa init"):
        load_config(tmp_path / "nope.yaml")
