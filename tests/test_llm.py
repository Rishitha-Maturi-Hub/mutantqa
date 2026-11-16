import types

import pytest

from mutantqa import llm


@pytest.mark.parametrize("text,expected", [
    ('{"score": 1}', {"score": 1}),
    ('```json\n{"score": 0.5}\n```', {"score": 0.5}),
    ('Sure! Here you go: {"reasoning": "ok", "score": 0} hope that helps', {"reasoning": "ok", "score": 0}),
    ('[{"input": "q"}]', [{"input": "q"}]),
    ('Here you go: [{"input": "q"}] enjoy', [{"input": "q"}]),
    ('Note {braces} first: {"a": "x}y"}', {"a": "x}y"}),
    ('{"reasoning": "uses [brackets] and \\"quotes\\"", "score": 1}', {"reasoning": 'uses [brackets] and "quotes"', "score": 1}),
])
def test_parse_json(text, expected):
    assert llm.parse_json(text) == expected


def test_parse_json_failure():
    with pytest.raises(ValueError):
        llm.parse_json("no json here")


def test_offline_raises():
    with pytest.raises(llm.OfflineModelError):
        llm.call_llm("hi", "offline")


def test_disk_cache(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_completion(**kw):
        calls["n"] += 1
        msg = types.SimpleNamespace(content='{"score": 1}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    fake = types.SimpleNamespace(completion=fake_completion, completion_cost=lambda **kw: 0.002)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    llm.configure(tmp_path)
    llm.reset_spend()
    a = llm.call_llm("p", "some-model")
    b = llm.call_llm("p", "some-model")
    c = llm.call_llm("p", "other-model")
    assert calls["n"] == 2 and not a.cached and b.cached and not c.cached
    assert llm.spend() == {"usd": pytest.approx(0.004), "calls": 2, "cache_hits": 1}
    data, _ = llm.call_json("p", "some-model")
    assert data == {"score": 1}
