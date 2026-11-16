import asyncio
import json

import httpx
import pytest

from mutantqa.config import TargetConfig
from mutantqa.targets.base import extract_path, normalise_context
from mutantqa.targets.http_target import HttpTarget, fill_template
from mutantqa.targets.python_target import PythonTarget


def cfg(**kw):
    base = dict(type="http", url="http://app.test/chat", request_template={"question": "{input}"},
                response_path="answer", context_path="sources", retries=2)
    base.update(kw)
    return TargetConfig(**base)


def test_fill_template_recursive_and_brace_safe():
    t = {"messages": [{"role": "user", "content": "Q: {input}"}], "meta": {"json": "{not a field}"}, "n": 1}
    out = fill_template(t, "hi {there}")
    assert out["messages"][0]["content"] == "Q: hi {there}"
    assert out["meta"]["json"] == "{not a field}"
    assert out["n"] == 1


def test_extract_path_dicts_and_lists():
    data = {"choices": [{"message": {"content": "yes"}}]}
    assert extract_path(data, "choices.0.message.content") == "yes"
    with pytest.raises(KeyError):
        extract_path(data, "choices.0.nope")


def test_normalise_context_shapes():
    assert normalise_context(None) == []
    assert normalise_context("a") == ["a"]
    assert normalise_context([{"text": "a"}, {"page_content": "b"}, "c"]) == ["a", "b", "c"]


def test_http_target_happy_path():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answer": "42", "sources": [{"text": "ctx"}], "cost_usd": 0.001})

    t = HttpTarget(cfg(), transport=httpx.MockTransport(handler))
    r = asyncio.run(t.ask("what?"))
    assert seen["body"] == {"question": "what?"}
    assert r.output == "42" and r.context == ["ctx"] and r.cost_usd == 0.001
    assert r.latency_ms >= 0


def test_http_target_retries_network_errors_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"answer": "ok"})

    t = HttpTarget(cfg(context_path=None), transport=httpx.MockTransport(handler))
    assert asyncio.run(t.ask("x")).output == "ok"
    assert calls["n"] == 3


def test_http_target_gives_up_after_retries():
    def handler(request):
        raise httpx.ConnectError("down")

    t = HttpTarget(cfg(retries=1), transport=httpx.MockTransport(handler))
    with pytest.raises(ConnectionError, match="2 attempts"):
        asyncio.run(t.ask("x"))


def test_http_target_does_not_retry_http_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"detail": "bad"})

    t = HttpTarget(cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(t.ask("x"))
    assert calls["n"] == 1


def test_http_target_plain_text_response():
    t = HttpTarget(cfg(), transport=httpx.MockTransport(lambda r: httpx.Response(200, text="plain")))
    assert asyncio.run(t.ask("x")).output == "plain"


def _sync_bot(q):
    return {"answer": q.upper(), "sources": ["s1"]}


async def _async_bot(q):
    return f"async:{q}"


def test_python_target_sync_and_async():
    r = asyncio.run(PythonTarget("tests.test_targets:_sync_bot").ask("hi"))
    assert r.output == "HI" and r.context == ["s1"]
    r = asyncio.run(PythonTarget("tests.test_targets:_async_bot").ask("hi"))
    assert r.output == "async:hi" and r.context == []
