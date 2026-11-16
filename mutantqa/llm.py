"""One door for every LLM call (generation and judging).

Why one wrapper:
  * LiteLLM means any provider works by changing a model string in the config.
  * The on-disk cache, keyed by sha256(model, system, prompt, temperature), makes
    re-scoring an unchanged output free. Judge calls are the bulk of the cost, and
    between two runs most outputs are identical, so this saves a lot.
  * Cost accounting lives in one place.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

OFFLINE = "offline"


class OfflineModelError(RuntimeError):
    """Raised when code asks the 'offline' model for a completion.

    Callers check `is_offline()` first and use their deterministic fallback.
    """


@dataclass
class LLMResponse:
    text: str
    cost_usd: float = 0.0
    cached: bool = False


_cache_dir = Path(".mutantqa/cache")
_lock = threading.Lock()
_spend = {"usd": 0.0, "calls": 0, "cache_hits": 0}


def configure(cache_dir: str | Path) -> None:
    global _cache_dir
    _cache_dir = Path(cache_dir)


def is_offline(model: str | None) -> bool:
    return model is None or model == OFFLINE


def spend() -> dict:
    """Cumulative LLM spend in this process (judges + generation)."""
    return dict(_spend)


def reset_spend() -> None:
    _spend.update(usd=0.0, calls=0, cache_hits=0)


def _key(model: str, system: str | None, prompt: str, temperature: float) -> str:
    blob = json.dumps([model, system or "", prompt, round(temperature, 3)])
    return hashlib.sha256(blob.encode()).hexdigest()


def call_llm(
    prompt: str,
    model: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    use_cache: bool = True,
) -> LLMResponse:
    if is_offline(model):
        raise OfflineModelError("model is 'offline'; use the heuristic fallback")

    key = _key(model, system, prompt, temperature)
    path = _cache_dir / key[:2] / f"{key}.json"
    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        with _lock:
            _spend["cache_hits"] += 1
        return LLMResponse(text=data["text"], cost_usd=0.0, cached=True)

    import litellm  # imported lazily: it is slow to import and not needed offline

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = litellm.completion(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, num_retries=2
    )
    text = resp.choices[0].message.content or ""
    try:
        cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:  # unknown pricing for local models etc.
        cost = 0.0

    with _lock:
        _spend["usd"] += cost
        _spend["calls"] += 1
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "text": text, "cost_usd": cost}), encoding="utf-8")
    return LLMResponse(text=text, cost_usd=cost, cached=False)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str):
    """Parse JSON from a model reply, tolerating code fences and leading chatter."""
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} or [...] block, trying whichever bracket comes
    # first, so "Here you go: [{...}]" yields the list rather than the object inside it.
    starts = [i for i, ch in enumerate(cleaned) if ch in "{["][:50]
    for start in starts:
        open_, close = cleaned[start], "}" if cleaned[start] == "{" else "]"
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:  # brackets inside strings don't count
                esc = not esc and ch == "\\"
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_:
                depth += 1
            elif ch == close:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not parse JSON from model output: {text[:200]!r}")


def call_json(prompt: str, model: str, system: str | None = None, temperature: float = 0.0, retries: int = 1):
    """call_llm + parse_json, re-asking once (uncached) if the reply isn't valid JSON."""
    resp = call_llm(prompt, model, system=system, temperature=temperature)
    for attempt in range(retries + 1):
        try:
            return parse_json(resp.text), resp
        except ValueError:
            if attempt == retries:
                raise
            resp = call_llm(prompt + "\n\nReturn ONLY valid JSON.", model, system=system,
                            temperature=temperature, use_cache=False)
    raise AssertionError("unreachable")
