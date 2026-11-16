"""The contract between mutantqa and the app under test.

Anything that can take a string and return an answer (plus, optionally, the chunks it
retrieved) is a Target. Keeping this tiny is what lets the runner, scorers and
reports ignore whether the app is an HTTP service or a Python function.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class TargetResponse(BaseModel):
    output: str
    context: list[str] = []
    latency_ms: float
    cost_usd: float = 0.0


@runtime_checkable
class Target(Protocol):
    async def ask(self, text: str) -> TargetResponse: ...

    async def aclose(self) -> None: ...


def extract_path(data, path: str | None):
    """Follow a dotted path like 'data.choices.0.text' through dicts and lists."""
    if path is None or path == "":
        return data
    cur = data
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"'{part}' not found while following path '{path}'")
            cur = cur[part]
        else:
            raise KeyError(f"cannot follow '{part}' into a {type(cur).__name__} (path '{path}')")
    return cur


def normalise_context(raw) -> list[str]:
    """Apps return sources as strings, dicts or None. Normalise to a list of strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for key in ("text", "content", "page_content", "chunk", "snippet"):
                if key in item and isinstance(item[key], str):
                    out.append(item[key])
                    break
            else:
                out.append(str(item))
        else:
            out.append(str(item))
    return out
