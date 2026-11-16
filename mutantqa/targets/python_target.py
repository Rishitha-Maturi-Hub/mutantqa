"""Call a Python function directly, so apps without a web server can be tested.

The function may be sync or async and may return:
  * a string (the answer), or
  * a dict with "answer" (or "output") and optionally "sources"/"context" and "cost_usd".
Sync functions run in a worker thread so they don't block the event loop.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
import time

from .base import TargetResponse, normalise_context


def import_function(spec: str):
    module_name, _, func_name = spec.partition(":")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_name)
    func = module
    for part in func_name.split("."):
        func = getattr(func, part)
    if not callable(func):
        raise TypeError(f"{spec} is not callable")
    return func


class PythonTarget:
    def __init__(self, spec: str):
        self.func = import_function(spec)
        self.is_async = inspect.iscoroutinefunction(self.func)

    async def ask(self, text: str) -> TargetResponse:
        start = time.perf_counter()
        if self.is_async:
            result = await self.func(text)
        else:
            result = await asyncio.to_thread(self.func, text)
        latency = (time.perf_counter() - start) * 1000
        if isinstance(result, dict):
            output = result.get("answer", result.get("output", ""))
            ctx = normalise_context(result.get("sources", result.get("context")))
            cost = float(result.get("cost_usd", 0.0) or 0.0)
            return TargetResponse(output=str(output), context=ctx, latency_ms=latency, cost_usd=cost)
        return TargetResponse(output=str(result), latency_ms=latency)

    async def aclose(self) -> None:
        return None
