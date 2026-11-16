"""Call an app over HTTP.

Template filling replaces the literal `{input}` inside every string of the request
template (recursively). We use str.replace rather than str.format so JSON-ish
templates containing other braces don't explode.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx

from ..config import TargetConfig
from .base import TargetResponse, extract_path, normalise_context

PLACEHOLDER = "{input}"


def fill_template(template, text: str):
    if isinstance(template, str):
        return template.replace(PLACEHOLDER, text)
    if isinstance(template, dict):
        return {k: fill_template(v, text) for k, v in template.items()}
    if isinstance(template, list):
        return [fill_template(v, text) for v in template]
    return template


class HttpTarget:
    def __init__(self, cfg: TargetConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        # `transport` lets tests inject httpx.MockTransport instead of a real server.
        self.client = httpx.AsyncClient(timeout=cfg.timeout_s, transport=transport, headers=cfg.headers)

    async def ask(self, text: str) -> TargetResponse:
        body = fill_template(self.cfg.request_template, text)
        last_exc: Exception | None = None
        # Retry only on network-level failures. A 4xx/5xx is the app's real answer to this
        # input and must be recorded, not hidden by retries.
        for attempt in range(self.cfg.retries + 1):
            start = time.perf_counter()
            try:
                if isinstance(body, str):
                    resp = await self.client.request(self.cfg.method, self.cfg.url, content=body)
                else:
                    resp = await self.client.request(self.cfg.method, self.cfg.url, json=body)
                latency = (time.perf_counter() - start) * 1000
                break
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.cfg.retries:
                    await asyncio.sleep(0.2 * 2**attempt)
        else:
            raise ConnectionError(f"{self.cfg.url} unreachable after {self.cfg.retries + 1} attempts: {last_exc}")

        resp.raise_for_status()
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return TargetResponse(output=resp.text, latency_ms=latency)
        output = extract_path(data, self.cfg.response_path)
        context = []
        if self.cfg.context_path:
            try:
                context = normalise_context(extract_path(data, self.cfg.context_path))
            except (KeyError, IndexError):
                context = []
        cost = 0.0
        if isinstance(data, dict) and isinstance(data.get("cost_usd"), (int, float)):
            cost = float(data["cost_usd"])
        return TargetResponse(output=str(output), context=context, latency_ms=latency, cost_usd=cost)

    async def aclose(self) -> None:
        await self.client.aclose()
