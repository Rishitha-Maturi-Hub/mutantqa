"""Shared types for scorers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Check:
    """One scorer's verdict on one output."""

    name: str
    score: float
    passed: bool
    reason: str = ""
    cost_usd: float = 0.0
