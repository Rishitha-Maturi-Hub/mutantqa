"""Correctness: LLM judge comparing the output with the reference and expected behavior.

Scores are 0 / 0.5 / 1. A 3-point scale is coarse on purpose: judges are much more
consistent picking among three anchored options than producing a 1-10 score.
Reasoning comes *before* the score in the JSON so the model explains, then commits.
"""
from __future__ import annotations

from .. import llm
from ..text import recall
from .base import Check
from .refusal import pattern_refusal

PROMPT = """You are grading an AI assistant's answer.
Question: {input}
Reference answer: {reference_answer}
Expected behavior: {expected_behavior}
Assistant's answer: {output}
Rules: Judge meaning, not wording. Extra correct detail is fine.
Any factual contradiction of the reference = 0.
Score 1 = fully correct, 0.5 = partially correct/incomplete, 0 = wrong.
Return JSON: {{"reasoning": "<2 sentences>", "score": 0 | 0.5 | 1}}"""


def _snap(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return min((0.0, 0.5, 1.0), key=lambda a: abs(a - v))


def heuristic_correctness(reference: str | None, output: str) -> tuple[float, str] | None:
    """Offline fallback: share of the reference's key words present in the output."""
    if not reference:
        return None
    if pattern_refusal(output):
        return 0.0, "Offline heuristic: output is a refusal but a reference answer exists."
    r = recall(reference, output)
    score = 1.0 if r >= 0.6 else 0.5 if r >= 0.3 else 0.0
    return score, f"Offline heuristic: {r:.0%} of the reference's key words appear in the output."


def judge(model: str, test_input: str, reference: str | None, expected: str, output: str) -> tuple[float, str, float]:
    prompt = PROMPT.format(
        input=test_input, reference_answer=reference or "(none: grade against expected behavior)",
        expected_behavior=expected, output=output,
    )
    data, resp = llm.call_json(prompt, model)
    return _snap(data.get("score")), str(data.get("reasoning", "")), resp.cost_usd


def correctness_check(test, output: str, cfg) -> list[Check]:
    primary = cfg.judge.primary
    if llm.is_offline(primary):
        h = heuristic_correctness(test.reference_answer, output)
        if h is None:
            return []  # nothing to grade against offline
        score, why = h
        return [Check("correctness", score, score >= cfg.thresholds.correctness, why)]

    score, why, cost = judge(primary, test.input, test.reference_answer, test.expected_behavior, output)
    checks = [Check("correctness", score, score >= cfg.thresholds.correctness, why, cost)]
    secondary = cfg.judge.secondary
    if secondary and not llm.is_offline(secondary):
        s2, why2, cost2 = judge(secondary, test.input, test.reference_answer, test.expected_behavior, output)
        agree = (score >= cfg.thresholds.correctness) == (s2 >= cfg.thresholds.correctness)
        # Recorded, not gated: a disagreement means "needs human review".
        checks.append(Check("correctness_secondary", s2, True,
                            ("Judges agree. " if agree else "NEEDS REVIEW: judges disagree. ") + why2, cost2))
        checks.append(Check("judge_agreement", 1.0 if agree else 0.0, True, ""))
    return checks
