"""Groundedness (hallucination check).

1. Split the answer into atomic factual claims (LLM).
2. Check every claim against the retrieved context (one batched LLM call).
Score = fraction of claims supported. A refusal makes no claims, so it scores 1.0:
groundedness measures "made nothing up", not "was helpful". Correctness covers that.

Not applicable when the app doesn't return its retrieved context.
"""
from __future__ import annotations

from .. import llm
from ..text import content_tokens, sentences
from .base import Check
from .refusal import pattern_refusal

CLAIMS_PROMPT = """Split the following answer into a list of short, self-contained factual claims.
Ignore greetings, hedges and refusals. If there are no factual claims, return [].
Answer: {output}
Return JSON: {{"claims": ["...", "..."]}}"""

VERIFY_PROMPT = """For each claim, decide whether it is supported by the context.
"Supported" means the context states it or it follows directly from the context.
Context:
\"\"\"{context}\"\"\"
Claims:
{claims}
Return JSON: {{"verdicts": [{{"claim": "...", "reasoning": "<short>", "supported": true | false}}]}}"""


def heuristic_groundedness(output: str, context: list[str]) -> tuple[float, str]:
    """Offline fallback: a sentence is supported if >=75% of its content words occur in the context."""
    ctx = set(content_tokens(" ".join(context)))
    claims = [s for s in sentences(output) if not pattern_refusal(s)]
    if not claims:
        return 1.0, "No factual claims."
    unsupported = []
    for s in claims:
        words = set(content_tokens(s))
        if words and len(words & ctx) / len(words) < 0.75:
            unsupported.append(s)
    score = 1 - len(unsupported) / len(claims)
    why = f"Offline heuristic: {len(claims) - len(unsupported)}/{len(claims)} sentences supported by context."
    if unsupported:
        why += f" First unsupported: {unsupported[0][:120]!r}"
    return score, why


def groundedness_check(output: str, context: list[str], cfg) -> Check | None:
    if not context:
        return None
    model = cfg.judge.primary
    if pattern_refusal(output) and len(sentences(output)) <= 2:
        return Check("groundedness", 1.0, True, "Refusal: no factual claims.")
    if llm.is_offline(model):
        score, why = heuristic_groundedness(output, context)
        return Check("groundedness", score, score >= cfg.thresholds.groundedness, why)

    data, r1 = llm.call_json(CLAIMS_PROMPT.format(output=output), model)
    claims = [c for c in data.get("claims", []) if isinstance(c, str) and c.strip()]
    if not claims:
        return Check("groundedness", 1.0, True, "No factual claims.", r1.cost_usd)
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    data2, r2 = llm.call_json(VERIFY_PROMPT.format(context="\n---\n".join(context), claims=numbered), model)
    verdicts = data2.get("verdicts", [])
    supported = sum(1 for v in verdicts if v.get("supported") is True)
    score = supported / len(claims)
    bad = [v.get("claim", "") for v in verdicts if v.get("supported") is not True]
    why = f"{supported}/{len(claims)} claims supported."
    if bad:
        why += f" Unsupported: {bad[0][:150]!r}"
    return Check("groundedness", score, score >= cfg.thresholds.groundedness, why, r1.cost_usd + r2.cost_usd)
