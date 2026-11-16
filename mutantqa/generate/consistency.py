"""Consistency: groups of 3-5 paraphrases of the same question, sharing a paraphrase_group."""
from __future__ import annotations

import uuid

from ..models import TestCase
from .common import GenContext, new_id

PROMPT = '''Write {k} paraphrases of this question. Each must ask for exactly the same information,
but vary wording, sentence structure, formality and length. Do not add new constraints.
Question: {q}
Return JSON: {{"paraphrases": ["...", "..."]}}'''

OFFLINE_PARAPHRASES = [
    "{q}",
    "Quick question: {ql}",
    "Hi! {q} Thanks in advance.",
    "Could you help me understand this? {q}",
    "{q} Please keep it brief.",
]


def generate_consistency(ctx: GenContext, n: int, sources: list[TestCase], group_size: int = 4) -> list[TestCase]:
    if n <= 0 or not sources:
        return []
    out: list[TestCase] = []
    srcs = list(sources)
    ctx.rng.shuffle(srcs)
    for src in srcs:
        k = min(group_size, n - len(out))
        if k < 2:
            break
        if ctx.offline:
            variants = [p.format(q=src.input, ql=src.input[0].lower() + src.input[1:]) for p in OFFLINE_PARAPHRASES[:k]]
        else:
            try:
                data = ctx.ask_json(PROMPT.format(k=k - 1, q=src.input), temperature=0.8)
            except ValueError:
                continue
            paras = [p for p in data.get("paraphrases", []) if isinstance(p, str) and p.strip()]
            variants = [src.input] + paras[: k - 1]
            if len(variants) < 2:
                continue
        gid = f"pg-{uuid.uuid4().hex[:6]}"
        for v in variants:
            out.append(TestCase(id=new_id("consistency"), category="consistency", input=v,
                                reference_answer=src.reference_answer,
                                expected_behavior="Gives the same factual answer as the other paraphrases in its group. "
                                                  + src.expected_behavior,
                                source_doc=src.source_doc, paraphrase_group=gid, tags=[f"from:{src.id}"]))
    return out[:n]
