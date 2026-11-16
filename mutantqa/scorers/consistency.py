"""Consistency across paraphrases and repeats.

All outputs in one paraphrase group (every paraphrase x every repeat) should give the
same factual answer. We judge pairs, not all n^2 of them: every pair of paraphrases'
first repeats, plus each later repeat against its own first repeat. The LLM judge sees
each pair in both orders (A,B) and (B,A) and only counts agreement if both say "same",
which cancels position bias.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .. import llm
from ..models import TestCase, TestResult
from ..text import jaccard
from .base import Check
from .refusal import pattern_refusal

PROMPT = """Two answers were given to the same question (possibly phrased differently).
Question: {question}
Answer A: {a}
Answer B: {b}
Do A and B give the same factual answer? Ignore wording, length and style; only
contradictions or different facts count as different.
Return JSON: {{"reasoning": "<1 sentence>", "same": true | false}}"""


def _same(question: str, a: str, b: str, model: str, swap: bool) -> tuple[bool, float]:
    if llm.is_offline(model):
        ra, rb = pattern_refusal(a), pattern_refusal(b)
        if ra or rb:
            return ra == rb, 0.0
        return jaccard(a, b) >= 0.35, 0.0
    d1, r1 = llm.call_json(PROMPT.format(question=question, a=a, b=b), model)
    if not swap:
        return bool(d1.get("same")), r1.cost_usd
    d2, r2 = llm.call_json(PROMPT.format(question=question, a=b, b=a), model)
    return bool(d1.get("same")) and bool(d2.get("same")), r1.cost_usd + r2.cost_usd


def consistency_checks(tests: list[TestCase], results: list[TestResult], cfg) -> dict[tuple[str, int], Check]:
    """Returns a Check per (test_id, repeat) for every result in a paraphrase group."""
    by_test: dict[str, list[TestResult]] = defaultdict(list)
    for r in results:
        if r.error is None:
            by_test[r.test_id].append(r)
    groups: dict[str, list[TestCase]] = defaultdict(list)
    for t in tests:
        if t.paraphrase_group and by_test.get(t.id):
            groups[t.paraphrase_group].append(t)

    model, swap = cfg.judge.primary, cfg.judge.position_swap
    out: dict[tuple[str, int], Check] = {}
    for gid, members in groups.items():
        firsts = {t.id: sorted(by_test[t.id], key=lambda r: r.repeat)[0] for t in members}
        pairs = [(firsts[a.id], firsts[b.id]) for a, b in combinations(members, 2)]
        for t in members:
            rs = sorted(by_test[t.id], key=lambda r: r.repeat)
            pairs += [(rs[0], r) for r in rs[1:]]
        if not pairs:
            continue
        question = members[0].input
        agree, cost, disagreeing = 0, 0.0, []
        for a, b in pairs:
            same, c = _same(question, a.output, b.output, model, swap)
            cost += c
            agree += same
            if not same:
                disagreeing.append((a.test_id, a.repeat, b.test_id, b.repeat))
        score = agree / len(pairs)
        why = f"Group {gid}: {agree}/{len(pairs)} answer pairs agree."
        if disagreeing:
            a_id, a_rep, b_id, b_rep = disagreeing[0]
            why += f" e.g. {a_id}#{a_rep} vs {b_id}#{b_rep} differ."
        members_results = [r for t in members for r in by_test[t.id]]
        share = cost / max(1, len(members_results))
        for r in members_results:
            out[(r.test_id, r.repeat)] = Check("consistency", score, score >= cfg.thresholds.consistency, why, share)
    return out
