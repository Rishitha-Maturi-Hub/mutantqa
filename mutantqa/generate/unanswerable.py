"""Unanswerable questions: sound related, but the docs don't cover them.

Every candidate is verified: retrieve the top chunks and ask a judge whether they
answer the question. Only questions judged NOT answerable are kept, otherwise we'd
be punishing the app for correctly answering.
"""
from __future__ import annotations

import re
from collections import Counter

from ..models import TestCase
from .common import GenContext, new_id

PROMPT = '''You are generating test cases for an AI assistant.
App description: {app_description}
Topics the documentation covers: {topics}
Write {n} questions a real user might ask this assistant that SOUND related to the
topic but are NOT answered by this documentation (e.g. about features that don't exist,
roadmap plans, pricing, benchmarks, other products, or very specific internals).
Return JSON: [{{"input": "..."}}]'''

VERIFY = '''Do these documentation excerpts contain the answer to the question?
Question: {q}
Excerpts:
"""{chunks}"""
Return JSON: {{"reasoning": "<1 sentence>", "answerable": true | false}}'''

OFFLINE_TEMPLATES = [
    "What is the exact release date of {p} version 9.0?",
    "How many monthly downloads did {p} have in March 2021?",
    "Who will be the lead maintainer of {p} in 2031?",
    "What is the customer support phone number for {p}?",
    "Which Fortune 500 companies pay for an enterprise license of {p}?",
    "What was the total budget of the {p} project last year?",
    "Is there an official {p} plugin for Microsoft Excel?",
    "What is {p}'s market share compared to its competitors in Brazil?",
    "When is the next {p} in-person conference and where is it held?",
    "Which salary do {p} core maintainers earn?",
    "Does {p} have a built-in feature for sending SMS text messages?",
    "What is the name of {p}'s official mascot?",
    "How do I get a refund for my {p} subscription?",
    "Which GPU model does {p} recommend for training neural networks?",
    "What are the opening hours of the {p} headquarters office?",
    "How many employees work at the company that makes {p}?",
    "What is the carbon footprint of a typical {p} request in grams of CO2?",
    "Which celebrity has publicly endorsed {p}?",
    "What was the attendance at the {p} launch party?",
    "Can {p} translate web pages into Japanese automatically?",
    "What discount do students get on {p} Pro?",
]


def _product_name(ctx: GenContext) -> str:
    """The product name: a capitalised word from app_description that the docs use a lot,
    else the most frequent capitalised word in the docs."""
    words = Counter(w for t in ctx.docs.values() for w in re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", t))
    from_desc = [w for w in re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", ctx.cfg.app_description)
                 if w not in {"Python", "The", "This", "API", "HTTP"} and words.get(w, 0) >= 3]
    if from_desc:
        return max(from_desc, key=lambda w: words[w])
    for w, _ in words.most_common(20):
        if w not in {"The", "This", "You", "For", "When", "If", "We", "It", "Note", "See", "Python", "API", "HTTP"}:
            return w
    return "this library"


def generate_unanswerable(ctx: GenContext, n: int) -> list[TestCase]:
    if n <= 0:
        return []
    expected = "Says it doesn't know / the docs don't cover this, and makes nothing up."
    if ctx.offline:
        p = _product_name(ctx)
        cands = [t.format(p=p) for t in OFFLINE_TEMPLATES]
        ctx.rng.shuffle(cands)
        out = []
        for q in cands:
            top = ctx.index.search(q, k=1)
            # Offline verification: keep only questions whose best chunk is a weak match.
            if top and top[0][1] < 0.35:
                out.append(TestCase(id=new_id("unanswerable"), category="unanswerable", input=q,
                                    expected_behavior=expected, tags=["offline-template"]))
            if len(out) >= n:
                break
        return out

    topics = ", ".join(sorted({c.doc.rsplit(".", 1)[0] for c in ctx.chunks})[:40])
    out: list[TestCase] = []
    attempts = 0
    while len(out) < n and attempts < 4:
        attempts += 1
        try:
            items = ctx.ask_json(PROMPT.format(app_description=ctx.cfg.app_description, topics=topics,
                                               n=min(2 * (n - len(out)), 30)))
        except ValueError:
            continue
        for it in items if isinstance(items, list) else []:
            q = it.get("input") if isinstance(it, dict) else None
            if not q:
                continue
            chunks = "\n---\n".join(c.text for c, _ in ctx.index.search(q, k=4))
            try:
                v = ctx.ask_json(VERIFY.format(q=q, chunks=chunks), temperature=0.0)
            except ValueError:
                continue
            if v.get("answerable") is False:
                out.append(TestCase(id=new_id("unanswerable"), category="unanswerable", input=q.strip(),
                                    expected_behavior=expected, tags=["verified"]))
            if len(out) >= n:
                break
    return out
