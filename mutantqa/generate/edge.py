"""Edge cases: transformations of normal questions.

Plain-code transforms (typos, noisy long inputs, formatting demands) are cheap and
deterministic. LLM transforms (code-mixed Hinglish/Tenglish, vague, false premise)
need a model; offline we use templates for code-mixing and skip the other two.
"""
from __future__ import annotations

import random

from ..models import TestCase
from .common import GenContext, new_id


def add_typos(text: str, rng: random.Random, rate: float = 0.18) -> str:
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            j = rng.randrange(len(w) - 1)
            if rng.random() < 0.5:
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]  # swap two letters
            else:
                w = w[:j] + w[j + 1:]  # drop a letter
            words[i] = w
    out = " ".join(words)
    return out if out != text else text.replace(" ", "  ", 1).lower()


NOISE = [
    "Sorry for the long message, I've been stuck on this all afternoon and my manager wants a fix before the "
    "standup tomorrow. We are migrating a legacy service and the previous developer left almost no comments.",
    "Some background: our team uses a monorepo with about forty services, we deploy on Kubernetes, and half of "
    "the codebase still targets an older Python version that we're slowly upgrading. None of that matters much.",
    "Also unrelated, but if you know a good pizza place near the office that would be great. Anyway, back to "
    "the actual problem I wanted to ask about.",
    "I tried searching the issue tracker and a couple of forum threads, but every answer was for a different "
    "version, so I'm not sure what applies to us anymore.",
]

FORMAT_DEMANDS = [
    "Answer in a markdown table.",
    "Reply in exactly two bullet points.",
    "Answer in one sentence, no code.",
    "Give the answer as JSON with keys 'answer' and 'source'.",
]

CODEMIX_TEMPLATES = [
    "Bhai ek doubt hai, {q} Thoda simple mein samjha do please.",
    "Yaar mujhe samajh nahi aa raha: {q}",
    "Anna oka doubt undi, {q} Konchem explain cheyyandi.",
    "{q} Ye kaise karte hain, quick batao na?",
]

LLM_PROMPTS = {
    "codemix": '''Rewrite this user question the way an Indian developer might type it in a chat, mixing
English with Hinglish or Tenglish (Roman script). Keep every technical term in English and keep the meaning.
Question: {q}
Return JSON: {{"input": "..."}}''',
    "vague": '''Rewrite this question so it is vague and underspecified, as a hurried user would write it
(drop the key detail that makes it answerable precisely).
Question: {q}
Return JSON: {{"input": "..."}}''',
    "false_premise": '''Here is a question and its correct answer from the documentation.
Question: {q}
Correct answer: {a}
Write a new question that contains a FALSE premise contradicting the documentation
(e.g. "Since version 2 removed X, how do I..."). Also write the correction a good assistant should give.
Return JSON: {{"input": "...", "correction": "..."}}''',
}


def _code_transform(kind: str, src: TestCase, rng: random.Random) -> TestCase:
    if kind == "typo":
        text, why = add_typos(src.input, rng), "typos"
    elif kind == "long":
        a, b = rng.sample(NOISE, 2)
        text, why = f"{a} {src.input} {b}", "irrelevant surrounding text"
    elif kind == "format":
        text, why = f"{src.input} {rng.choice(FORMAT_DEMANDS)}", "a formatting demand"
    else:  # codemix template
        text, why = rng.choice(CODEMIX_TEMPLATES).format(q=src.input), "code-mixed phrasing"
    return TestCase(id=new_id("edge"), category="edge", input=text, reference_answer=src.reference_answer,
                    expected_behavior=f"Answers correctly despite {why}: {src.expected_behavior}",
                    source_doc=src.source_doc, tags=[kind, f"from:{src.id}"])


def generate_edge(ctx: GenContext, n: int, sources: list[TestCase]) -> list[TestCase]:
    if n <= 0 or not sources:
        return []
    kinds = ["typo", "long", "format", "codemix"] if ctx.offline else \
        ["typo", "long", "format", "codemix", "vague", "false_premise"]
    out: list[TestCase] = []
    i = 0
    while len(out) < n and i < n * 3:
        kind = kinds[i % len(kinds)]
        src = sources[(i // len(kinds) + i) % len(sources)]
        i += 1
        if kind in ("typo", "long", "format") or (kind == "codemix" and ctx.offline):
            out.append(_code_transform(kind, src, ctx.rng))
            continue
        try:
            data = ctx.ask_json(LLM_PROMPTS[kind].format(q=src.input, a=src.reference_answer or ""))
        except ValueError:
            continue
        if not isinstance(data, dict) or not data.get("input"):
            continue
        if kind == "codemix":
            out.append(TestCase(id=new_id("edge"), category="edge", input=data["input"],
                                reference_answer=src.reference_answer,
                                expected_behavior=f"Answers correctly despite code-mixed phrasing: {src.expected_behavior}",
                                source_doc=src.source_doc, tags=["codemix", f"from:{src.id}"]))
        elif kind == "vague":
            out.append(TestCase(id=new_id("edge"), category="edge", input=data["input"], reference_answer=None,
                                expected_behavior="Asks a clarifying question, or answers the most likely reading "
                                                  "while saying what it assumed. Does not invent specifics.",
                                source_doc=src.source_doc, tags=["vague", f"from:{src.id}"]))
        else:
            out.append(TestCase(id=new_id("edge"), category="edge", input=data["input"],
                                reference_answer=data.get("correction"),
                                expected_behavior="Points out that the premise is false and gives the correct "
                                                  "information from the docs; does not play along.",
                                source_doc=src.source_doc, tags=["false-premise", f"from:{src.id}"]))
    return out[:n]
