"""Normal questions: realistic user questions answered by one (or two) doc chunks."""
from __future__ import annotations

from ..models import TestCase
from .common import GenContext, new_id, prose_sentences

PROMPT = '''You are generating test cases for an AI assistant.
App description: {app_description}
Source passage:
"""{chunk}"""
Write {n} questions a real user would ask that this passage answers.
Vary difficulty and phrasing. Do not copy sentences from the passage.
Return JSON: [{{"input": "...", "reference_answer": "...",
"expected_behavior": "Gives <key fact>, grounded in docs"}}]'''

MULTIHOP_PROMPT = '''You are generating test cases for an AI assistant.
App description: {app_description}
Passage A:
"""{a}"""
Passage B:
"""{b}"""
Write 1 question a real user would ask that can only be answered by combining facts
from BOTH passages. Return JSON: [{{"input": "...", "reference_answer": "...",
"expected_behavior": "Combines <fact from A> and <fact from B>, grounded in docs"}}]'''


def _llm_normal(ctx: GenContext, n: int) -> list[TestCase]:
    out: list[TestCase] = []
    n_multi = max(1, n // 8) if n >= 8 else 0
    pool = [c for c in ctx.chunks if len(c.text.split()) > 60]
    ctx.rng.shuffle(pool)
    i = 0
    while len(out) < n - n_multi and i < len(pool):
        chunk = pool[i]
        i += 1
        try:
            items = ctx.ask_json(PROMPT.format(app_description=ctx.cfg.app_description, chunk=chunk.text, n=2))
        except ValueError:
            continue
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and it.get("input") and it.get("reference_answer"):
                out.append(TestCase(id=new_id("normal"), category="normal", input=it["input"].strip(),
                                    reference_answer=it["reference_answer"].strip(),
                                    expected_behavior=it.get("expected_behavior") or "Answers correctly from the docs.",
                                    source_doc=chunk.doc, tags=["single-hop"]))
    out = out[: n - n_multi]  # each chunk yields 2, so trim before adding multi-hop questions
    # Multi-hop: two chunks from the same document are most likely to combine sensibly.
    by_doc: dict[str, list] = {}
    for c in pool:
        by_doc.setdefault(c.doc, []).append(c)
    multi_docs = [d for d, cs in by_doc.items() if len(cs) >= 2]
    for _ in range(n_multi * 2):
        if len([t for t in out if "multi-hop" in t.tags]) >= n_multi or not multi_docs:
            break
        a, b = ctx.rng.sample(by_doc[ctx.rng.choice(multi_docs)], 2)
        try:
            items = ctx.ask_json(MULTIHOP_PROMPT.format(app_description=ctx.cfg.app_description, a=a.text, b=b.text))
        except ValueError:
            continue
        for it in (items if isinstance(items, list) else [])[:1]:
            if isinstance(it, dict) and it.get("input") and it.get("reference_answer"):
                out.append(TestCase(id=new_id("normal"), category="normal", input=it["input"].strip(),
                                    reference_answer=it["reference_answer"].strip(),
                                    expected_behavior=it.get("expected_behavior") or "Combines facts from two passages.",
                                    source_doc=a.doc, tags=["multi-hop"]))
    return out[:n]


QUESTION_TEMPLATES = [
    "What does the documentation say about {h}?",
    "How does {h} work?",
    "Can you explain {h}?",
    "I'm reading about {h}. What should I know?",
]


def _offline_normal(ctx: GenContext, n: int) -> list[TestCase]:
    """No-LLM fallback: one question per doc section, built from its heading; the
    reference is the section's first prose sentences. Crude, but grounded and
    deterministic, which is all the tool's own smoke tests need."""
    out = []
    sections = [s for s in ctx.sections() if len(prose_sentences(s.body)) >= 1 and 1 <= len(s.heading.split()) <= 8]
    ctx.rng.shuffle(sections)
    seen = set()
    for s in sections:
        key = s.heading.lower()
        if key in seen or key in {"introduction", "overview", "example", "examples", "usage", "notes"}:
            continue
        seen.add(key)
        ref = " ".join(prose_sentences(s.body)[:2])
        q = ctx.rng.choice(QUESTION_TEMPLATES).format(h=s.heading.strip().rstrip(".:"))
        out.append(TestCase(id=new_id("normal"), category="normal", input=q, reference_answer=ref,
                            expected_behavior=f"Explains {s.heading} consistently with the docs.",
                            source_doc=s.doc, tags=["single-hop", "offline-template"]))
        if len(out) >= n:
            break
    return out


def generate_normal(ctx: GenContext, n: int) -> list[TestCase]:
    if n <= 0:
        return []
    return _offline_normal(ctx, n) if ctx.offline else _llm_normal(ctx, n)
