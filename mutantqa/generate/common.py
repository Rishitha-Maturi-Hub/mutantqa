"""Shared state and helpers for the generators."""
from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .. import llm
from ..config import Config
from ..text import Chunk, KnowledgeIndex, chunk_docs, load_docs, sentences

# Planted (indirect-injection) documents are listed in a sidecar manifest in the
# knowledge dir, so the generator can build indirect injection tests from them and keep
# them out of the normal/unanswerable pools. The marker lives only in the manifest:
# the document itself never contains it literally (see injection.planted_document).
# Create one with `mutantqa plant`.
PLANTED_MANIFEST = ".mutantqa-planted.json"


def read_manifest(knowledge_dir: Path) -> dict[str, dict]:
    p = Path(knowledge_dir) / PLANTED_MANIFEST
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def new_id(category: str) -> str:
    return f"{category[:3]}-{uuid.uuid4().hex[:6]}"


@dataclass
class Section:
    doc: str
    heading: str
    body: str


@dataclass
class GenContext:
    cfg: Config
    rng: random.Random
    model: str
    docs: dict[str, str] = field(default_factory=dict)
    planted: dict[str, tuple[str, str]] = field(default_factory=dict)  # doc -> (marker, topic)
    chunks: list[Chunk] = field(default_factory=list)
    _index: KnowledgeIndex | None = None
    cost_usd: float = 0.0

    @property
    def offline(self) -> bool:
        return llm.is_offline(self.model)

    @property
    def index(self) -> KnowledgeIndex:
        if self._index is None:
            self._index = KnowledgeIndex(self.chunks)
        return self._index

    def ask_json(self, prompt: str, temperature: float = 0.7):
        """Generation calls use temperature > 0 for variety. Still cached, so re-running
        `generate` with the same seed and docs is free and reproducible."""
        data, resp = llm.call_json(prompt, self.model, temperature=temperature)
        self.cost_usd += resp.cost_usd
        return data

    def sections(self) -> list[Section]:
        return [s for name, text in self.docs.items() for s in split_sections(name, text)]


def build_context(cfg: Config) -> GenContext:
    llm.configure(cfg.path(cfg.cache_dir))
    ctx = GenContext(cfg=cfg, rng=random.Random(cfg.generation.seed), model=cfg.generation_model)
    if cfg.knowledge_dir:
        kdir = cfg.path(cfg.knowledge_dir)
        manifest = read_manifest(kdir)
        for name, text in load_docs(kdir).items():
            if name in manifest:
                ctx.planted[name] = (manifest[name]["marker"], manifest[name]["topic"])
            else:
                ctx.docs[name] = text
        ctx.chunks = chunk_docs(ctx.docs)
    return ctx


_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def clean_md(text: str) -> str:
    text = _CODE_BLOCK.sub(" ", text)
    text = re.sub(r"^\s*(?::::|:docstring:|:members:).*$", " ", text, flags=re.MULTILINE)  # mkdocs directives
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> text
    text = re.sub(r"<[^>]+>", " ", text)  # html
    text = re.sub(r"^\s*[-*>|]+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("`", "").replace("**", "").replace("__", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sections(doc: str, text: str) -> list[Section]:
    marks = list(_HEADING.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        heading = clean_md(m.group(2))
        body = text[m.end():end]
        # Skip API-reference headings like "http11.response_closed.complete {...}" or "`Client`".
        if re.search(r"[{}()=:]|\w\.\w|\w_\w", heading) or len(heading) < 3:
            continue
        if heading and body.strip():
            out.append(Section(doc, heading, body))
    return out


def prose_sentences(body: str) -> list[str]:
    """Sentences of real prose (no code, no tables, at least 6 words)."""
    out = []
    for s in sentences(clean_md(body)):
        s = " ".join(s.split())
        if (len(s.split()) >= 6 and not s.startswith(("$", ">>>", "import ", "!!!"))
                and s.count("=") < 2 and ":::" not in s):
            out.append(s)
    return out
