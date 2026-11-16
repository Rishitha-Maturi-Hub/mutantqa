"""Text helpers shared by the generator and the heuristic (offline) scorers.

Embeddings: if `sentence-transformers` is installed (pip install mutantqa[embeddings])
we use all-MiniLM-L6-v2, which runs fine on a CPU. Otherwise we fall back to TF-IDF
over word 1-2 grams. The fallback is weaker for paraphrase detection but has no
download and is fully deterministic, which is what the tool's own test suite wants.
Force a backend with MUTANTQA_EMBEDDINGS=tfidf or MUTANTQA_EMBEDDINGS=<st model name>.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being
    this that these those it its it's i you he she we they me my your our their them do does did doing
    have has had having can could should would will shall may might must not no yes so than too very
    what which who whom whose when where why how all any both each few more most other some such only own
    same just also into over under again further once here there about above below up down out off
    use using used get got via per e.g i.e etc""".split()
)

_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-\.']*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])|\n{2,}")


def tokens(text: str) -> list[str]:
    return [t.lower().strip(".'") for t in _WORD.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t and t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    return [p for p in parts if len(content_tokens(p)) > 0]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def recall(reference: str, candidate: str) -> float:
    """Fraction of the reference's content words that appear in the candidate."""
    ref = set(content_tokens(reference))
    if not ref:
        return 1.0
    return len(ref & set(content_tokens(candidate))) / len(ref)


# ---------------------------------------------------------------- documents & chunks


@dataclass
class Chunk:
    doc: str
    text: str


def load_docs(directory: str | Path) -> dict[str, str]:
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"knowledge_dir {d} does not exist")
    docs = {}
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() in {".md", ".txt", ".rst"} and p.is_file() and "license" not in p.name.lower():
            docs[str(p.relative_to(d))] = p.read_text(errors="ignore", encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"no .md/.txt/.rst files in {d}")
    return docs


def chunk_text(text: str, max_words: int = 380, overlap_words: int = 60) -> list[str]:
    """Pack paragraphs into ~500-token chunks (~380 words), overlapping by ~60 words."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    count = 0
    for para in paras:
        n = len(para.split())
        if cur and count + n > max_words:
            chunks.append("\n\n".join(cur))
            tail = " ".join("\n\n".join(cur).split()[-overlap_words:])
            cur, count = [tail], len(tail.split())
        cur.append(para)
        count += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_docs(docs: dict[str, str], max_words: int = 380, overlap_words: int = 60) -> list[Chunk]:
    return [Chunk(name, c) for name, text in docs.items() for c in chunk_text(text, max_words, overlap_words)]


# ---------------------------------------------------------------- embeddings


@lru_cache(maxsize=2)
def _st_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class Embedder:
    """Fit on a corpus, then encode queries into the same space. Rows are L2-normalised."""

    def __init__(self, backend: str | None = None):
        backend = backend or os.environ.get("MUTANTQA_EMBEDDINGS", "auto")
        self.backend = "tfidf"
        self._st_name = None
        if backend != "tfidf":
            name = "all-MiniLM-L6-v2" if backend == "auto" else backend
            try:
                _st_model(name)
                self.backend, self._st_name = "st", name
            except Exception:
                if backend != "auto":
                    raise
        self._vec = None

    def fit_encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            return np.asarray(_st_model(self._st_name).encode(texts, normalize_embeddings=True))
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=list(STOPWORDS))
        return self._vec.fit_transform(texts).toarray()

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            return np.asarray(_st_model(self._st_name).encode(texts, normalize_embeddings=True))
        if self._vec is None:
            raise RuntimeError("call fit_encode() first")
        return self._vec.transform(texts).toarray()


class KnowledgeIndex:
    """Tiny in-memory vector index over the app's documents.

    The generator uses it to verify that 'unanswerable' questions really are not
    answered by the docs.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or Embedder()
        self.matrix = self.embedder.fit_encode([c.text for c in chunks])

    @classmethod
    def from_dir(cls, directory: str | Path) -> "KnowledgeIndex":
        return cls(chunk_docs(load_docs(directory)))

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.embedder.encode([query])[0]
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]
