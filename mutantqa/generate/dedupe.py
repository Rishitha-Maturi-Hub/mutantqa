"""Remove near-duplicate tests.

Dedupe happens *within* a category: an edge-case typo variant is meant to be similar
to its source normal question, and paraphrase groups are meant to be similar to each
other, so comparing across those boundaries would delete exactly the tests we want.
For consistency tests we compare one representative per group and drop whole groups.
"""
from __future__ import annotations

import numpy as np

from ..models import TestCase
from ..text import Embedder


def _keep_mask(texts: list[str], threshold: float) -> list[bool]:
    if len(texts) < 2:
        return [True] * len(texts)
    vecs = Embedder().fit_encode(texts)
    keep: list[int] = []
    mask = []
    for i in range(len(texts)):
        dup = bool(keep) and float(np.max(vecs[keep] @ vecs[i])) > threshold
        mask.append(not dup)
        if not dup:
            keep.append(i)
    return mask


def dedupe(tests: list[TestCase], threshold: float = 0.9) -> tuple[list[TestCase], int]:
    out: list[TestCase] = []
    dropped = 0
    cats = list(dict.fromkeys(t.category for t in tests))
    for cat in cats:
        ts = [t for t in tests if t.category == cat]
        if cat == "consistency":
            groups: dict[str, list[TestCase]] = {}
            for t in ts:
                groups.setdefault(t.paraphrase_group or t.id, []).append(t)
            reps = list(groups)
            mask = _keep_mask([groups[g][0].input for g in reps], threshold)
            for g, k in zip(reps, mask):
                if k:
                    out += groups[g]
                else:
                    dropped += len(groups[g])
        else:
            mask = _keep_mask([t.input for t in ts], threshold)
            out += [t for t, k in zip(ts, mask) if k]
            dropped += mask.count(False)
    return out, dropped
