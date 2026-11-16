import random

from mutantqa.generate import generate_suite, write_spot_check
from mutantqa.generate.common import build_context, split_sections
from mutantqa.generate.dedupe import dedupe
from mutantqa.generate.edge import add_typos
from mutantqa.generate.injection import planted_document
from mutantqa.models import TestCase


def test_offline_suite_has_all_categories(offline_cfg):
    offline_cfg.generation.per_category = {"normal": 8, "edge": 4, "unanswerable": 4, "injection": 6,
                                           "consistency": 6, "safety": 3}
    tests, stats = generate_suite(offline_cfg, log=lambda *a: None)
    cats = {t.category for t in tests}
    assert cats == {"normal", "edge", "unanswerable", "injection", "consistency", "safety"}
    assert len({t.id for t in tests}) == len(tests)
    groups = {t.paraphrase_group for t in tests if t.category == "consistency"}
    assert all(sum(t.paraphrase_group == g for t in tests) >= 2 for g in groups)
    assert stats["planted_docs"] == ["zz_connection_pool_tuning.md"]


def test_planted_docs_excluded_and_indirect_tests_built(offline_cfg):
    ctx = build_context(offline_cfg)
    assert "zz_connection_pool_tuning.md" not in ctx.docs
    assert ctx.planted["zz_connection_pool_tuning.md"][0] == "PWNED-7C2D"
    offline_cfg.generation.per_category = {"injection": 8}
    tests, _ = generate_suite(offline_cfg, log=lambda *a: None)
    indirect = [t for t in tests if "indirect" in t.tags]
    assert indirect and all(t.tag_value("marker") == "PWNED-7C2D" for t in indirect)


def test_planted_doc_never_contains_marker_literally():
    text, marker = planted_document("pool tuning", "PWNED-ABCD")
    assert marker not in text and "PWNED" in text and "ABCD" in text


def test_typos_are_deterministic_and_change_text():
    q = "How do I configure connection timeouts for asynchronous clients?"
    a, b = add_typos(q, random.Random(1)), add_typos(q, random.Random(1))
    assert a == b and a != q


def test_dedupe_within_category_only():
    t = lambda i, cat, text, g=None: TestCase(id=i, category=cat, input=text, expected_behavior="x",  # noqa: E731
                                              paraphrase_group=g)
    tests = [t("1", "normal", "How do I set a timeout in HTTPX?"),
             t("2", "normal", "How do I set a timeout in HTTPX?"),
             t("3", "edge", "How do I set a timeout in HTTPX?"),  # same text, other category: kept
             t("4", "consistency", "How do I set a timeout in HTTPX?", "g1"),
             t("5", "consistency", "How do I set a timeout in HTTPX?", "g1")]  # paraphrase group: kept
    kept, dropped = dedupe(tests, 0.9)
    assert [x.id for x in kept] == ["1", "3", "4", "5"] and dropped == 1


def test_split_sections_skips_api_headings():
    md = "# Guide\n\nIntro text here.\n\n## `httpx.Client(timeout=5)`\n\nsig\n\n## Timeouts\n\nThey default to five seconds."
    assert [s.heading for s in split_sections("d.md", md)] == ["Guide", "Timeouts"]


def test_spot_check_sample(tmp_path):
    tests = [TestCase(id=str(i), category="normal" if i < 30 else "safety", input="q" * 10, expected_behavior="x")
             for i in range(35)]
    n = write_spot_check(tests, tmp_path / "s.csv")
    assert n == 4  # 3 normal + 1 safety
