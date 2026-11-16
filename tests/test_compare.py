import pytest
from conftest import make_bundle

from mutantqa.compare import compare_runs, mcnemar_exact, summarize
from mutantqa.config import GateConfig
from mutantqa.report import comparison_markdown


@pytest.mark.parametrize("b,c", [(0, 0), (1, 0), (5, 1), (10, 2), (3, 3), (30, 9), (0, 7)])
def test_mcnemar_matches_statsmodels(b, c):
    sm = pytest.importorskip("statsmodels.stats.contingency_tables")
    expected = 1.0 if b + c == 0 else sm.mcnemar([[10, b], [c, 10]], exact=True).pvalue
    assert mcnemar_exact(b, c) == pytest.approx(expected)


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(2 * 0.5 ** 6)  # 0.03125


def test_labels_and_counts():
    base = make_bundle("v1", {"a": [True], "b": [False], "c": [True], "d": [False], "e": [True]})
    cand = make_bundle("v2", {"a": [True], "b": [True], "c": [False], "d": [False], "f": [True]})
    cmp = compare_runs(base, cand, GateConfig(min_pass_rate=0.0))
    status = {ch.test_id: ch.status for ch in cmp.changes}
    assert status == {"a": "still_passing", "b": "fixed", "c": "regressed", "d": "still_failing",
                      "e": "removed", "f": "added"}
    assert (cmp.regressions, cmp.fixes, cmp.n_paired) == (1, 1, 4)


def test_flaky_label():
    base = make_bundle("v1", {"a": [True, True, True]})
    cand = make_bundle("v2", {"a": [True, False, True]})
    cmp = compare_runs(base, cand, GateConfig(min_pass_rate=0.0))
    ch = cmp.changes[0]
    assert ch.status == "still_passing" and ch.flaky and ch.label == "flaky" and cmp.flaky == 1


def test_gate_critical_regression():
    base = make_bundle("v1", {"i": [True], **{f"n{k}": [True] for k in range(20)}}, {"i": "injection"})
    cand = make_bundle("v2", {"i": [False], **{f"n{k}": [True] for k in range(20)}}, {"i": "injection"})
    cmp = compare_runs(base, cand, GateConfig(min_pass_rate=0.5))
    assert not cmp.gate.passed and "critical" in cmp.gate.reasons[0]
    assert cmp.changes[0].failed_checks == {"correctness": "why"}


def test_gate_noise_is_not_significant():
    # 2 regressions vs 1 fix in a big suite: no critical category, above the floor -> pass.
    base = make_bundle("v1", {**{f"n{k}": [True] for k in range(40)}, "x": [False]})
    outcomes = {f"n{k}": [k >= 2] for k in range(40)} | {"x": [True]}
    cmp = compare_runs(base, make_bundle("v2", outcomes), GateConfig(min_pass_rate=0.85))
    assert cmp.regressions == 2 and not cmp.significant_drop and cmp.gate.passed


def test_gate_significant_drop_and_floor():
    base = make_bundle("v1", {f"n{k}": [True] for k in range(40)})
    cand = make_bundle("v2", {f"n{k}": [k >= 10] for k in range(40)})
    cmp = compare_runs(base, cand, GateConfig(min_pass_rate=0.85))
    assert cmp.significant_drop and cmp.p_value < 0.01
    assert len(cmp.gate.reasons) == 2  # floor + significance


def test_markdown_and_summary():
    base = make_bundle("v1", {"a": [True], "b": [True]})
    cand = make_bundle("v2", {"a": [False], "b": [True]})
    md = comparison_markdown(compare_runs(base, cand, GateConfig(min_pass_rate=0.9)))
    assert "❌" in md and "Regressions (1)" in md and "out a 0" in md
    s = summarize(cand)
    assert s.pass_rate == 0.5 and s.per_category == {"normal": (1, 2)}
