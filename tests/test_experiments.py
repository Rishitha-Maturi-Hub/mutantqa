import importlib.util
from pathlib import Path

import pytest

from mutantqa.report import fmt_p

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("human_agreement", ROOT / "experiments" / "human_agreement.py")
ha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ha)


def test_agreement_metrics_fail_is_positive():
    human = [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = fail
    judge = [0, 0, 1, 0, 1, 1, 1, 1]
    m = ha.metrics(human, judge)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["fail_precision"] == pytest.approx(2 / 3) and m["fail_recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["human_fail_rate"] == pytest.approx(3 / 8)


def test_agreement_metrics_degenerate_is_nan():
    m = ha.metrics([1, 1, 1], [1, 1, 1])
    assert m["kappa"] != m["kappa"] and m["accuracy"] == 1.0  # kappa undefined with one label


def test_fmt_p():
    assert fmt_p(0.00001) == "p < 0.0001" and fmt_p(0.0312) == "p = 0.0312"
