from conftest import make_bundle

from mutantqa.models import Run
from mutantqa.storage import Storage, read_bundle, resolve_run, write_bundle


def test_roundtrip(tmp_path):
    st = Storage(tmp_path / "db.sqlite")
    b = make_bundle("v1", {"a": [True, False, True], "b": [False]}, {"b": "injection"})
    b.results[0].scores = {"correctness": 1.0}
    b.results[0].retrieved_context = ["c1", "c2"]
    st.import_bundle(b)
    got = st.load_bundle("v1")
    assert [t.id for t in got.tests] == ["a", "b"]
    assert got.tests[1].category == "injection"
    assert len(got.results) == 4
    r0 = next(r for r in got.results if r.test_id == "a" and r.repeat == 0)
    assert r0.scores == {"correctness": 1.0} and r0.retrieved_context == ["c1", "c2"]
    assert next(r for r in got.results if r.test_id == "b").failed_checks == ["correctness"]


def test_rerun_overwrites(tmp_path):
    st = Storage(tmp_path / "db.sqlite")
    st.import_bundle(make_bundle("v1", {"a": [True], "b": [True]}))
    st.import_bundle(make_bundle("v1", {"c": [False]}))
    got = st.load_bundle("v1")
    assert [t.id for t in got.tests] == ["c"] and len(got.results) == 1
    assert [r.id for r in st.list_runs()] == ["v1"]


def test_resolve_baseline_from_file(tmp_path):
    st = Storage(tmp_path / "db.sqlite")
    b = make_bundle("baseline", {"a": [True]})
    write_bundle(b, tmp_path / "baseline.json")
    assert read_bundle(tmp_path / "baseline.json").run.id == "baseline"
    assert resolve_run("baseline", st, tmp_path / "baseline.json").tests[0].id == "a"
    st.start_run(Run(id="baseline"), [])  # a DB run of the same name wins
    assert resolve_run("baseline", st, tmp_path / "baseline.json").tests == []
