"""mutantqa dashboard.

    streamlit run dashboard/app.py -- --config mutantqa.yaml

Four views: Overview (pass rate by category per run, cost, latency), Compare (gate
verdict, regressions with old/new outputs side by side), Test explorer (filter, read the
judge's reasoning) and Flaky tests (every repeat's output).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mutantqa.compare import compare_runs, summarize  # noqa: E402
from mutantqa.config import load_config  # noqa: E402
from mutantqa.models import CATEGORIES  # noqa: E402
from mutantqa.report import fmt_p  # noqa: E402
from mutantqa.scorers import verdicts  # noqa: E402
from mutantqa.storage import Storage, read_bundle  # noqa: E402

PASS, FAIL, FLAKY, MUTED = "#2E7D4F", "#B3261E", "#A86B12", "#5B6475"

st.set_page_config(page_title="mutantqa", page_icon="🧬", layout="wide")
st.markdown(
    """<style>
    .verdict { border-left: 6px solid var(--c); padding: .9rem 1.2rem; margin: .4rem 0 1.2rem;
               background: color-mix(in srgb, var(--c) 7%, transparent); border-radius: 0 6px 6px 0; }
    .verdict h2 { margin: 0; color: var(--c); font-size: 2rem; letter-spacing: -.01em; }
    .verdict p { margin: .3rem 0 0; }
    .out { white-space: pre-wrap; font-size: .9rem; line-height: 1.45; padding: .7rem .9rem;
           border: 1px solid rgba(128,128,128,.25); border-radius: 6px; max-height: 22rem; overflow: auto; }
    .tag { display: inline-block; padding: 0 .45rem; border-radius: 4px; font-size: .8rem;
           color: white; background: var(--c); margin-right: .3rem; }
    </style>""",
    unsafe_allow_html=True,
)


def config_path() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("MUTANTQA_CONFIG", "mutantqa.yaml"))
    args, _ = ap.parse_known_args()
    return args.config


@st.cache_resource
def get_storage(db_path: str) -> Storage:
    return Storage(db_path)


def load_bundle(storage: Storage, run_id: str, baseline_path: Path):
    if run_id == "baseline (file)":
        return read_bundle(baseline_path)
    return storage.load_bundle(run_id)


# Run colours carry no meaning (no red/green: those mean fail/pass everywhere else here),
# and adjacent runs get clearly different hues.
RUN_PALETTE = ["#2B59C3", "#9AA5B8", "#6A4C93", "#2F8F8F", "#444B59", "#B4BFD9"]


def category_chart(df: pd.DataFrame, runs: list[str]):
    import altair as alt

    order = [c for c in CATEGORIES if c in set(df["Category"])]
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Category:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            xOffset=alt.XOffset("Run:N", sort=runs),
            y=alt.Y("Pass rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
            color=alt.Color("Run:N", sort=runs, scale=alt.Scale(domain=runs, range=RUN_PALETTE[: len(runs)]),
                            legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
            tooltip=["Run", "Category", alt.Tooltip("Pass rate:Q", format=".0%")],
        )
        .properties(height=340)
    )


def tag(text: str, color: str) -> str:
    return f'<span class="tag" style="--c:{color}">{text}</span>'


def esc(s: str | None) -> str:
    import html

    return html.escape(s or "–")


# ------------------------------------------------------------------ setup

cfg_file = st.sidebar.text_input("Config file", config_path())
try:
    cfg = load_config(cfg_file)
except Exception as exc:
    st.error(f"Couldn't load {cfg_file}: {exc}. Pass a config with `-- --config path/to/mutantqa.yaml`.")
    st.stop()
storage = get_storage(str(cfg.path(cfg.db_path)))
run_ids = [r.id for r in storage.list_runs()]
baseline_path = cfg.path(cfg.baseline_path)
if baseline_path.exists():
    run_ids.append("baseline (file)")
if st.sidebar.button("Reload runs"):
    st.cache_resource.clear()
    st.rerun()
st.sidebar.caption(f"Database: `{cfg.db_path}` · {len(run_ids)} runs")

if not run_ids:
    st.title("No runs yet")
    st.write("Run your suite, then reload this page:")
    st.code(f"mutantqa run --tag v1 --config {cfg_file}")
    st.stop()

overview, compare_tab, explorer, flaky_tab = st.tabs(["Overview", "Compare", "Test explorer", "Flaky tests"])

# ------------------------------------------------------------------ overview

with overview:
    chosen = st.multiselect("Runs", run_ids, default=run_ids[-6:])
    rows, cat_rows = [], []
    for rid in chosen:
        s = summarize(load_bundle(storage, rid, baseline_path))
        rows.append({"Run": rid, "Tests": s.n_tests, "Pass rate": s.pass_rate, "Flaky": s.flaky_rate,
                     "p50 ms": round(s.latency_p50_ms), "p95 ms": round(s.latency_p95_ms),
                     "App cost $": round(s.app_cost_usd, 4), "Errors": s.errors})
        for cat, (p, n) in s.per_category.items():
            cat_rows.append({"Run": rid, "Category": cat, "Pass rate": p / n if n else 0})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Pass rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Flaky": st.column_config.NumberColumn(format="percent"),
        })
        st.subheader("Pass rate by category")
        st.altair_chart(category_chart(pd.DataFrame(cat_rows), chosen), width="stretch")

# ------------------------------------------------------------------ compare

with compare_tab:
    c1, c2 = st.columns(2)
    base_id = c1.selectbox("Baseline", run_ids, index=0)
    cand_id = c2.selectbox("Candidate", run_ids, index=len(run_ids) - 1)
    cmp = compare_runs(load_bundle(storage, base_id, baseline_path), load_bundle(storage, cand_id, baseline_path),
                       cfg.gate)
    color = PASS if cmp.gate.passed else FAIL
    reasons = "".join(f"<p>{esc(r)}</p>" for r in cmp.gate.reasons) or "<p>No gate rule was triggered.</p>"
    st.markdown(
        f'<div class="verdict" style="--c:{color}"><h2>{"Gate passes" if cmp.gate.passed else "Gate fails"}</h2>'
        f"<p>Pass rate {cmp.baseline_pass_rate:.1%} → {cmp.candidate_pass_rate:.1%} · {cmp.regressions} regressed · "
        f"{cmp.fixes} fixed · {cmp.flaky} flaky · McNemar {fmt_p(cmp.p_value)}</p>{reasons}</div>",
        unsafe_allow_html=True,
    )
    cat_df = pd.DataFrame([{"Category": c.category, "Tests": c.n, base_id: c.baseline_pass_rate,
                            cand_id: c.candidate_pass_rate} for c in cmp.categories])
    st.dataframe(cat_df, hide_index=True, width="stretch", column_config={
        base_id: st.column_config.NumberColumn(format="percent"),
        cand_id: st.column_config.NumberColumn(format="percent")})

    status = st.radio("Show", ["regressed", "fixed", "flaky", "still_failing"], horizontal=True)
    items = [ch for ch in cmp.changes if (ch.label == status if status == "flaky" else ch.status == status)]
    st.caption(f"{len(items)} tests")
    for ch in items[:100]:
        with st.expander(f"[{ch.category}] {ch.input[:120]}"):
            st.markdown(f"`{ch.test_id}` " + (tag("flaky", FLAKY) if ch.flaky else ""), unsafe_allow_html=True)
            st.markdown(f"**Input**\n\n{ch.input}")
            left, right = st.columns(2)
            left.markdown(f"**{base_id}**")
            left.markdown(f'<div class="out">{esc(ch.old_output)}</div>', unsafe_allow_html=True)
            right.markdown(f"**{cand_id}**")
            right.markdown(f'<div class="out">{esc(ch.new_output)}</div>', unsafe_allow_html=True)
            for name, why in ch.failed_checks.items():
                st.markdown(f"{tag(name, FAIL)} {esc(why)}", unsafe_allow_html=True)

# ------------------------------------------------------------------ test explorer

with explorer:
    rid = st.selectbox("Run", run_ids, index=len(run_ids) - 1, key="explore_run")
    bundle = load_bundle(storage, rid, baseline_path)
    vs = verdicts(bundle.tests, bundle.results)
    f1, f2, f3 = st.columns([2, 2, 3])
    cats = f1.multiselect("Category", list(CATEGORIES), default=[])
    stat = f2.selectbox("Status", ["all", "failing", "passing", "flaky"])
    query = f3.text_input("Search input text")
    tests = {t.id: t for t in bundle.tests}
    shown = []
    for t in bundle.tests:
        v = vs.get(t.id)
        if not v or (cats and t.category not in cats) or (query and query.lower() not in t.input.lower()):
            continue
        if (stat == "failing" and v.passed) or (stat == "passing" and not v.passed) or (stat == "flaky" and not v.flaky):
            continue
        shown.append((t, v))
    st.caption(f"{len(shown)} of {len(vs)} tests")
    table = pd.DataFrame([{"id": t.id, "category": t.category, "passed": v.passed,
                           "repeats passed": f"{v.pass_count}/{v.repeats}", "input": t.input[:140]}
                          for t, v in shown])
    if not table.empty:
        pick = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                            selection_mode="single-row", key="explorer_table")
        sel = pick.selection.rows
        if sel:
            t = tests[table.iloc[sel[0]]["id"]]
            st.markdown(f"#### {esc(t.input)}")
            st.markdown(f"**Expected behaviour:** {esc(t.expected_behavior)}")
            if t.reference_answer:
                st.markdown(f"**Reference:** {esc(t.reference_answer)}")
            if t.tags:
                st.caption("tags: " + ", ".join(t.tags))
            for r in sorted((r for r in bundle.results if r.test_id == t.id), key=lambda r: r.repeat):
                st.markdown(f"**Repeat {r.repeat}** " + tag("pass" if r.passed else "fail", PASS if r.passed else FAIL)
                            + f" <span style='color:{MUTED}'>{r.latency_ms:.0f} ms</span>", unsafe_allow_html=True)
                st.markdown(f'<div class="out">{esc(r.output)}</div>', unsafe_allow_html=True)
                score_rows = [{"check": k, "score": v, "failed": k in r.failed_checks, "reason": r.reasons.get(k, "")}
                              for k, v in r.scores.items()]
                st.dataframe(pd.DataFrame(score_rows), hide_index=True, width="stretch")
                if r.retrieved_context:
                    with st.expander(f"Retrieved context ({len(r.retrieved_context)} chunks)"):
                        for c in r.retrieved_context:
                            st.markdown(f'<div class="out">{esc(c)}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ flaky

with flaky_tab:
    rid = st.selectbox("Run", run_ids, index=len(run_ids) - 1, key="flaky_run")
    bundle = load_bundle(storage, rid, baseline_path)
    vs = verdicts(bundle.tests, bundle.results)
    flaky = [t for t in bundle.tests if vs.get(t.id) and vs[t.id].flaky]
    st.caption(f"{len(flaky)} flaky of {len(vs)} tests ({len(flaky) / max(1, len(vs)):.0%}). A test is flaky when "
               "its repeats disagree; its verdict is the majority.")
    if not flaky:
        st.write("No flaky tests in this run. Flakiness needs `run.repeats` of at least 2.")
    for t in flaky:
        v = vs[t.id]
        with st.expander(f"[{t.category}] {t.input[:110]}  ({v.pass_count}/{v.repeats} passed)"):
            rs = sorted((r for r in bundle.results if r.test_id == t.id), key=lambda r: r.repeat)
            cols = st.columns(len(rs))
            for col, r in zip(cols, rs):
                col.markdown(f"Repeat {r.repeat} " + tag("pass" if r.passed else "fail", PASS if r.passed else FAIL),
                             unsafe_allow_html=True)
                col.markdown(f'<div class="out">{esc(r.output)}</div>', unsafe_allow_html=True)
                if r.failed_checks:
                    col.caption("failed: " + ", ".join(r.failed_checks))
