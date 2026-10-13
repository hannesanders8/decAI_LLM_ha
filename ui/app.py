"""LocAssess — Streamlit UI (SPEC §10.3).

Start with `locassess ui` (or `streamlit run ui/app.py`). Database and configuration can be
overridden with the environment variables LOCASSESS_DB and LOCASSESS_CONFIG.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from locassess import pipeline, ui_data
from locassess.config import load_config
from locassess.db import repo

DB_PATH = Path(os.environ.get("LOCASSESS_DB", "data/locassess.db"))
CONFIG_PATH = Path(os.environ.get("LOCASSESS_CONFIG", "config.yaml"))
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("config.example.yaml")

st.set_page_config(page_title="LocAssess", layout="wide")


@st.cache_resource
def get_conn(path: str):
    conn = repo.connect(path, check_same_thread=False)   # Streamlit reruns the script in worker threads
    repo.init_db(conn)
    return conn


conn = get_conn(str(DB_PATH))
cfg = load_config(CONFIG_PATH)
crit_table_all = pd.DataFrame([dict(r) for r in conn.execute("SELECT * FROM criterion ORDER BY main_id")])
if crit_table_all.empty:
    st.error("No criteria in the database. Run `locassess load-criteria` first.")
    st.stop()
FACTORS = list(dict.fromkeys(crit_table_all["factor"]))


def colour_rating(v):
    c = ui_data.RATING_COLOURS.get(v)
    return f"background-color: {c}; color: white" if c else ""


def styled(df: pd.DataFrame):
    return df.style.map(colour_rating, subset=["rating"]).format({"score": lambda x: "" if pd.isna(x) else f"{x:+.1f}"})


def slider_values() -> dict[str, int]:
    return {m: int(st.session_state.get(f"w_{m}", cfg.weights_default)) for m in crit_table_all["main_id"]}


# ---------------------------------------------------------------------------
# Sidebar: run selection and weight sliders
# ---------------------------------------------------------------------------

st.sidebar.title("LocAssess")
st.sidebar.caption(f"Database: `{DB_PATH}` · Config: `{CONFIG_PATH}`")
runs = ui_data.scoring_runs(conn)
selected_run = None
if not runs.empty:
    titles = {int(r["scoring_run_id"]): ui_data.run_title(r) for r in runs.to_dict("records")}
    selected_run = st.sidebar.selectbox("Location profile (scoring run)", list(titles), format_func=titles.get)

st.sidebar.subheader("Criterion weights")
st.sidebar.caption("0 excludes a criterion; all 1 = equal weighting. The overall score updates live.")
if st.sidebar.button("Reset all to default"):
    for m in crit_table_all["main_id"]:
        st.session_state[f"w_{m}"] = cfg.weights_default
for factor in FACTORS:
    with st.sidebar.expander(factor):
        for r in crit_table_all[crit_table_all["factor"] == factor].itertuples():
            st.slider(f"{r.main_id} {r.name}", 0, 10, value=cfg.weights_default, key=f"w_{r.main_id}")
sliders = slider_values()

tab_profile, tab_compare, tab_report, tab_log, tab_run, tab_export = st.tabs(
    ["Profile", "Compare locations", "Report", "Selection log", "Run pipeline", "Export"])

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

with tab_profile:
    if selected_run is None:
        st.info("No location profiles yet. Use the **Run pipeline** tab (or the CLI) to build one.")
    else:
        run = runs.set_index("scoring_run_id").loc[selected_run]
        table = ui_data.criteria_table(conn, selected_run)
        overall = ui_data.overall_with_sliders(table, sliders)
        st.header(f"{run['location']} ({run['country']})")
        st.caption(f"Period {run['period']} · corpus {run['corpus_id']} · {run['model']} repetition "
                   f"{run['repetition']} · scoring run {selected_run}")

        cols = st.columns(1 + len(FACTORS))
        cols[0].metric("Overall score", "n/a" if overall.score is None else f"{overall.score:+.2f}",
                       help="Weighted mean over criteria with evidence (SPEC 9.7)")
        for col, f in zip(cols[1:], FACTORS):
            s = overall.factor_scores.get(f)
            col.metric(f.replace(" factors", ""), "—" if s is None else f"{s:+.2f}")

        flags = ui_data.red_flags(conn, selected_run)
        if flags:
            st.subheader(f"Red flags ({len(flags)})")
            for f in flags:
                with st.container(border=True):
                    st.error(f"**{f['rule']} · {f['main_id']} {f['criterion']}** — {f['description']}")
                    for e in f["evidence"].itertuples():
                        st.markdown(f"> {e.sentence}  \n[{e.evidence}] {e.outlet}, {e.date} — [{e.title}]({e.url})")

        st.subheader("Criteria by factor")
        for factor in FACTORS:
            part = table[table["factor"] == factor]
            st.markdown(f"**{factor}**")
            st.dataframe(styled(part.drop(columns=["factor"])), hide_index=True, width="stretch")

        st.subheader("Drill-down")
        evidenced = table[table["evidence"] > 0]
        if evidenced.empty:
            st.caption("No evidence in this run.")
        for r in evidenced.itertuples():
            thin = " · thin evidence" if r.status == "thin" else ""
            label = f"{r.main_id} {r.criterion} — {r.score:+.1f} {r.rating}{thin} · trend {r.trend} · {r.evidence} evidence"
            with st.expander(label):
                ev = ui_data.evidence_table(conn, selected_run, main_id=r.main_id)
                for sub, group in ev.groupby("sub_criterion", sort=True):
                    st.markdown(f"**{sub}**")
                    for e in group.itertuples():
                        review = " · needs review" if e.needs_review else ""
                        st.markdown(
                            f"- **{e.evidence}** direction {e.direction:+d}, intensity {e.intensity}, {e.type}, "
                            f"confidence {e.confidence:.2f}{review}  \n  > {e.sentence}  \n"
                            f"  {e.outlet}, {e.date} — [{e.title}]({e.url})")

# ---------------------------------------------------------------------------
# Compare locations
# ---------------------------------------------------------------------------

with tab_compare:
    if runs.empty:
        st.info("No location profiles yet.")
    else:
        titles = {int(r["scoring_run_id"]): ui_data.run_title(r) for r in runs.to_dict("records")}
        chosen = st.multiselect("Profiles to compare", list(titles), default=list(titles)[:min(3, len(titles))],
                                format_func=titles.get)
        if chosen:
            st.subheader("Ranking (current weights)")
            st.dataframe(ui_data.ranking(conn, chosen, sliders), hide_index=True, width="stretch")
            st.subheader("Criteria side by side")
            st.dataframe(ui_data.comparison(conn, chosen), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

with tab_report:
    if selected_run is None:
        st.info("No location profile selected.")
    else:
        rep = ui_data.latest_report(conn, selected_run)
        if rep is None:
            st.info("No report for this profile yet.")
        elif rep["passed"]:
            st.success(f"Report passed validation after {rep['attempts']} attempt(s).")
            st.markdown(rep["text"])
        else:
            st.warning(f"The report failed validation after {rep['attempts']} attempts and is not shown. "
                       "The profile in the Profile tab remains valid.")
            last = json.loads(rep["checks_json"])[-1]["checks"]
            st.dataframe(pd.DataFrame([{"rule": c["rule"], "violation": v} for c in last for v in c["violations"]]),
                         hide_index=True)
        if st.button("Generate report", help="LLM writes the report from the profile; validated with retries"):
            with st.status("Writing report …", expanded=True) as s:
                rid = pipeline.report(conn, cfg, selected_run, progress=s.write)
                s.update(label=f"Report {rid} done", state="complete")
            st.rerun()

# ---------------------------------------------------------------------------
# Selection log (Stage 0)
# ---------------------------------------------------------------------------

with tab_log:
    corp = ui_data.corpora(conn)
    if corp.empty:
        st.info("No corpora yet.")
    else:
        default = 0
        if selected_run is not None:
            cid = int(runs.set_index("scoring_run_id").loc[selected_run, "corpus_id"])
            default = corp.index[corp["corpus_id"] == cid][0]
        cid = st.selectbox("Corpus", corp["corpus_id"].tolist(), index=int(default),
                           format_func=lambda c: f"#{c} {corp.set_index('corpus_id').loc[c, 'location']} "
                                                 f"({corp.set_index('corpus_id').loc[c, 'note'] or 'no note'})")
        summary, articles = ui_data.selection_log_tables(conn, cid)
        c1, c2 = st.columns([1, 2])
        c1.dataframe(summary, hide_index=True, width="stretch")
        c2.dataframe(articles[["article_id", "selected", "exclusion_reason", "relevance", "published_at", "outlet",
                               "title", "url"]] if not articles.empty else articles,
                     hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("url")})
        d1, d2 = st.columns(2)
        d1.download_button("Selection log (CSV)", ui_data.csv_bytes(summary), f"selection_log_corpus{cid}.csv")
        d2.download_button("Articles (CSV)", ui_data.csv_bytes(articles), f"selection_articles_corpus{cid}.csv")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

with tab_run:
    st.caption("Each step makes real model / search calls and is logged in call_log. "
               "Steps can also be run from the CLI.")
    st.subheader("1 · Build corpus (Stage 0)")
    with st.form("build"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Location name", cfg.location.name)
        region = c2.text_input("Region", cfg.location.region or "")
        country = c3.text_input("Country", cfg.location.country)
        c4, c5, c6 = st.columns(3)
        lang = c4.text_input("Local language (ISO code)", cfg.location.local_language)
        aliases = c5.text_input("Aliases (comma-separated)", ", ".join(cfg.location.aliases))
        years = c6.number_input("Years back", 1, 20, cfg.period.years_back)
        c7, c8 = st.columns(2)
        run_date = c7.date_input("Run date", cfg.period.run_date or date.today())
        only = c8.text_input("Only these criteria (test run, e.g. H02,H20)", "")
        go = st.form_submit_button("Build corpus")
    if go:
        run_cfg = pipeline.with_location(cfg, name, country, lang, region,
                                         [a.strip() for a in aliases.split(",") if a.strip()], int(years), run_date)
        ids = [x.strip().upper() for x in only.split(",") if x.strip()] or None
        with st.status("Building corpus …", expanded=True) as s:
            cid = pipeline.build_corpus(conn, run_cfg, ids, note=f"UI run{' ' + ','.join(ids) if ids else ''}",
                                        progress=s.write)
            s.update(label=f"Corpus {cid} built — review it in the Selection log tab", state="complete")

    st.subheader("2 · Freeze corpus")
    corp = ui_data.corpora(conn)
    open_corp = corp[~corp["frozen"]] if not corp.empty else corp
    if open_corp.empty:
        st.caption("No unfrozen corpus.")
    else:
        fc = st.selectbox("Corpus to freeze", open_corp["corpus_id"].tolist(),
                          format_func=lambda c: f"#{c} {open_corp.set_index('corpus_id').loc[c, 'location']} "
                                                f"({open_corp.set_index('corpus_id').loc[c, 'selected']} selected)")
        if st.button("Freeze (irreversible)"):
            n = pipeline.freeze_corpus(conn, cfg, int(fc))
            st.success(f"Corpus {fc} frozen with {n} articles.")

    st.subheader("3 · Extract evidence (Stage 1)")
    frozen = corp[corp["frozen"]] if not corp.empty else corp
    if frozen.empty:
        st.caption("No frozen corpus.")
    else:
        with st.form("extract"):
            ec = st.selectbox("Frozen corpus", frozen["corpus_id"].tolist(),
                              format_func=lambda c: f"#{c} {frozen.set_index('corpus_id').loc[c, 'location']}")
            model = st.selectbox("Decision model", ["llm_structured", "openai_decisions", "jev"],
                                 index=["llm_structured", "openai_decisions", "jev"].index(cfg.models.decision_provider))
            rep_n = st.number_input("Repetitions", 1, 10, 1)
            go_x = st.form_submit_button("Extract")
        if go_x:
            with st.status("Extracting …", expanded=True) as s:
                ids = pipeline.extract(conn, cfg, int(ec), model, int(rep_n), progress=s.write)
                s.update(label=f"Extraction run(s) {ids} done", state="complete")

    st.subheader("4 · Score (Stage 2)")
    xr = ui_data.extraction_runs(conn)
    if xr.empty:
        st.caption("No extraction runs.")
    else:
        with st.form("score"):
            er = st.selectbox("Extraction run", xr["extraction_run_id"].tolist(),
                              format_func=lambda i: (lambda r: f"#{i} corpus {r['corpus_id']} {r['model']} rep "
                                                               f"{r['repetition']} ({r['evidence']} evidence)")(
                                  xr.set_index("extraction_run_id").loc[i]))
            label = st.text_input("Label", "")
            go_s = st.form_submit_button("Score")
        if go_s:
            sid, _ = pipeline.score(conn, cfg, int(er), label=label or None)
            st.success(f"Scoring run {sid} created — select it in the sidebar.")
            st.rerun()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

with tab_export:
    if selected_run is None:
        st.info("No location profile selected.")
    else:
        md = ui_data.profile_markdown(conn, selected_run, sliders)
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Profile + report (Markdown)", md.encode("utf-8"), f"profile_scoring{selected_run}.md")
        c2.download_button("Profile + report (PDF)", ui_data.markdown_to_pdf(md), f"profile_scoring{selected_run}.pdf")
        c3.download_button("Criteria (CSV)", ui_data.csv_bytes(ui_data.criteria_table(conn, selected_run)),
                           f"criteria_scoring{selected_run}.csv")
        c4.download_button("Evidence (CSV)", ui_data.csv_bytes(ui_data.evidence_table(conn, selected_run)),
                           f"evidence_scoring{selected_run}.csv")
        st.markdown(md)
