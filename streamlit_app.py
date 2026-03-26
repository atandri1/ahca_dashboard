from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ahca_dashboard import aggregations as agg  # noqa: E402
from ahca_dashboard.config import TAG_DESCRIPTIONS, TARGET_TAGS  # noqa: E402
from ahca_dashboard.io import load_all_regions_from_excels, load_processed_csv  # noqa: E402
from ahca_dashboard.transform import add_citation_id, add_derived_columns, filter_to_min_year  # noqa: E402
from ahca_dashboard.viz import (  # noqa: E402
    fig_pval_heatmap,
    fig_avg_word_count_region_severity,
    fig_monthly_by_region,
    fig_monthly_trend,
    fig_region_counts,
    fig_severity_by_region,
    fig_severity_pct_by_region,
    fig_severity_heatmap,
    fig_severity_violin,
    fig_state_counts,
    fig_tag_counts,
    fig_tag_region_heatmap,
    fig_tag_severity,
    fig_word_count_box_by_region,
    fig_word_count_box_by_severity,
)
from ahca_dashboard import severity as sev  # noqa: E402
from ahca_dashboard.alignment import similarity_interpretation, state_similarity_zscores  # noqa: E402


st.set_page_config(page_title="AHCA Citations Dashboard", layout="wide")

st.title("AHCA Citations Dashboard")

DEFAULT_PROCESSED_CSV = REPO_ROOT / "data" / "processed" / "analysis_dataset_jkl_top10tags.csv"
DEMO_PROCESSED_CSV = REPO_ROOT / "data" / "demo" / "analysis_dataset_demo.csv"


@st.cache_data(show_spinner=False)
def load_dataset_from_processed(csv_path: str, demo_csv_path: str) -> tuple[pd.DataFrame, str | None]:
    df = load_processed_csv(Path(csv_path))
    dataset_note = None
    if df.empty:
        demo_path = Path(demo_csv_path)
        demo_df = load_processed_csv(demo_path)
        if not demo_df.empty:
            df = demo_df
            dataset_note = f"Processed dataset not found at `{csv_path}`. Loaded bundled demo dataset from `{demo_path}` instead."
    if df.empty:
        return df, dataset_note
    if "inspection_year" not in df.columns or "inspection_month" not in df.columns or "word_count" not in df.columns:
        df = add_derived_columns(df)
    df = add_citation_id(df)
    return df, dataset_note


@st.cache_data(show_spinner=False)
def load_dataset_from_raw_excel(input_dir: str, min_year: int) -> pd.DataFrame:
    df_all = load_all_regions_from_excels(Path(input_dir))
    if df_all.empty:
        return df_all
    df_all = add_derived_columns(df_all)
    df_all = add_citation_id(df_all)
    df_all = filter_to_min_year(df_all, min_year=min_year)
    return df_all


with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", options=["Processed CSV", "Raw Excel folder"], index=0)
    min_year = st.number_input("Min year", min_value=2000, max_value=2100, value=2022, step=1)

    if mode == "Processed CSV":
        default_csv = str(DEFAULT_PROCESSED_CSV.relative_to(REPO_ROOT))
        csv_path = st.text_input("CSV path", value=default_csv)
        st.caption("Note: the bundled processed CSV includes only the top 10 J/K/L deficiency tags.")
        st.caption("If that file is missing, the app falls back to a bundled demo processed CSV.")
    else:
        default_raw = str(Path("data/raw"))
        raw_dir = st.text_input("Excel folder", value=default_raw)

    st.divider()
    st.header("Filters")
    only_text_available = st.checkbox("Text available only", value=False)


if mode == "Processed CSV":
    df, processed_note = load_dataset_from_processed(csv_path, str(DEMO_PROCESSED_CSV))
    if df.empty:
        st.error("Processed dataset not found or empty.")
        st.info("Run: `python scripts/build_analysis_dataset.py --input data/raw --output data/processed`")
        st.stop()
    if processed_note:
        with st.sidebar:
            st.warning(processed_note)
else:
    df = load_dataset_from_raw_excel(raw_dir, min_year=min_year)
    if df.empty:
        st.error("No Excel files found.")
        st.info("Point the folder to where the `*_cms_reg*.xlsx` files live.")
        st.stop()


# Ensure derived columns/IDs exist even if the CSV was produced elsewhere.
if "inspection_year" not in df.columns or "inspection_month" not in df.columns or "word_count" not in df.columns:
    df = add_derived_columns(df)
df = add_citation_id(df)


def _sorted_unique(series: pd.Series) -> list:
    return sorted([x for x in series.dropna().unique().tolist()])


with st.sidebar:
    years = _sorted_unique(df["inspection_year"]) if "inspection_year" in df.columns else []
    if years:
        year_values = sorted({int(y) for y in years if pd.notna(y)})
        if len(year_values) == 1:
            year_min = year_values[0]
            year_max = year_values[0]
            st.caption(f"Inspection year: {year_values[0]}")
        else:
            year_min, year_max = st.slider(
                "Inspection year",
                min_value=year_values[0],
                max_value=year_values[-1],
                value=(year_values[0], year_values[-1]),
            )
    else:
        year_min, year_max = (None, None)

    regions = _sorted_unique(df["cms_region"]) if "cms_region" in df.columns else []
    selected_regions = st.multiselect("CMS regions", options=regions, default=regions)

    severities = _sorted_unique(df["scope_severity"]) if "scope_severity" in df.columns else []
    selected_severities = st.multiselect("Severity", options=severities, default=severities)

    tags = _sorted_unique(df["deficiency_tag"]) if "deficiency_tag" in df.columns else []
    selected_tags = st.multiselect("Tags", options=tags, default=tags if tags else [])


df_f = df.copy()
if year_min is not None and "inspection_year" in df_f.columns:
    df_f = df_f[(df_f["inspection_year"] >= year_min) & (df_f["inspection_year"] <= year_max)]
if selected_regions and "cms_region" in df_f.columns:
    df_f = df_f[df_f["cms_region"].isin(selected_regions)]
if selected_severities and "scope_severity" in df_f.columns:
    df_f = df_f[df_f["scope_severity"].isin(selected_severities)]
if selected_tags and "deficiency_tag" in df_f.columns:
    df_f = df_f[df_f["deficiency_tag"].isin(selected_tags)]
if only_text_available and "text_available" in df_f.columns:
    df_f = df_f[df_f["text_available"]]


col1, col2, col3, col4 = st.columns(4)
col1.metric("Citations", f"{len(df_f):,}")
col2.metric("Facilities", f"{df_f['facility_id'].nunique():,}" if "facility_id" in df_f.columns else "n/a")
col3.metric("States", f"{df_f['state'].nunique():,}" if "state" in df_f.columns else "n/a")
if "text_available" in df_f.columns and len(df_f):
    col4.metric("Text Available %", f"{(df_f['text_available'].mean()*100):.1f}%")
else:
    col4.metric("Text Available %", "n/a")


tab_overview, tab_severity, tab_tags, tab_text, tab_time, tab_alignment, tab_drilldown, tab_tables = st.tabs(
    [
        "Overview",
        "Severity",
        "Tags",
        "Text",
        "Time & State",
        "Regulatory Alignment",
        "Citation Analysis",
        "Tables",
    ]
)

with tab_overview:
    rc = agg.region_counts(df_f)
    sev_counts = agg.severity_by_region(df_f)
    sev_pct = agg.severity_pct_by_region(df_f)

    a, b = st.columns(2)
    with a:
        st.plotly_chart(fig_region_counts(rc), width="stretch")
    with b:
        st.plotly_chart(fig_severity_by_region(sev_counts), width="stretch")
    st.plotly_chart(fig_severity_pct_by_region(sev_pct), width="stretch")

with tab_tags:
    pivot = agg.pivot_tag_region(df_f)
    tc = agg.tag_counts(df_f)
    ts = agg.tag_severity(df_f)

    st.caption("Tag descriptions")
    st.dataframe(
        pd.DataFrame(
            [{"tag": t, "label": f"F-0{t}", "description": TAG_DESCRIPTIONS.get(t, "")} for t in TARGET_TAGS]
        ),
        width="stretch",
        hide_index=True,
    )

    a, b = st.columns(2)
    with a:
        st.plotly_chart(fig_tag_region_heatmap(pivot), width="stretch")
    with b:
        st.plotly_chart(fig_tag_counts(tc), width="stretch")
    st.plotly_chart(fig_tag_severity(ts), width="stretch")

with tab_text:
    df_text = df_f[df_f["text_available"]].copy() if "text_available" in df_f.columns else df_f.iloc[0:0]
    a, b = st.columns(2)
    with a:
        st.plotly_chart(fig_word_count_box_by_region(df_text), width="stretch")
    with b:
        st.plotly_chart(fig_word_count_box_by_severity(df_text), width="stretch")

    avg_wc = agg.avg_word_count_region_severity(df_f)
    st.plotly_chart(fig_avg_word_count_region_severity(avg_wc), width="stretch")

with tab_time:
    monthly = agg.monthly_counts(df_f)
    monthly_reg = agg.monthly_by_region(df_f)
    st.plotly_chart(fig_monthly_trend(monthly), width="stretch")
    st.plotly_chart(fig_monthly_by_region(monthly_reg, top_n=4), width="stretch")

    sc = agg.state_counts(df_f)
    st.plotly_chart(fig_state_counts(sc, top_n=30), width="stretch")

with tab_tables:
    st.subheader("Regional Summary")
    st.dataframe(agg.regional_summary_statistics(df_f), width="stretch", hide_index=True)

    st.subheader("Tag Summary")
    st.dataframe(agg.tag_summary_statistics(df_f), width="stretch", hide_index=True)

with tab_severity:
    st.caption("Uses the currently loaded dataset and sidebar filters. For a full all-severity view, load raw Excel and select all severities.")

    df_sev = sev.add_scope_numeric(df_f)
    df_sev_plot = df_sev.dropna(subset=["cms_region", "scope_map"]) if not df_sev.empty else df_sev

    a, b = st.columns(2)
    with a:
        st.plotly_chart(fig_severity_violin(df_sev_plot), width="stretch")
    with b:
        table = sev.severity_region_table(df_sev, severity_order=tuple("ABCDEFGHIJKL"), normalize_by_region=True)
        st.plotly_chart(fig_severity_heatmap(table, title="Heatmap of Severity Distribution by Region (Normalized)"), width="stretch")

    kw = sev.kruskal_wallis_by_region(df_sev)
    if kw is None:
        st.info("Install `scipy` to enable Kruskal-Wallis and Mann-Whitney tests.")
    else:
        h_stat, p_value = kw
        c1, c2 = st.columns(2)
        c1.metric("Kruskal-Wallis H", f"{h_stat:.3f}")
        c2.metric("p-value", f"{p_value:.3g}")

        pvals = sev.pairwise_mannwhitney_fdr(df_sev)
        if not pvals.empty:
            st.plotly_chart(fig_pval_heatmap(pvals), width="stretch")

with tab_alignment:
    st.caption("Compares citation text to Appendix PP baseline text for the same F-tag using cosine similarity (TF-IDF or optional embeddings).")

    baseline_default = str(Path("data/processed/appendix_pp_ftags.csv"))
    baseline_path = st.text_input("Appendix PP baseline CSV", value=baseline_default, key="baseline_csv_path")
    method = st.selectbox(
        "Similarity method",
        options=[
            "TF-IDF (word importance / keyword overlap; fast)",
            "Sentence Transformers (semantic meaning / paraphrase-aware; slower)",
        ],
        index=0,
        help=(
            "TF-IDF emphasizes important words and rewards exact keyword overlap. "
            "Sentence Transformers compares semantic meaning (better for paraphrases/synonyms) but is slower."
        ),
        key="alignment_similarity_method",
    )
    if method.startswith("TF-IDF"):
        st.caption("TF-IDF: emphasizes word importance and exact term overlap; may miss paraphrases.")
    else:
        st.caption("Sentence Transformers: compares semantic meaning and is more robust to paraphrases/synonyms; slower to compute.")

    baseline_file = Path(baseline_path)

    cache_path = Path("data/processed/regulatory_alignment_cache.csv")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    @st.cache_data(show_spinner=False)
    def _load_baseline_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @st.cache_data(show_spinner=False)
    def _sha256_for_file(path: str, mtime_ns: int, size: int) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    baseline_df = None
    baseline_sha256 = None
    if not baseline_file.exists():
        st.info(
            "Generate the baseline file with: `python scripts/build_appendixpp_baseline.py --pdf /path/to/SOM-Appendix-PP.pdf --output data/processed`"
        )
    else:
        baseline_df = _load_baseline_csv(str(baseline_file))
        baseline_stat = baseline_file.stat()
        baseline_sha256 = _sha256_for_file(str(baseline_file), baseline_stat.st_mtime_ns, baseline_stat.st_size)

    compute_clicked = st.button(
        "Compute similarity for current filtered dataset",
        disabled=(baseline_df is None or baseline_sha256 is None),
    )
    if compute_clicked:
        with st.spinner("Computing similarity..."):
            progress_bar = st.progress(0)
            progress_msg = st.empty()

            def _set_progress(pct: int, msg: str) -> None:
                pct = max(0, min(100, int(pct)))
                progress_bar.progress(pct)
                progress_msg.caption(msg)

            _set_progress(0, "Starting...")
            _set_progress(3, "Preparing filtered dataset...")
            df_align = df_f.copy()
            df_align = add_citation_id(df_align)
            valid_id_mask = df_align["citation_id"].notna() if "citation_id" in df_align.columns else pd.Series(False, index=df_align.index)
            df_work = df_align.loc[valid_id_mask].copy()

            _set_progress(6, "Selecting similarity method...")
            try:
                if method.startswith("TF-IDF"):
                    from ahca_dashboard.alignment import compute_tfidf_similarity  # optional dependency

                    sim_col = "tfidf_sim"
                    method_key = "tfidf_min_df2_ngram12"
                    compute_fn = compute_tfidf_similarity
                else:
                    from ahca_dashboard.alignment import compute_sentence_transformer_similarity  # optional dependency

                    sim_col = "sentrans_sim"
                    method_key = "sentrans_all-MiniLM-L6-v2"
                    compute_fn = compute_sentence_transformer_similarity
            except Exception as e:
                st.error(str(e))
                st.info("Install dependencies: `pip install -r requirements.txt`")
                st.stop()

            # Load disk cache (if any) and compute only missing citation_ids.
            _set_progress(10, "Loading similarity cache...")
            if cache_path.exists():
                try:
                    cache_df = pd.read_csv(cache_path)
                except Exception:
                    cache_df = pd.DataFrame(columns=["baseline_sha256", "method_key", "citation_id", "similarity"])
            else:
                cache_df = pd.DataFrame(columns=["baseline_sha256", "method_key", "citation_id", "similarity"])

            if not cache_df.empty:
                cache_df = cache_df.dropna(subset=["citation_id"])

            cache_key_df = cache_df[
                (cache_df["baseline_sha256"] == baseline_sha256) & (cache_df["method_key"] == method_key)
            ]
            cached_ids = set(cache_key_df["citation_id"].astype(str).tolist()) if not cache_key_df.empty else set()

            missing_mask = ~df_work["citation_id"].astype(str).isin(cached_ids)
            df_missing = df_work.loc[missing_mask].copy()

            computed = 0
            if not df_missing.empty:
                _set_progress(15, f"Computing similarity for {len(df_missing):,} citations...")
                if method.startswith("TF-IDF"):
                    sim_missing = compute_fn(df_missing, baseline_df)
                else:
                    def _cb(frac: float) -> None:
                        _set_progress(15 + int(frac * 75), f"Computing semantic similarity... {int(frac * 100)}%")

                    sim_missing = compute_fn(df_missing, baseline_df, progress_callback=_cb)
                new_cache = pd.DataFrame(
                    {
                        "baseline_sha256": baseline_sha256,
                        "method_key": method_key,
                        "citation_id": df_missing["citation_id"].astype(str).to_numpy(),
                        "similarity": pd.to_numeric(sim_missing, errors="coerce").to_numpy(),
                    }
                )
                _set_progress(92, "Writing to cache...")
                cache_df = pd.concat([cache_df, new_cache], ignore_index=True)
                cache_df = cache_df.drop_duplicates(["baseline_sha256", "method_key", "citation_id"], keep="last")
                cache_df.to_csv(cache_path, index=False)
                computed = int(len(df_missing))
            else:
                _set_progress(85, "All citations already in cache. Attaching results...")

            # Rebuild the mapping from the updated cache and attach to df_align.
            _set_progress(96, "Attaching similarity results...")
            cache_key_df = cache_df[
                (cache_df["baseline_sha256"] == baseline_sha256) & (cache_df["method_key"] == method_key)
            ]
            sim_map = (
                cache_key_df.set_index("citation_id")["similarity"].to_dict() if not cache_key_df.empty else {}
            )
            df_align[sim_col] = pd.NA
            df_align.loc[valid_id_mask, sim_col] = df_align.loc[valid_id_mask, "citation_id"].astype(str).map(sim_map)

            loaded = int((~missing_mask).sum())
            _set_progress(100, "Done.")
            st.success(f"Loaded {loaded:,} from cache, computed {computed:,} new. Cache: {cache_path}")

            st.session_state["df_align"] = df_align
            st.session_state["sim_col"] = sim_col

    df_align = st.session_state.get("df_align")
    sim_col = st.session_state.get("sim_col", "tfidf_sim")
    if df_align is None or sim_col not in df_align.columns:
        st.warning("Click the button above to compute similarity for the current filtered dataset. Results are cached to disk to avoid recomputation.")
    else:
        sim_series = pd.to_numeric(df_align[sim_col], errors="coerce")
        st.write(sim_series.describe())

        import plotly.express as px
        import plotly.graph_objects as go

        hist = px.histogram(df_align, x=sim_col, nbins=40, title=f"Similarity Distribution ({method})")
        st.plotly_chart(hist, width="stretch")

        if "cms_region" in df_align.columns:
            region_stats = (
                df_align.assign(_sim=sim_series)
                .dropna(subset=["cms_region", "_sim"])
                .groupby("cms_region")["_sim"]
                .agg(["count", "mean", "std"])
                .reset_index()
                .sort_values("mean")
            )
            fig = go.Figure(
                data=go.Bar(
                    x=region_stats["mean"],
                    y=region_stats["cms_region"].astype(str),
                    orientation="h",
                    error_x={"type": "data", "array": region_stats["std"].fillna(0.0)},
                )
            ).update_layout(
                title="Mean Similarity by CMS Region (+/- 1 SD)",
                xaxis_title="Mean Similarity",
                yaxis_title="CMS Region",
                xaxis={"range": [0, 1]},
            )
            st.plotly_chart(fig, width="stretch")

        if "state" in df_align.columns:
            st.subheader("State Outliers (Z-scores)")
            df_tmp = df_align.copy()
            df_tmp[sim_col] = sim_series
            state_stats = state_similarity_zscores(df_tmp, sim_col=sim_col)
            if not state_stats.empty:
                plot_height = max(450, min(2200, 18 * len(state_stats) + 200))

                states_by_mean = state_stats.sort_values("mean")
                fig_mean = go.Figure(
                    data=go.Bar(
                        x=states_by_mean["mean"],
                        y=states_by_mean["state"],
                        orientation="h",
                        error_x={"type": "data", "array": states_by_mean["std"].fillna(0.0)},
                    )
                ).update_layout(
                    title="State-Level Mean Similarity (+/- 1 SD)",
                    xaxis_title="Mean Similarity",
                    yaxis_title="State",
                    xaxis={"range": [0, 1]},
                    height=plot_height,
                )
                st.plotly_chart(fig_mean, width="stretch")

                fig = px.bar(
                    state_stats,
                    x="z_score",
                    y="state",
                    orientation="h",
                    title="State Deviation from National Mean (Z-score)",
                    height=plot_height,
                )
                st.plotly_chart(fig, width="stretch")
                low = state_stats.dropna(subset=["z_score"]).head(3)
                if not low.empty:
                    st.write("Lowest-alignment states (largest negative deviations):")
                    st.write(low[["state", "count", "mean", "z_score"]])
                st.dataframe(state_stats, width="stretch", hide_index=True)

        if "deficiency_tag" in df_align.columns and "cms_region" in df_align.columns:
            st.subheader("Mean Similarity by Region and Tag")
            heat = (
                df_align.assign(_sim=sim_series)
                .dropna(subset=["deficiency_tag", "cms_region", "_sim"])
                .groupby(["deficiency_tag", "cms_region"])["_sim"]
                .mean()
                .unstack()
            )
            if not heat.empty:
                y_tags = [str(i) for i in heat.index.tolist()]
                x_regions = [str(c) for c in heat.columns.tolist()]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=heat.values,
                        x=x_regions,
                        y=y_tags,
                        colorscale="Viridis",
                        zmin=0,
                        zmax=1,
                        colorbar={"title": "Mean sim"},
                    )
                ).update_layout(
                    title="Mean Similarity by CMS Region and Tag",
                    xaxis_title="CMS Region",
                    yaxis_title="Deficiency Tag",
                )
                # Ensure equal spacing between tags (categorical axis), even though the labels are numeric-looking.
                fig.update_yaxes(type="category", categoryorder="array", categoryarray=y_tags)
                fig.update_xaxes(type="category", categoryorder="array", categoryarray=x_regions)
                st.plotly_chart(fig, width="stretch")

        if "state" in df_align.columns and "deficiency_tag" in df_align.columns:
            st.subheader("Mean Similarity by State and Tag")
            top_tags_n = st.slider(
                "Top tags (by volume) to display", min_value=5, max_value=20, value=10, step=1
            )

            all_states = sorted([str(s) for s in df_align["state"].dropna().unique().tolist()])
            top_tags = df_align["deficiency_tag"].value_counts().head(int(top_tags_n)).index.tolist()
            df_top = df_align[df_align["deficiency_tag"].isin(top_tags)].copy()

            heat_state = (
                df_top.assign(_sim=sim_series.loc[df_top.index])
                .dropna(subset=["state", "deficiency_tag", "_sim"])
                .groupby(["deficiency_tag", "state"])["_sim"]
                .mean()
                .unstack()
            )
            if not heat_state.empty:
                heat_state = heat_state.reindex(columns=all_states)
            if not heat_state.empty:
                y_tags = [str(i) for i in heat_state.index.tolist()]
                x_states = [str(c) for c in heat_state.columns.tolist()]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=heat_state.values,
                        x=x_states,
                        y=y_tags,
                        colorscale="Viridis",
                        zmin=0,
                        zmax=1,
                        colorbar={"title": "Mean sim"},
                    )
                ).update_layout(
                    title="Mean Similarity by State and Tag",
                    xaxis_title="State",
                    yaxis_title="Deficiency Tag",
                )
                fig.update_yaxes(type="category", categoryorder="array", categoryarray=y_tags)
                fig.update_xaxes(type="category", categoryorder="array", categoryarray=x_states)
                st.plotly_chart(fig, width="stretch")

with tab_drilldown:
    st.caption("Analyze a single citation (by citation number) and compare its inspection text to the Appendix PP baseline text for its F-tag.")

    df_align = st.session_state.get("df_align")
    sim_col = st.session_state.get("sim_col", "tfidf_sim")

    baseline_path = st.session_state.get("baseline_csv_path", str(Path("data/processed/appendix_pp_ftags.csv")))
    baseline_file = Path(baseline_path)
    baseline_df = None
    if baseline_file.exists():
        try:
            baseline_df = _load_baseline_csv(str(baseline_file))
        except Exception as e:
            st.warning(f"Failed to read baseline CSV: {e}")
    else:
        st.info("Appendix PP baseline CSV not found. Set the path in the Regulatory Alignment tab to enable baseline text in citation analysis.")

    if df_align is None or sim_col not in df_align.columns:
        st.warning("Compute similarity in the Regulatory Alignment tab first to enable citation analysis.")
    else:
        st.subheader("Citation Analysis")
        st.caption(
            "Citation number format: `<facility_id>_<deficiency_tag>_<cms_region>_<n>` "
            "(example: `75011_600_1_0`)."
        )

        def _normalize_citation_number(s: str) -> str:
            s = str(s or "").strip()
            parts = s.split("_")
            if len(parts) < 4:
                return s

            def _normalize_numeric_token(tok: str) -> str:
                tok = str(tok or "").strip()
                if not tok:
                    return tok

                # Common user inputs for tags: "F0688", "F-0688"
                if tok.upper().startswith("F"):
                    tok = tok[1:].lstrip("-").strip()

                try:
                    f = float(tok)
                    if f.is_integer():
                        return str(int(f))
                except Exception:
                    pass

                if tok.endswith(".0"):
                    tok = tok[:-2]
                if tok.isdigit():
                    # Strip leading zeros (e.g., "0688" -> "688")
                    return str(int(tok))
                return tok

            facility = _normalize_numeric_token(parts[0])
            tag = _normalize_numeric_token(parts[1])
            region = _normalize_numeric_token(parts[2])
            n = _normalize_numeric_token(parts[3])
            return "_".join([facility, tag, region, n])

        example_id = ""
        if "citation_id" in df_align.columns:
            non_null_ids = df_align["citation_id"].dropna()
            if len(non_null_ids):
                example_id = _normalize_citation_number(str(non_null_ids.iloc[0]))

        citation_number = st.text_input(
            "Citation number",
            value=example_id,
            help="Format: <facility_id>_<deficiency_tag>_<cms_region>_<n> (example: 75011_600_1_0)",
            key="drilldown_citation_number",
        ).strip()

        citation_number_norm = _normalize_citation_number(citation_number)
        norm_to_orig: dict[str, str] = {}
        if "citation_id" in df_align.columns:
            for cid in df_align["citation_id"].dropna().astype(str).unique().tolist():
                norm_to_orig[_normalize_citation_number(cid)] = cid

        orig_id = norm_to_orig.get(citation_number_norm)
        matches = df_align[df_align["citation_id"].astype(str) == orig_id] if orig_id else df_align.iloc[0:0]
        if matches.empty:
            st.error("Citation not found in the current computed alignment dataset.")
        else:
            row = matches.iloc[0]

            tag_num = row.get("deficiency_tag")
            tag_int = int(tag_num) if pd.notna(tag_num) else None

            baseline_text = None
            if baseline_df is not None and tag_int is not None and "deficiency_tag" in baseline_df.columns:
                match = baseline_df[pd.to_numeric(baseline_df["deficiency_tag"], errors="coerce") == tag_int]
                if not match.empty and "text" in match.columns:
                    baseline_text = str(match["text"].iloc[0])

            sim_raw = row.get(sim_col)
            sim_val = float(sim_raw) if pd.notna(sim_raw) else float("nan")
            interpretation = similarity_interpretation(sim_val)
            a, b = st.columns(2)
            a.metric("Similarity Score", f"{sim_val:.3f}" if pd.notna(sim_val) else "n/a")
            b.metric("Interpretation", interpretation)
            st.text_area(
                "Inspection Text",
                value=str(row.get("inspection_text", "")),
                height=200,
                key="drilldown_inspection_text",
            )
            st.text_area(
                "Appendix PP Baseline Text",
                value=baseline_text or "Baseline text not found for this tag.",
                height=200,
                key="drilldown_baseline_text",
            )
