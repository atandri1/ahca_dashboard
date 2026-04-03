from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ahca_dashboard import aggregations as agg  # noqa: E402
from ahca_dashboard.config import TAG_DESCRIPTIONS, TARGET_TAGS  # noqa: E402
from ahca_dashboard import deep_analysis as deep  # noqa: E402
from ahca_dashboard.io import load_all_regions_from_excels, load_processed_csv  # noqa: E402
from ahca_dashboard.transform import add_citation_id, add_derived_columns, filter_to_min_year  # noqa: E402
from ahca_dashboard.viz import (  # noqa: E402
    fig_domain_indicator_profile,
    fig_keyword_difference,
    fig_keyword_profile,
    fig_pval_heatmap,
    fig_avg_word_count_region_severity,
    fig_monthly_by_region,
    fig_monthly_trend,
    fig_region_tag_severity_heatmap,
    fig_region_counts,
    fig_regional_divergence,
    fig_severity_by_region,
    fig_severity_pct_by_region,
    fig_severity_heatmap,
    fig_severity_violin,
    fig_state_counts,
    fig_state_league_table,
    fig_tag_counts,
    fig_tag_bifurcation,
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
def load_demo_dataset(demo_csv_path: str) -> pd.DataFrame:
    df = load_processed_csv(Path(demo_csv_path))
    if df.empty:
        return df
    if "inspection_year" not in df.columns or "inspection_month" not in df.columns or "word_count" not in df.columns:
        df = add_derived_columns(df)
    df = add_citation_id(df)
    return df


@st.cache_data(show_spinner=False)
def load_dataset_from_processed(csv_path: str, demo_csv_path: str) -> tuple[pd.DataFrame, str | None]:
    df = load_processed_csv(Path(csv_path))
    dataset_note = None
    if df.empty:
        demo_path = Path(demo_csv_path)
        demo_df = load_demo_dataset(str(demo_path))
        if not demo_df.empty:
            df = demo_df
            dataset_note = f"Processed dataset not found at `{csv_path}`. Loaded bundled demo dataset from `{demo_path}` instead."
    if df.empty:
        return df, dataset_note
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
    raw_note = None
    if df.empty:
        df = load_demo_dataset(str(DEMO_PROCESSED_CSV))
        if df.empty:
            st.error("No Excel files found.")
            st.info("Point the folder to where the `*_cms_reg*.xlsx` files live.")
            st.stop()
        raw_note = f"No Excel files found at `{raw_dir}`. Loaded bundled demo dataset from `{DEMO_PROCESSED_CSV}` instead."
    if raw_note:
        with st.sidebar:
            st.warning(raw_note)


# Ensure derived columns/IDs exist even if the CSV was produced elsewhere.
if "inspection_year" not in df.columns or "inspection_month" not in df.columns or "word_count" not in df.columns:
    df = add_derived_columns(df)
df = add_citation_id(df)


def _sorted_unique(series: pd.Series) -> list:
    return sorted([x for x in series.dropna().unique().tolist()])


def _format_pct(value: float | int | None) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "n/a"


def _excerpt(text: str, limit: int = 800) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


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
        return str(int(tok))
    return tok


def _normalize_citation_number(s: str) -> str:
    s = str(s or "").strip()
    parts = s.split("_")
    if len(parts) < 4:
        return s
    normalized = [_normalize_numeric_token(part) for part in parts[:4]]
    return "_".join(normalized)


def _citation_examples(df_align: pd.DataFrame) -> tuple[str, str]:
    if "citation_id" not in df_align.columns:
        return "", ""

    examples: list[str] = []
    for citation_id in df_align["citation_id"].dropna().astype(str).unique().tolist():
        normalized = _normalize_citation_number(citation_id)
        if normalized and normalized not in examples:
            examples.append(normalized)
        if len(examples) == 2:
            break

    if not examples:
        return "", ""
    if len(examples) == 1:
        return examples[0], examples[0]
    return examples[0], examples[1]


def _citation_lookup(df_align: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if "citation_id" not in df_align.columns:
        return lookup

    for citation_id in df_align["citation_id"].dropna().astype(str).unique().tolist():
        normalized = _normalize_citation_number(citation_id)
        if normalized:
            lookup.setdefault(normalized, citation_id)
    return lookup


def _lookup_citation_row(df_align: pd.DataFrame, citation_number: str, lookup: dict[str, str]) -> pd.Series | None:
    if "citation_id" not in df_align.columns:
        return None
    normalized = _normalize_citation_number(citation_number)
    orig_id = lookup.get(normalized)
    if not orig_id:
        return None
    matches = df_align[df_align["citation_id"].astype(str) == orig_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def _parse_search_terms(search_text: str) -> list[str]:
    if not search_text:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in re.split(r"[,;\n]+", search_text):
        term = str(raw_term or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def _highlight_text_html(text: str, search_terms: list[str]) -> tuple[str, int]:
    raw_text = str(text or "")
    if not search_terms:
        return html.escape(raw_text).replace("\n", "<br>"), 0

    pattern = re.compile("|".join(re.escape(term) for term in sorted(search_terms, key=len, reverse=True)), flags=re.IGNORECASE)
    parts: list[str] = []
    last_end = 0
    matches = 0
    for match in pattern.finditer(raw_text):
        matches += 1
        parts.append(html.escape(raw_text[last_end:match.start()]))
        parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
        last_end = match.end()
    parts.append(html.escape(raw_text[last_end:]))
    return "".join(parts).replace("\n", "<br>"), matches


def _render_highlighted_text(title: str, text: str, search_terms: list[str], *, empty_message: str) -> None:
    display_text = str(text or "")
    if not display_text.strip():
        display_text = empty_message

    text_html, match_count = _highlight_text_html(display_text, search_terms)
    if search_terms:
        st.caption(f"{match_count} highlighted match{'es' if match_count != 1 else ''} in {title.lower()}.")

    st.markdown(
        (
            "<div class='citation-text-panel'>"
            f"<div class='citation-text-label'>{html.escape(title)}</div>"
            f"<div class='citation-text-body'>{text_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _format_context_value(value: object, *, prefix: str = "") -> str:
    if pd.isna(value):
        return "n/a"
    try:
        number = float(value)
        if number.is_integer():
            return f"{prefix}{int(number)}"
    except Exception:
        pass
    return f"{prefix}{value}"


def _default_tag_pair(df_align: pd.DataFrame, sim_col: str) -> tuple[int | None, int | None]:
    available = sorted(
        [int(t) for t in pd.to_numeric(df_align.get("deficiency_tag", pd.Series(dtype="float64")), errors="coerce").dropna().unique().tolist()]
    )
    if 686 in available and 689 in available:
        return 686, 689
    return deep.notebook_default_tags(df_align, sim_col)


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


tab_overview, tab_severity, tab_tags, tab_text, tab_time, tab_alignment, tab_tag_severity, tab_keywords, tab_domain, tab_drilldown, tab_tables = st.tabs(
    [
        "Overview",
        "Severity",
        "Tags",
        "Text",
        "Time & State",
        "Regulatory Alignment",
        "Tag x Severity",
        "Keyword Comparison",
        "Domain Evidence",
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

with tab_tag_severity:
    st.caption(
        "Notebook v2 addition: breaks regulatory alignment down by CMS region, deficiency tag, and J/K/L severity to surface recurring problem combinations."
    )

    df_align = st.session_state.get("df_align")
    sim_col = st.session_state.get("sim_col", "tfidf_sim")

    if df_align is None or sim_col not in df_align.columns:
        st.warning("Compute similarity in the Regulatory Alignment tab first to enable the Tag x Severity analysis.")
    else:
        combo_stats = deep.tag_severity_alignment_by_region(df_align, sim_col)
        tag_rank = deep.tag_ranking(df_align, sim_col)
        severity_rank = deep.severity_ranking(df_align, sim_col)
        region_spread = deep.regional_divergence(combo_stats)
        state_stats = deep.state_alignment_summary(df_align, sim_col)

        if combo_stats.empty:
            st.info("No tag-by-severity combinations are available for the current filtered dataset.")
        else:
            critical_threshold = st.slider(
                "Critical similarity threshold",
                min_value=0.20,
                max_value=0.60,
                value=0.40,
                step=0.01,
                key="tag_severity_critical_threshold",
            )
            critical_df = deep.critical_combinations(combo_stats, threshold=float(critical_threshold))
            worst_combo = combo_stats.nsmallest(1, "alignment_mean").iloc[0]
            best_combo = combo_stats.nlargest(1, "alignment_mean").iloc[0]
            avg_spread = float(region_spread["spread"].mean()) if not region_spread.empty else float("nan")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Worst combo",
                f"R{int(worst_combo['region'])} / F-0{int(worst_combo['tag'])} / {worst_combo['severity']}",
                _format_pct(worst_combo["alignment_mean"]),
            )
            c2.metric(
                "Best combo",
                f"R{int(best_combo['region'])} / F-0{int(best_combo['tag'])} / {best_combo['severity']}",
                _format_pct(best_combo["alignment_mean"]),
            )
            c3.metric("Critical combos", f"{len(critical_df):,}", f"< {_format_pct(critical_threshold)}")
            c4.metric("Avg regional spread", _format_pct(avg_spread))

            left, right = st.columns(2)
            with left:
                st.plotly_chart(fig_tag_bifurcation(tag_rank), width="stretch")
            with right:
                st.plotly_chart(fig_regional_divergence(region_spread), width="stretch")

            detail_cols = st.columns([1.2, 1])
            regions_available = sorted([int(r) for r in combo_stats["region"].dropna().unique().tolist()])
            selected_region = detail_cols[0].selectbox(
                "Region detail",
                options=regions_available,
                index=0,
                key="tag_severity_region_detail",
            )
            region_pivot = deep.region_tag_severity_pivot(combo_stats, int(selected_region))
            with detail_cols[0]:
                st.plotly_chart(fig_region_tag_severity_heatmap(region_pivot, int(selected_region)), width="stretch")
            with detail_cols[1]:
                st.plotly_chart(fig_state_league_table(state_stats), width="stretch")

            table_a, table_b, table_c = st.columns(3)
            with table_a:
                st.subheader("Problem Combos by Region")
                st.dataframe(
                    deep.regional_problem_tags(combo_stats, top_n=3)
                    .rename(
                        columns={
                            "region": "Region",
                            "rank": "Rank",
                            "tag_label": "Tag",
                            "severity": "Severity",
                            "alignment_mean": "Mean similarity",
                            "citation_count": "Citations",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            with table_b:
                st.subheader("Severity Ranking")
                st.dataframe(
                    severity_rank.rename(
                        columns={
                            "severity": "Severity",
                            "severity_name": "Description",
                            "alignment_mean": "Mean similarity",
                            "alignment_std": "Std dev",
                            "citation_count": "Citations",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            with table_c:
                st.subheader("Critical Combos")
                if critical_df.empty:
                    st.info("No region/tag/severity combinations fall below the selected threshold.")
                else:
                    st.dataframe(
                        critical_df.rename(
                            columns={
                                "region": "Region",
                                "tag_label": "Tag",
                                "severity": "Severity",
                                "alignment_mean": "Mean similarity",
                                "alignment_std": "Std dev",
                                "citation_count": "Citations",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

with tab_keywords:
    st.caption(
        "Notebook v2 addition: compares the language of lower- and higher-alignment tags using TF-IDF to identify differentiating words and phrases."
    )

    df_align = st.session_state.get("df_align")
    sim_col = st.session_state.get("sim_col", "tfidf_sim")

    if df_align is None or sim_col not in df_align.columns:
        st.warning("Compute similarity in the Regulatory Alignment tab first to enable keyword comparison.")
    else:
        available_tags = sorted(
            [int(t) for t in pd.to_numeric(df_align.get("deficiency_tag", pd.Series(dtype="float64")), errors="coerce").dropna().unique().tolist()]
        )
        if not available_tags:
            st.info("No tags are available in the current alignment dataset.")
        else:
            default_tag_a, default_tag_b = _default_tag_pair(df_align, sim_col)
            default_tag_a = default_tag_a if default_tag_a in available_tags else available_tags[0]
            default_tag_b = default_tag_b if default_tag_b in available_tags else available_tags[-1]

            k1, k2 = st.columns(2)
            tag_a = k1.selectbox(
                "Lower-alignment tag",
                options=available_tags,
                index=available_tags.index(default_tag_a),
                format_func=lambda x: f"F-0{x}",
                key="keyword_tag_a",
            )
            tag_b = k2.selectbox(
                "Higher-alignment tag",
                options=available_tags,
                index=available_tags.index(default_tag_b),
                format_func=lambda x: f"F-0{x}",
                key="keyword_tag_b",
            )

            if tag_a == tag_b:
                st.info("Choose two different tags to compare their language profiles.")
            else:
                try:
                    keyword_result = deep.keyword_comparison(df_align, sim_col, int(tag_a), int(tag_b))
                except RuntimeError as e:
                    st.info(str(e))
                else:
                    stats = keyword_result["stats"] or {}
                    mean_a = stats.get("tag_a_mean")
                    mean_b = stats.get("tag_b_mean")
                    count_a = stats.get("tag_a_count", 0)
                    count_b = stats.get("tag_b_count", 0)
                    gap = float(mean_b - mean_a) if pd.notna(mean_a) and pd.notna(mean_b) else float("nan")

                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"F-0{int(tag_a)} mean similarity", _format_pct(mean_a), f"{int(count_a):,} citations")
                    m2.metric(f"F-0{int(tag_b)} mean similarity", _format_pct(mean_b), f"{int(count_b):,} citations")
                    m3.metric("Gap", _format_pct(gap))

                    a, b = st.columns(2)
                    with a:
                        st.plotly_chart(
                            fig_keyword_profile(
                                keyword_result["profile_a"],
                                title=f"Top TF-IDF Terms for F-0{int(tag_a)}",
                                color="#b22222",
                            ),
                            width="stretch",
                        )
                    with b:
                        st.plotly_chart(
                            fig_keyword_profile(
                                keyword_result["profile_b"],
                                title=f"Top TF-IDF Terms for F-0{int(tag_b)}",
                                color="#2f855a",
                            ),
                            width="stretch",
                        )

                    st.plotly_chart(
                        fig_keyword_difference(
                            keyword_result["shared_diff"],
                            title=f"Shared Terms with the Largest TF-IDF Advantage for F-0{int(tag_a)}",
                        ),
                        width="stretch",
                    )

                    u1, u2 = st.columns(2)
                    with u1:
                        st.subheader(f"Terms Unique to F-0{int(tag_a)}")
                        st.dataframe(keyword_result["unique_a"], width="stretch", hide_index=True)
                    with u2:
                        st.subheader(f"Terms Unique to F-0{int(tag_b)}")
                        st.dataframe(keyword_result["unique_b"], width="stretch", hide_index=True)

                    st.subheader("Notebook Heuristic: Expansion vs Reduction Terms")
                    st.dataframe(stats.get("pattern_counts", pd.DataFrame()), width="stretch", hide_index=True)

with tab_domain:
    st.caption(
        "Notebook v2 addition: tests the domain-mismatch hypothesis with representative citations, notebook-specific vocabulary families, and Appendix PP baseline text when available."
    )
    st.caption("This framing is most interpretable for the notebook's original comparison of F-0686 vs F-0689.")

    df_align = st.session_state.get("df_align")
    sim_col = st.session_state.get("sim_col", "tfidf_sim")

    if df_align is None or sim_col not in df_align.columns:
        st.warning("Compute similarity in the Regulatory Alignment tab first to enable the domain evidence view.")
    else:
        available_tags = sorted(
            [int(t) for t in pd.to_numeric(df_align.get("deficiency_tag", pd.Series(dtype="float64")), errors="coerce").dropna().unique().tolist()]
        )
        if not available_tags:
            st.info("No tags are available in the current alignment dataset.")
        else:
            default_left, default_right = _default_tag_pair(df_align, sim_col)
            default_left = default_left if default_left in available_tags else available_tags[0]
            default_right = default_right if default_right in available_tags else available_tags[-1]

            d1, d2 = st.columns(2)
            tag_left = d1.selectbox(
                "Tag A",
                options=available_tags,
                index=available_tags.index(default_left),
                format_func=lambda x: f"F-0{x}",
                key="domain_tag_left",
            )
            tag_right = d2.selectbox(
                "Tag B",
                options=available_tags,
                index=available_tags.index(default_right),
                format_func=lambda x: f"F-0{x}",
                key="domain_tag_right",
            )

            if tag_left == tag_right:
                st.info("Choose two different tags to compare their domain evidence.")
            else:
                try:
                    domain_result = deep.keyword_comparison(df_align, sim_col, int(tag_left), int(tag_right))
                except RuntimeError as e:
                    st.info(str(e))
                else:
                    profile_left = domain_result["profile_a"]
                    profile_right = domain_result["profile_b"]
                    domain_counts = pd.concat(
                        [
                            deep.domain_indicator_summary(profile_left, int(tag_left)),
                            deep.domain_indicator_summary(profile_right, int(tag_right)),
                        ],
                        ignore_index=True,
                    )
                    st.plotly_chart(fig_domain_indicator_profile(domain_counts), width="stretch")

                    unique_left = deep.domain_unique_terms(profile_left, profile_right)
                    unique_right = deep.domain_unique_terms(profile_right, profile_left)
                    worst_left, best_left = deep.tag_examples(df_align, sim_col, int(tag_left))
                    worst_right, best_right = deep.tag_examples(df_align, sim_col, int(tag_right))

                    left_baseline = deep.baseline_text_for_tag(baseline_df, int(tag_left)) if baseline_df is not None else ""
                    right_baseline = deep.baseline_text_for_tag(baseline_df, int(tag_right)) if baseline_df is not None else ""

                    card_left, card_right = st.columns(2)
                    with card_left:
                        st.subheader(f"F-0{int(tag_left)}")
                        left_mean = (domain_result["stats"] or {}).get("tag_a_mean")
                        left_note = deep.dominant_domain_note(unique_left)
                        st.metric("Mean similarity", _format_pct(left_mean))
                        st.metric(
                            "Dominant notebook vocabulary",
                            str(left_note["label"]),
                            f"{int(left_note['count'])}/{int(left_note['total_terms'])} unique terms",
                        )
                        st.dataframe(unique_left, width="stretch", hide_index=True)
                        if worst_left is not None:
                            st.text_area(
                                "Lowest-alignment citation",
                                value=_excerpt(str(worst_left.get("inspection_text", ""))),
                                height=180,
                                key=f"domain_worst_{int(tag_left)}",
                            )
                        if best_left is not None:
                            st.text_area(
                                "Highest-alignment citation",
                                value=_excerpt(str(best_left.get("inspection_text", ""))),
                                height=180,
                                key=f"domain_best_{int(tag_left)}",
                            )
                        st.text_area(
                            "Appendix PP baseline text",
                            value=_excerpt(left_baseline) if left_baseline else "Baseline text not found for this tag.",
                            height=180,
                            key=f"domain_baseline_{int(tag_left)}",
                        )
                        if worst_left is not None and left_baseline:
                            st.dataframe(
                                deep.citation_vs_baseline_overlap(str(worst_left.get("inspection_text", "")), left_baseline),
                                width="stretch",
                                hide_index=True,
                            )

                    with card_right:
                        st.subheader(f"F-0{int(tag_right)}")
                        right_mean = (domain_result["stats"] or {}).get("tag_b_mean")
                        right_note = deep.dominant_domain_note(unique_right)
                        st.metric("Mean similarity", _format_pct(right_mean))
                        st.metric(
                            "Dominant notebook vocabulary",
                            str(right_note["label"]),
                            f"{int(right_note['count'])}/{int(right_note['total_terms'])} unique terms",
                        )
                        st.dataframe(unique_right, width="stretch", hide_index=True)
                        if worst_right is not None:
                            st.text_area(
                                "Lowest-alignment citation",
                                value=_excerpt(str(worst_right.get("inspection_text", ""))),
                                height=180,
                                key=f"domain_worst_{int(tag_right)}",
                            )
                        if best_right is not None:
                            st.text_area(
                                "Highest-alignment citation",
                                value=_excerpt(str(best_right.get("inspection_text", ""))),
                                height=180,
                                key=f"domain_best_{int(tag_right)}",
                            )
                        st.text_area(
                            "Appendix PP baseline text",
                            value=_excerpt(right_baseline) if right_baseline else "Baseline text not found for this tag.",
                            height=180,
                            key=f"domain_baseline_{int(tag_right)}",
                        )
                        if worst_right is not None and right_baseline:
                            st.dataframe(
                                deep.citation_vs_baseline_overlap(str(worst_right.get("inspection_text", "")), right_baseline),
                                width="stretch",
                                hide_index=True,
                            )

with tab_drilldown:
    st.caption(
        "Compare two citations side by side using each citation's existing Appendix PP similarity score and interpretation."
    )

    st.markdown(
        """
        <style>
        .citation-text-panel {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
        }
        .citation-panel-title {
            font-size: 1.05rem;
            font-weight: 400;
            margin-bottom: 0.35rem;
        }
        .citation-text-label {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .citation-text-body {
            white-space: pre-wrap;
            line-height: 1.5;
            word-break: break-word;
        }
        .citation-text-body mark {
            background: #fff3a3;
            padding: 0 0.15rem;
            border-radius: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        example_left, example_right = _citation_examples(df_align)
        lookup = _citation_lookup(df_align)

        input_left, input_right = st.columns(2)
        with input_left:
            citation_number_left = st.text_input(
                "Citation number (left)",
                value=example_left,
                help="Format: <facility_id>_<deficiency_tag>_<cms_region>_<n> (example: 75011_600_1_0)",
                key="drilldown_citation_number_left",
            ).strip()
        with input_right:
            citation_number_right = st.text_input(
                "Citation number (right)",
                value=example_right,
                help="Format: <facility_id>_<deficiency_tag>_<cms_region>_<n> (example: 75011_600_1_0)",
                key="drilldown_citation_number_right",
            ).strip()

        search_text = st.text_input(
            "Search and highlight keywords or phrases",
            value="",
            placeholder="fall, smoking, pressure ulcer",
            help="Highlights matching text in both citation panels. Separate multiple terms with commas.",
            key="drilldown_text_search",
        )
        search_terms = _parse_search_terms(search_text)

        def _citation_payload(citation_number: str) -> tuple[dict[str, object] | None, str | None]:
            if not citation_number:
                return None, "Enter a citation number to populate this comparison panel."

            row = _lookup_citation_row(df_align, citation_number, lookup)
            if row is None:
                return None, f"Citation `{citation_number}` was not found in the current computed alignment dataset."

            tag_num = row.get("deficiency_tag")
            tag_value = pd.to_numeric(pd.Series([tag_num]), errors="coerce").iloc[0]
            tag_int = int(tag_value) if pd.notna(tag_value) else None
            baseline_text = deep.baseline_text_for_tag(baseline_df, tag_int) if tag_int is not None else ""
            sim_raw = row.get(sim_col)
            sim_val = float(sim_raw) if pd.notna(sim_raw) else float("nan")

            return (
                {
                    "citation_id": str(row.get("citation_id", "")),
                    "tag": tag_int,
                    "state": row.get("state"),
                    "region": row.get("cms_region"),
                    "severity": row.get("scope_severity"),
                    "similarity_score": sim_val,
                    "interpretation": similarity_interpretation(sim_val),
                    "inspection_text": str(row.get("inspection_text", "")),
                    "baseline_text": baseline_text,
                },
                None,
            )

        payload_left, error_left = _citation_payload(citation_number_left)
        payload_right, error_right = _citation_payload(citation_number_right)

        def _render_citation_panel(payload: dict[str, object] | None, error_message: str | None) -> None:
            if error_message is not None:
                st.warning(error_message)
                return
            if payload is None:
                st.info("No citation selected.")
                return

            citation_id = str(payload["citation_id"])
            state_label = _format_context_value(payload["state"])
            tag_label = _format_context_value(payload["tag"], prefix="F-0")
            region_label = _format_context_value(payload["region"], prefix="Region ")
            severity_label = _format_context_value(payload["severity"])

            st.markdown(
                f"<div class='citation-panel-title'>{html.escape(state_label if state_label != 'n/a' else 'State unavailable')}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"`{citation_id}`")
            st.caption(f"{tag_label} | {region_label} | Severity {severity_label}")

            metric_left, metric_right = st.columns(2)
            metric_left.metric(
                "Similarity Score",
                f"{float(payload['similarity_score']):.3f}" if pd.notna(payload["similarity_score"]) else "n/a",
            )
            metric_right.metric("Interpretation", str(payload["interpretation"]))

            _render_highlighted_text(
                "Inspection Text",
                str(payload["inspection_text"]),
                search_terms,
                empty_message="Inspection text not available for this citation.",
            )
            _render_highlighted_text(
                "Appendix PP Baseline Text",
                str(payload["baseline_text"]),
                search_terms,
                empty_message="Baseline text not found for this tag.",
            )

        compare_left, compare_right = st.columns(2)
        with compare_left:
            _render_citation_panel(payload_left, error_left)
        with compare_right:
            _render_citation_panel(payload_right, error_right)
