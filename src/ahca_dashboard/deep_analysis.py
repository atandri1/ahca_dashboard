from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .alignment import clean_text_basic

EXPANSION_INDICATORS = [
    "additional",
    "comprehensive",
    "enhanced",
    "extended",
    "expanded",
    "full",
    "complete",
    "thorough",
    "require",
    "must",
    "shall",
    "ensure",
    "maintain",
    "provide",
    "document",
    "implement",
]

REDUCTION_INDICATORS = [
    "limited",
    "minimal",
    "only",
    "restrict",
    "limit",
    "exclude",
    "exception",
    "exempt",
    "not",
    "no",
    "none",
    "avoid",
]

WOUND_CARE_INDICATORS = {
    "wound": "wound characteristic",
    "wounds": "wound characteristic",
    "ulcer": "pressure ulcer",
    "ulcers": "pressure ulcer",
    "pressure": "pressure ulcer",
    "pressure ulcer": "pressure ulcer",
    "pressure injury": "pressure injury",
    "skin": "skin condition",
    "tissue": "tissue condition",
    "dressing": "wound treatment",
    "treatment": "medical treatment",
    "treatments": "medical treatment",
    "wound care": "wound management",
    "heel": "anatomy - common ulcer site",
    "sacrum": "anatomy - common ulcer site",
    "foot": "anatomy - common ulcer site",
    "cm": "wound measurement",
    "stage": "ulcer classification",
    "injury": "tissue injury",
}

BEHAVIORAL_SAFETY_INDICATORS = {
    "elopement": "behavioral - elopement risk",
    "fall": "safety - fall risk",
    "falls": "safety - fall risk",
    "abuse": "safety - abuse/neglect",
    "smoking": "behavioral - smoking risk",
    "door": "environmental - security",
    "bed": "environmental - bed safety",
    "alarm": "environmental - alarm systems",
    "incident": "behavioral - incident documentation",
    "hospital": "outcome - hospital transfer",
    "restraint": "behavioral - restraint use",
    "behavior": "behavioral - behavior management",
    "behavioral": "behavioral - behavior management",
}

ALL_DOMAIN_INDICATORS = {**WOUND_CARE_INDICATORS, **BEHAVIORAL_SAFETY_INDICATORS}


def alignment_working_frame(df: pd.DataFrame, sim_col: str) -> pd.DataFrame:
    if df.empty or sim_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["alignment_score"] = pd.to_numeric(work[sim_col], errors="coerce")
    work["tag"] = pd.to_numeric(work.get("deficiency_tag"), errors="coerce").astype("Int64")
    work["region"] = pd.to_numeric(work.get("cms_region"), errors="coerce").astype("Int64")
    work["severity"] = work.get("scope_severity", pd.Series(index=work.index, dtype="object")).fillna("").astype(str)
    work["state"] = work.get("state", pd.Series(index=work.index, dtype="object")).fillna("").astype(str)
    work["inspection_text"] = work.get("inspection_text", pd.Series(index=work.index, dtype="object")).fillna("").astype(str)
    work["citation_id"] = work.get("citation_id", pd.Series(index=work.index, dtype="object")).fillna("").astype(str)
    work = work.dropna(subset=["alignment_score", "tag"])
    work["tag"] = work["tag"].astype(int)
    work["tag_label"] = work["tag"].apply(lambda x: f"F-0{x}")
    return work


def tag_severity_alignment_by_region(df: pd.DataFrame, sim_col: str) -> pd.DataFrame:
    work = alignment_working_frame(df, sim_col)
    if work.empty or "region" not in work.columns or "severity" not in work.columns:
        return pd.DataFrame()

    out = (
        work.dropna(subset=["region", "severity"])
        .groupby(["region", "tag", "tag_label", "severity"])["alignment_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "alignment_mean", "std": "alignment_std", "count": "citation_count"})
    )
    out["citation_count"] = out["citation_count"].astype(int)
    return out.sort_values(["region", "alignment_mean", "tag", "severity"])


def region_tag_severity_pivot(combo_stats: pd.DataFrame, region: int) -> pd.DataFrame:
    if combo_stats.empty:
        return pd.DataFrame()
    subset = combo_stats[combo_stats["region"] == region]
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.pivot_table(index="tag_label", columns="severity", values="alignment_mean")
    ordered_cols = [c for c in ["J", "K", "L"] if c in pivot.columns]
    if ordered_cols:
        pivot = pivot[ordered_cols]
    return pivot.sort_index()


def tag_ranking(df: pd.DataFrame, sim_col: str) -> pd.DataFrame:
    work = alignment_working_frame(df, sim_col)
    if work.empty:
        return pd.DataFrame()
    out = (
        work.groupby(["tag", "tag_label"])["alignment_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "alignment_mean", "std": "alignment_std", "count": "citation_count"})
        .sort_values("alignment_mean")
    )
    out["citation_count"] = out["citation_count"].astype(int)
    out["zone"] = out["alignment_mean"].apply(zone_label)
    return out


def severity_ranking(df: pd.DataFrame, sim_col: str) -> pd.DataFrame:
    work = alignment_working_frame(df, sim_col)
    if work.empty:
        return pd.DataFrame()
    out = (
        work.groupby("severity")["alignment_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "alignment_mean", "std": "alignment_std", "count": "citation_count"})
        .sort_values("alignment_mean")
    )
    out["citation_count"] = out["citation_count"].astype(int)
    out["severity_name"] = out["severity"].map(
        {"J": "Immediate Jeopardy", "K": "Pattern / Widespread Potential", "L": "Isolated Potential"}
    ).fillna(out["severity"])
    return out


def critical_combinations(combo_stats: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    if combo_stats.empty:
        return pd.DataFrame()
    out = combo_stats[combo_stats["alignment_mean"] < threshold].copy()
    return out.sort_values(["alignment_mean", "region", "tag", "severity"])


def regional_problem_tags(combo_stats: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if combo_stats.empty:
        return pd.DataFrame()
    rows = []
    for region in sorted(combo_stats["region"].dropna().unique().tolist()):
        subset = combo_stats[combo_stats["region"] == region].sort_values("alignment_mean").head(top_n)
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            rows.append(
                {
                    "region": int(region),
                    "rank": rank,
                    "tag": int(row["tag"]),
                    "tag_label": row["tag_label"],
                    "severity": row["severity"],
                    "alignment_mean": row["alignment_mean"],
                    "citation_count": int(row["citation_count"]),
                }
            )
    return pd.DataFrame(rows)


def state_alignment_summary(df: pd.DataFrame, sim_col: str) -> pd.DataFrame:
    work = alignment_working_frame(df, sim_col)
    if work.empty:
        return pd.DataFrame()
    out = (
        work[work["state"].str.len() > 0]
        .groupby(["state", "region"])["alignment_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "alignment_mean", "std": "alignment_std", "count": "citation_count"})
        .sort_values("alignment_mean")
    )
    out["citation_count"] = out["citation_count"].astype(int)
    out["tier"] = out["alignment_mean"].apply(tier_label)
    return out


def regional_divergence(combo_stats: pd.DataFrame) -> pd.DataFrame:
    if combo_stats.empty:
        return pd.DataFrame()
    out = (
        combo_stats.groupby("region")["alignment_mean"]
        .agg(["min", "max", "mean"])
        .reset_index()
        .rename(columns={"mean": "region_mean"})
    )
    out["spread"] = out["max"] - out["min"]
    return out.sort_values("region")


def notebook_default_tags(df: pd.DataFrame, sim_col: str) -> tuple[int | None, int | None]:
    ranked = tag_ranking(df, sim_col)
    if ranked.empty:
        return None, None
    worst_tag = int(ranked.iloc[0]["tag"])
    best_tag = int(ranked.iloc[-1]["tag"])
    return worst_tag, best_tag


def tag_examples(df: pd.DataFrame, sim_col: str, tag: int) -> tuple[pd.Series | None, pd.Series | None]:
    work = alignment_working_frame(df, sim_col)
    subset = work[(work["tag"] == int(tag)) & work["inspection_text"].str.strip().ne("")]
    if subset.empty:
        return None, None
    worst = subset.nsmallest(1, "alignment_score").iloc[0]
    best = subset.nlargest(1, "alignment_score").iloc[0]
    return worst, best


def keyword_comparison(df: pd.DataFrame, sim_col: str, tag_a: int, tag_b: int) -> dict[str, pd.DataFrame | dict | None]:
    work = alignment_working_frame(df, sim_col)
    tag_a_df = work[(work["tag"] == int(tag_a)) & work["inspection_text"].str.strip().ne("")]
    tag_b_df = work[(work["tag"] == int(tag_b)) & work["inspection_text"].str.strip().ne("")]

    profile_a = _tfidf_profile(tag_a_df["inspection_text"])
    profile_b = _tfidf_profile(tag_b_df["inspection_text"])

    unique_a = pd.DataFrame(columns=["word", "tfidf_score"])
    unique_b = pd.DataFrame(columns=["word", "tfidf_score"])
    shared_diff = pd.DataFrame(columns=["word", "tfidf_score_a", "tfidf_score_b", "difference"])

    if not profile_a.empty and not profile_b.empty:
        unique_a = profile_a[~profile_a["word"].isin(profile_b["word"])].head(20).reset_index(drop=True)
        unique_b = profile_b[~profile_b["word"].isin(profile_a["word"])].head(20).reset_index(drop=True)

        common = set(profile_a["word"]).intersection(profile_b["word"])
        if common:
            shared_diff = (
                profile_a[profile_a["word"].isin(common)][["word", "tfidf_score"]]
                .merge(
                    profile_b[profile_b["word"].isin(common)][["word", "tfidf_score"]],
                    on="word",
                    suffixes=("_a", "_b"),
                )
                .assign(difference=lambda x: x["tfidf_score_a"] - x["tfidf_score_b"])
                .sort_values("difference", ascending=False)
                .reset_index(drop=True)
            )

    stats = {
        "tag_a_mean": float(tag_a_df["alignment_score"].mean()) if not tag_a_df.empty else np.nan,
        "tag_b_mean": float(tag_b_df["alignment_score"].mean()) if not tag_b_df.empty else np.nan,
        "tag_a_count": int(len(tag_a_df)),
        "tag_b_count": int(len(tag_b_df)),
        "pattern_counts": pd.DataFrame(
            [
                {
                    "tag": int(tag_a),
                    "expansion_terms_in_top100": _pattern_count(profile_a, EXPANSION_INDICATORS),
                    "reduction_terms_in_top100": _pattern_count(profile_a, REDUCTION_INDICATORS),
                },
                {
                    "tag": int(tag_b),
                    "expansion_terms_in_top100": _pattern_count(profile_b, EXPANSION_INDICATORS),
                    "reduction_terms_in_top100": _pattern_count(profile_b, REDUCTION_INDICATORS),
                },
            ]
        ),
    }

    return {
        "profile_a": profile_a,
        "profile_b": profile_b,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "shared_diff": shared_diff,
        "stats": stats,
    }


def domain_indicator_summary(profile: pd.DataFrame, tag: int, top_n: int = 30) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(columns=["tag", "domain", "count"])

    top_terms = profile.head(top_n).copy()
    if top_terms.empty:
        return pd.DataFrame(columns=["tag", "domain", "count"])

    top_terms["domain"] = top_terms["word"].map(ALL_DOMAIN_INDICATORS).fillna("notebook uncategorized")
    summary = (
        top_terms.groupby("domain")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "domain"], ascending=[False, True])
    )
    summary["tag"] = int(tag)
    return summary[["tag", "domain", "count"]]


def infer_domain_family(profile: pd.DataFrame, top_n: int = 30) -> str:
    if profile.empty:
        return "generic"
    top_words = set(profile.head(top_n)["word"].tolist())
    wound_count = sum(1 for word in top_words if word in WOUND_CARE_INDICATORS)
    behavioral_count = sum(1 for word in top_words if word in BEHAVIORAL_SAFETY_INDICATORS)
    if wound_count == 0 and behavioral_count == 0:
        return "generic"
    return "wound" if wound_count >= behavioral_count else "behavioral"


def domain_unique_terms(profile_primary: pd.DataFrame, profile_other: pd.DataFrame, *, domain: str | None = None) -> pd.DataFrame:
    if profile_primary.empty:
        return pd.DataFrame(columns=["word", "tfidf_score", "domain_note"])
    other_words = set(profile_other["word"]) if not profile_other.empty else set()
    unique_terms = profile_primary[~profile_primary["word"].isin(other_words)].copy().head(20)
    domain = infer_domain_family(profile_primary) if domain is None else domain
    if domain == "wound":
        unique_terms["domain_note"] = unique_terms["word"].map(WOUND_CARE_INDICATORS).fillna("medical / clinical")
    elif domain == "behavioral":
        unique_terms["domain_note"] = unique_terms["word"].map(BEHAVIORAL_SAFETY_INDICATORS).fillna("behavioral / safety")
    else:
        unique_terms["domain_note"] = unique_terms["word"].map(ALL_DOMAIN_INDICATORS).fillna("notebook uncategorized")
    return unique_terms.reset_index(drop=True)


def dominant_domain_note(unique_terms: pd.DataFrame) -> dict[str, object]:
    if unique_terms.empty or "domain_note" not in unique_terms.columns:
        return {"label": "No domain note", "count": 0, "total_terms": 0}

    summary = (
        unique_terms.dropna(subset=["domain_note"])
        .groupby("domain_note")
        .agg(term_count=("domain_note", "size"), total_tfidf=("tfidf_score", "sum"))
        .reset_index()
        .sort_values(["term_count", "total_tfidf", "domain_note"], ascending=[False, False, True])
    )
    if summary.empty:
        return {"label": "No domain note", "count": 0, "total_terms": 0}

    top_row = summary.iloc[0]
    return {
        "label": str(top_row["domain_note"]),
        "count": int(top_row["term_count"]),
        "total_terms": int(summary["term_count"].sum()),
    }


def baseline_text_for_tag(baseline_df: pd.DataFrame | None, tag: int) -> str:
    if baseline_df is None or baseline_df.empty or "deficiency_tag" not in baseline_df.columns:
        return ""
    tags = pd.to_numeric(baseline_df["deficiency_tag"], errors="coerce")
    match = baseline_df.loc[tags == int(tag)]
    if match.empty or "text" not in match.columns:
        return ""
    return str(match.iloc[0]["text"] or "")


def citation_vs_baseline_overlap(citation_text: str, baseline_text: str, top_n: int = 12) -> pd.DataFrame:
    citation_terms = _top_terms_from_text(citation_text, max_features=top_n)
    baseline_terms = _top_terms_from_text(baseline_text, max_features=top_n)
    citation_set = set(citation_terms["word"]) if not citation_terms.empty else set()
    baseline_set = set(baseline_terms["word"]) if not baseline_terms.empty else set()

    rows = []
    for word in sorted(citation_set.union(baseline_set)):
        rows.append(
            {
                "word": word,
                "in_citation": word in citation_set,
                "in_baseline": word in baseline_set,
                "overlap_type": overlap_label(word in citation_set, word in baseline_set),
            }
        )
    return pd.DataFrame(rows)


def overlap_label(in_citation: bool, in_baseline: bool) -> str:
    if in_citation and in_baseline:
        return "Shared"
    if in_citation:
        return "Citation only"
    return "Baseline only"


def zone_label(value: float) -> str:
    if value < 0.40:
        return "Red zone (<40%)"
    if value < 0.50:
        return "Yellow zone (40-50%)"
    return "Green zone (50%+)"


def tier_label(value: float) -> str:
    if value >= 0.50:
        return "Gold (50%+)"
    if value >= 0.45:
        return "Silver (45-50%)"
    if value >= 0.40:
        return "Bronze (40-45%)"
    return "Red (<40%)"


def _tfidf_profile(
    texts: pd.Series,
    *,
    max_features: int = 500,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.8,
) -> pd.DataFrame:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception as e:
        raise RuntimeError("scikit-learn is required for keyword analysis. Install `scikit-learn`.") from e

    clean = texts.fillna("").astype(str).map(clean_text_basic)
    clean = clean[clean.str.len() > 0]
    if clean.empty:
        return pd.DataFrame(columns=["word", "tfidf_score"])

    effective_min_df = 2 if len(clean) >= 2 else 1
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df if len(clean) >= min_df else effective_min_df,
            max_df=max_df if len(clean) > 1 else 1.0,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(clean.tolist())
    except ValueError:
        return pd.DataFrame(columns=["word", "tfidf_score"])

    scores = np.asarray(matrix.mean(axis=0)).ravel()
    out = pd.DataFrame({"word": vectorizer.get_feature_names_out(), "tfidf_score": scores})
    return out.sort_values("tfidf_score", ascending=False).reset_index(drop=True)


def _top_terms_from_text(text: str, max_features: int = 12) -> pd.DataFrame:
    clean = clean_text_basic(text)
    if not clean:
        return pd.DataFrame(columns=["word", "tfidf_score"])
    tokens = [t for t in re.findall(r"[a-z][a-z0-9]+", clean) if len(t) > 2]
    if not tokens:
        return pd.DataFrame(columns=["word", "tfidf_score"])
    freq = pd.Series(tokens).value_counts().head(max_features)
    return pd.DataFrame({"word": freq.index.tolist(), "tfidf_score": freq.values.astype(float)})


def _pattern_count(profile: pd.DataFrame, indicators: list[str]) -> int:
    if profile.empty:
        return 0
    words = set(profile.head(100)["word"].tolist())
    return sum(1 for word in words if any(ind in word for ind in indicators))
