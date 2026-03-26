from __future__ import annotations

import pandas as pd

from .config import TAG_DESCRIPTIONS, TARGET_TAGS


def region_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "cms_region" not in df.columns:
        return pd.Series(dtype="int64")
    return df.groupby("cms_region").size().sort_index()


def severity_by_region(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cms_region" not in df.columns or "scope_severity" not in df.columns:
        return pd.DataFrame()
    out = df.groupby(["cms_region", "scope_severity"]).size().unstack(fill_value=0)
    for col in ["J", "K", "L"]:
        if col not in out.columns:
            out[col] = 0
    return out[["J", "K", "L"]].sort_index()


def severity_pct_by_region(df: pd.DataFrame) -> pd.DataFrame:
    counts = severity_by_region(df)
    if counts.empty:
        return counts
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return pct


def pivot_tag_region(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "deficiency_tag" not in df.columns or "cms_region" not in df.columns:
        return pd.DataFrame()
    pivot = df.groupby(["deficiency_tag", "cms_region"]).size().unstack(fill_value=0)
    pivot.index = [f"F-0{int(t)}" if pd.notna(t) else "Unknown" for t in pivot.index]
    return pivot.sort_index()


def tag_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "tag_label" not in df.columns:
        return pd.Series(dtype="int64")
    return df["tag_label"].value_counts().sort_values(ascending=False)


def tag_severity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "tag_label" not in df.columns or "scope_severity" not in df.columns:
        return pd.DataFrame()
    out = df.groupby(["tag_label", "scope_severity"]).size().unstack(fill_value=0)
    for col in ["J", "K", "L"]:
        if col not in out.columns:
            out[col] = 0
    return out[["J", "K", "L"]]


def avg_word_count_region_severity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "word_count" not in df.columns:
        return pd.DataFrame()
    if "cms_region" not in df.columns or "scope_severity" not in df.columns or "text_available" not in df.columns:
        return pd.DataFrame()
    out = (
        df[df["text_available"]]
        .groupby(["cms_region", "scope_severity"])["word_count"]
        .mean()
        .unstack()
    )
    for col in ["J", "K", "L"]:
        if col not in out.columns:
            out[col] = float("nan")
    return out[["J", "K", "L"]].sort_index()


def monthly_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "inspection_month" not in df.columns:
        return pd.Series(dtype="int64")
    return df.groupby("inspection_month").size().sort_index()


def monthly_by_region(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "inspection_month" not in df.columns or "cms_region" not in df.columns:
        return pd.DataFrame()
    return df.groupby(["inspection_month", "cms_region"]).size().unstack(fill_value=0).sort_index()


def state_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "state" not in df.columns:
        return pd.Series(dtype="int64")
    return df["state"].value_counts().sort_values(ascending=False)


def regional_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for region in sorted([r for r in df.get("cms_region", pd.Series(dtype="int64")).dropna().unique().tolist()]):
        region_data = df[df["cms_region"] == region]
        if "text_available" in region_data.columns:
            text_data = region_data[region_data["text_available"]]
        else:
            text_data = region_data.iloc[0:0]
        rows.append(
            {
                "CMS Region": int(region),
                "Total Citations": int(len(region_data)),
                "J Count": int((region_data["scope_severity"] == "J").sum()) if "scope_severity" in region_data else 0,
                "K Count": int((region_data["scope_severity"] == "K").sum()) if "scope_severity" in region_data else 0,
                "L Count": int((region_data["scope_severity"] == "L").sum()) if "scope_severity" in region_data else 0,
                "Unique Facilities": int(region_data["facility_id"].nunique()) if "facility_id" in region_data else 0,
                "States": ", ".join(sorted([s for s in region_data.get("state", pd.Series(dtype=str)).dropna().unique().tolist()])),
                "Avg Word Count": float(text_data["word_count"].mean()) if "word_count" in text_data and len(text_data) else 0.0,
                "Median Word Count": float(text_data["word_count"].median()) if "word_count" in text_data and len(text_data) else 0.0,
                "Text Available %": float(region_data["text_available"].mean() * 100) if "text_available" in region_data.columns and len(region_data) else 0.0,
            }
        )
    return pd.DataFrame(rows).round(1)


def tag_summary_statistics(df: pd.DataFrame, tags: list[int] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tags = TARGET_TAGS if tags is None else tags

    rows = []
    for tag in tags:
        tag_data = df[df["deficiency_tag"] == tag] if "deficiency_tag" in df.columns else df.iloc[0:0]
        if "text_available" in tag_data.columns:
            text_data = tag_data[tag_data["text_available"]]
        else:
            text_data = tag_data.iloc[0:0]
        description = TAG_DESCRIPTIONS.get(tag, "")
        rows.append(
            {
                "Tag": f"F-0{tag}",
                "Description": description.split(": ", 1)[-1] if description else "",
                "Total Citations": int(len(tag_data)),
                "Regions with Citations": int(tag_data["cms_region"].nunique()) if "cms_region" in tag_data else 0,
                "J Count": int((tag_data["scope_severity"] == "J").sum()) if "scope_severity" in tag_data else 0,
                "K Count": int((tag_data["scope_severity"] == "K").sum()) if "scope_severity" in tag_data else 0,
                "L Count": int((tag_data["scope_severity"] == "L").sum()) if "scope_severity" in tag_data else 0,
                "Avg Word Count": float(text_data["word_count"].mean()) if "word_count" in text_data and len(text_data) else 0.0,
            }
        )
    out = pd.DataFrame(rows).round(1)
    if "Total Citations" in out.columns:
        out = out.sort_values("Total Citations", ascending=False)
    return out
