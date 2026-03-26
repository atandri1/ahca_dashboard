from __future__ import annotations

import pandas as pd

from .config import DEFAULT_MIN_YEAR, TARGET_TAGS


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
        df["inspection_year"] = df["inspection_date"].dt.year
        # Periods get awkward in CSV roundtrips; store YYYY-MM strings.
        df["inspection_month"] = df["inspection_date"].dt.to_period("M").astype(str)

    if "inspection_text" in df.columns:
        text = df["inspection_text"].fillna("").astype(str)
        df["text_length"] = text.str.len()
        df["word_count"] = text.str.split().str.len()
        df["text_available"] = ~text.str.contains("Not Available|not available", case=False, na=True)
    else:
        df["text_length"] = 0
        df["word_count"] = 0
        df["text_available"] = False

    if "deficiency_tag" in df.columns:
        tag_num = pd.to_numeric(df["deficiency_tag"], errors="coerce").astype("Int64")
        df["deficiency_tag"] = tag_num
        df["tag_label"] = tag_num.map(lambda x: f"F-0{int(x)}" if pd.notna(x) else None)

    return df


def filter_to_jkl(df: pd.DataFrame) -> pd.DataFrame:
    if "scope_severity" not in df.columns:
        return df.copy()
    return df[df["scope_severity"].isin(["J", "K", "L"])].copy()


def filter_to_target_tags(df: pd.DataFrame, tags: list[int] | None = None) -> pd.DataFrame:
    tags = TARGET_TAGS if tags is None else tags
    if "deficiency_tag" not in df.columns:
        return df.copy()
    return df[df["deficiency_tag"].isin(tags)].copy()


def filter_to_min_year(df: pd.DataFrame, min_year: int = DEFAULT_MIN_YEAR) -> pd.DataFrame:
    if "inspection_year" not in df.columns:
        return df.copy()
    return df[df["inspection_year"] >= min_year].copy()


def add_citation_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a deterministic `citation_id` column matching the pattern used in embedding metadata:
    `<facility_id>_<deficiency_tag>_<cms_region>_<n>` where `n` is a 0-based counter within each
    (facility_id, deficiency_tag, cms_region) group, preserving the current row order.
    """
    if "citation_id" in df.columns:
        return df
    df = df.copy()

    required = ["facility_id", "deficiency_tag", "cms_region"]
    if any(c not in df.columns for c in required):
        df["citation_id"] = pd.NA
        return df

    facility = df["facility_id"].astype("string")
    tag = pd.to_numeric(df["deficiency_tag"], errors="coerce").astype("Int64")
    region = pd.to_numeric(df["cms_region"], errors="coerce").astype("Int64")

    mask = facility.notna() & tag.notna() & region.notna()
    seq = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    if mask.any():
        seq.loc[mask] = (
            df.loc[mask]
            .groupby([facility.loc[mask], tag.loc[mask], region.loc[mask]], sort=False)
            .cumcount()
            .astype("Int64")
            .to_numpy()
        )

    df["citation_id"] = (
        facility
        + "_"
        + tag.astype("string")
        + "_"
        + region.astype("string")
        + "_"
        + seq.astype("string")
    ).where(mask, pd.NA)
    return df
