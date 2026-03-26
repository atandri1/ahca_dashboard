from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


SEVERITY_MAP = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
}


def add_scope_numeric(df: pd.DataFrame, col_in: str = "scope_severity", col_out: str = "scope_map") -> pd.DataFrame:
    df = df.copy()
    if col_in not in df.columns:
        df[col_out] = pd.Series([pd.NA] * len(df), dtype="Int64")
        return df
    mapped = df[col_in].astype(str).str.strip().str.upper().map(SEVERITY_MAP)
    df[col_out] = pd.to_numeric(mapped, errors="coerce").astype("Int64")
    return df


def severity_region_table(
    df: pd.DataFrame,
    *,
    severity_col: str = "scope_severity",
    region_col: str = "cms_region",
    severity_order: Iterable[str] = tuple("ABCDEFGHIJKL"),
    normalize_by_region: bool = True,
) -> pd.DataFrame:
    if df.empty or severity_col not in df.columns or region_col not in df.columns:
        return pd.DataFrame()
    table = df.groupby([severity_col, region_col]).size().unstack(fill_value=0)
    # Force consistent ordering for display.
    table = table.reindex(list(severity_order))
    if normalize_by_region:
        table = table.div(table.sum(axis=0), axis=1)
    return table


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (BH)."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity.
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def kruskal_wallis_by_region(
    df: pd.DataFrame,
    *,
    scope_num_col: str = "scope_map",
    region_col: str = "cms_region",
) -> tuple[float, float] | None:
    """Returns (H-stat, p-value) or None if SciPy isn't available / insufficient data."""
    try:
        from scipy.stats import kruskal  # type: ignore
    except Exception:
        return None

    if df.empty or scope_num_col not in df.columns or region_col not in df.columns:
        return None

    groups = []
    for _, g in df.groupby(region_col):
        vals = pd.to_numeric(g[scope_num_col], errors="coerce").dropna().values
        if vals.size:
            groups.append(vals)
    if len(groups) < 2:
        return None
    stat, p = kruskal(*groups)
    return float(stat), float(p)


def pairwise_mannwhitney_fdr(
    df: pd.DataFrame,
    *,
    scope_num_col: str = "scope_map",
    region_col: str = "cms_region",
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests across regions with BH-adjusted p-values.

    Returns a symmetric matrix (regions x regions) with NaN on the diagonal.
    """
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
    except Exception:
        return pd.DataFrame()

    if df.empty or scope_num_col not in df.columns or region_col not in df.columns:
        return pd.DataFrame()

    regions = sorted([int(r) for r in pd.to_numeric(df[region_col], errors="coerce").dropna().unique().tolist()])
    if len(regions) < 2:
        return pd.DataFrame()

    pairs: list[tuple[int, int]] = []
    pvals: list[float] = []

    for r1, r2 in combinations(regions, 2):
        x = pd.to_numeric(df.loc[df[region_col] == r1, scope_num_col], errors="coerce").dropna().values
        y = pd.to_numeric(df.loc[df[region_col] == r2, scope_num_col], errors="coerce").dropna().values
        if x.size == 0 or y.size == 0:
            continue
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        pairs.append((r1, r2))
        pvals.append(float(p))

    if not pvals:
        return pd.DataFrame()

    p_adj = _benjamini_hochberg(np.array(pvals))

    mat = pd.DataFrame(np.nan, index=regions, columns=regions, dtype=float)
    for (r1, r2), p in zip(pairs, p_adj):
        mat.loc[r1, r2] = p
        mat.loc[r2, r1] = p
    return mat

