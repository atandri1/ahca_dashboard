from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SEVERITY_TICKVALS = list(range(1, 13))
SEVERITY_TICKTEXT = list("ABCDEFGHIJKL")


def fig_region_counts(region_counts: pd.Series) -> go.Figure:
    df = region_counts.reset_index()
    df.columns = ["cms_region", "citations"]
    return px.bar(
        df,
        x="cms_region",
        y="citations",
        title="J/K/L Citations by CMS Region",
        labels={"cms_region": "CMS Region", "citations": "Citations"},
    )


def fig_severity_by_region(sev: pd.DataFrame) -> go.Figure:
    df = sev.reset_index().melt(id_vars=["cms_region"], var_name="severity", value_name="citations")
    return px.bar(
        df,
        x="cms_region",
        y="citations",
        color="severity",
        barmode="stack",
        title="Citations by Severity and Region",
        labels={"cms_region": "CMS Region", "citations": "Citations"},
        category_orders={"severity": ["J", "K", "L"]},
        color_discrete_map={"J": "#2ecc71", "K": "#f39c12", "L": "#e74c3c"},
    )


def fig_severity_pct_by_region(sev_pct: pd.DataFrame) -> go.Figure:
    df = sev_pct.reset_index().melt(id_vars=["cms_region"], var_name="severity", value_name="pct")
    return px.bar(
        df,
        x="cms_region",
        y="pct",
        color="severity",
        barmode="stack",
        title="Severity Distribution by Region (Normalized)",
        labels={"cms_region": "CMS Region", "pct": "Percent"},
        category_orders={"severity": ["J", "K", "L"]},
        color_discrete_map={"J": "#2ecc71", "K": "#f39c12", "L": "#e74c3c"},
    )


def fig_tag_region_heatmap(pivot: pd.DataFrame) -> go.Figure:
    if pivot.empty:
        return go.Figure()
    return go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns.tolist()],
            y=pivot.index.tolist(),
            colorscale="YlOrRd",
            colorbar={"title": "Count"},
        )
    ).update_layout(title="Citation Frequency: Tag x Region")


def fig_tag_counts(tag_counts: pd.Series) -> go.Figure:
    df = tag_counts.reset_index()
    df.columns = ["tag_label", "citations"]
    df = df.sort_values("citations", ascending=True)
    return px.bar(
        df,
        x="citations",
        y="tag_label",
        orientation="h",
        title="Citation Frequency by Tag",
        labels={"tag_label": "Deficiency Tag", "citations": "Citations"},
    )


def fig_tag_severity(tag_severity: pd.DataFrame) -> go.Figure:
    if tag_severity.empty:
        return go.Figure()
    df = tag_severity.reset_index().melt(id_vars=["tag_label"], var_name="severity", value_name="citations")
    return px.bar(
        df,
        x="citations",
        y="tag_label",
        color="severity",
        orientation="h",
        barmode="stack",
        title="Tag Distribution by Severity",
        labels={"tag_label": "Deficiency Tag", "citations": "Citations"},
        category_orders={"severity": ["J", "K", "L"]},
        color_discrete_map={"J": "#2ecc71", "K": "#f39c12", "L": "#e74c3c"},
    )


def fig_word_count_box_by_region(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    return px.box(
        df,
        x="cms_region",
        y="word_count",
        title="Citation Word Count by Region (Text Available Only)",
        labels={"cms_region": "CMS Region", "word_count": "Word Count"},
        points="outliers",
    )


def fig_word_count_box_by_severity(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    return px.box(
        df,
        x="scope_severity",
        y="word_count",
        title="Citation Word Count by Severity (Text Available Only)",
        labels={"scope_severity": "Severity", "word_count": "Word Count"},
        points="outliers",
        category_orders={"scope_severity": ["J", "K", "L"]},
    )


def fig_avg_word_count_region_severity(avg_wc: pd.DataFrame) -> go.Figure:
    if avg_wc.empty:
        return go.Figure()
    df = avg_wc.reset_index().melt(id_vars=["cms_region"], var_name="severity", value_name="avg_word_count")
    return px.bar(
        df,
        x="cms_region",
        y="avg_word_count",
        color="severity",
        barmode="group",
        title="Average Word Count by Region and Severity (Text Available Only)",
        labels={"cms_region": "CMS Region", "avg_word_count": "Average Word Count"},
        category_orders={"severity": ["J", "K", "L"]},
        color_discrete_map={"J": "#2ecc71", "K": "#f39c12", "L": "#e74c3c"},
    )


def fig_monthly_trend(monthly: pd.Series) -> go.Figure:
    df = monthly.reset_index()
    df.columns = ["inspection_month", "citations"]
    return px.line(
        df,
        x="inspection_month",
        y="citations",
        markers=True,
        title="Citations Over Time (Monthly)",
        labels={"inspection_month": "Month", "citations": "Citations"},
    )


def fig_monthly_by_region(monthly_by_region: pd.DataFrame, top_n: int = 4) -> go.Figure:
    if monthly_by_region.empty:
        return go.Figure()
    top_regions = monthly_by_region.sum(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    df = monthly_by_region[top_regions].reset_index().melt(id_vars=["inspection_month"], var_name="cms_region", value_name="citations")
    return px.line(
        df,
        x="inspection_month",
        y="citations",
        color="cms_region",
        markers=True,
        title=f"Citations Over Time (Top {top_n} Regions)",
        labels={"inspection_month": "Month", "citations": "Citations", "cms_region": "CMS Region"},
    )


def fig_state_counts(state_counts: pd.Series, top_n: int = 30) -> go.Figure:
    df = state_counts.head(top_n).reset_index()
    df.columns = ["state", "citations"]
    return px.bar(
        df,
        x="state",
        y="citations",
        title=f"Citations by State (Top {top_n})",
        labels={"state": "State", "citations": "Citations"},
    )


def fig_severity_violin(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = px.violin(
        df,
        x="cms_region",
        y="scope_map",
        box=True,
        points=False,
        title="Severity Distribution by Region (All Tags)",
        labels={"cms_region": "CMS Region", "scope_map": "Severity"},
    )
    fig.update_yaxes(tickmode="array", tickvals=SEVERITY_TICKVALS, ticktext=SEVERITY_TICKTEXT)
    return fig


def fig_severity_heatmap(table: pd.DataFrame, title: str) -> go.Figure:
    if table.empty:
        return go.Figure()
    return go.Figure(
        data=go.Heatmap(
            z=table.values,
            x=[str(c) for c in table.columns.tolist()],
            y=[str(i) for i in table.index.tolist()],
            colorscale="Viridis",
            colorbar={"title": "Proportion" if table.max().max() <= 1.0 else "Count"},
        )
    ).update_layout(title=title, xaxis_title="CMS Region", yaxis_title="Severity")


def fig_pval_heatmap(pvals: pd.DataFrame) -> go.Figure:
    if pvals.empty:
        return go.Figure()
    return go.Figure(
        data=go.Heatmap(
            z=pvals.values,
            x=[str(c) for c in pvals.columns.tolist()],
            y=[str(i) for i in pvals.index.tolist()],
            colorscale="Viridis_r",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "Adj p-value"},
        )
    ).update_layout(title="Pairwise Mann-Whitney Tests (BH-adjusted)", xaxis_title="CMS Region", yaxis_title="CMS Region")
