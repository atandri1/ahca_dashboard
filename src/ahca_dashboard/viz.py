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


def fig_tag_bifurcation(tag_rank: pd.DataFrame) -> go.Figure:
    if tag_rank.empty:
        return go.Figure()
    df = tag_rank.copy().sort_values("alignment_mean", ascending=True)
    color_map = {
        "Red zone (<40%)": "#b22222",
        "Yellow zone (40-50%)": "#d98e04",
        "Green zone (50%+)": "#2f855a",
    }
    fig = px.bar(
        df,
        x=df["alignment_mean"] * 100,
        y="tag_label",
        orientation="h",
        color="zone",
        color_discrete_map=color_map,
        custom_data=["citation_count"],
        title="Tag Performance Spectrum",
        labels={"x": "Mean Similarity (%)", "tag_label": "Deficiency Tag", "zone": "Zone"},
        text=df["alignment_mean"].map(lambda x: f"{x * 100:.1f}%"),
    )
    fig.update_traces(
        hovertemplate="%{y}<br>Mean similarity: %{x:.1f}%<br>Citations: %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(xaxis_range=[0, max(55, float(df["alignment_mean"].max() * 100) + 5)])
    fig.add_vline(x=40, line_dash="dash", line_color="black", opacity=0.7)
    fig.add_annotation(x=40, y=1.04, yref="paper", text="40% boundary", showarrow=False)
    return fig


def fig_regional_divergence(region_stats: pd.DataFrame) -> go.Figure:
    if region_stats.empty:
        return go.Figure()
    df = region_stats.copy().sort_values("region")
    x = [f"R{int(r)}" for r in df["region"]]
    overall = float(df["region_mean"].mean()) if not df.empty else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["max"] * 100,
            mode="lines",
            line={"color": "rgba(40,167,69,0.25)", "width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["min"] * 100,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(40,167,69,0.2)",
            line={"color": "rgba(40,167,69,0.25)", "width": 0},
            name="Range (worst to best combo)",
            hovertemplate="%{x}<br>Min similarity: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["region_mean"] * 100,
            mode="lines+markers+text",
            text=[f"{v * 100:.1f}%" for v in df["region_mean"]],
            textposition="top center",
            name="Region average",
            line={"color": "#1f4e79", "width": 3},
            marker={"size": 10},
            hovertemplate="%{x}<br>Region average: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["min"] * 100,
            mode="markers",
            marker={"symbol": "triangle-down", "size": 11, "color": "#b22222"},
            name="Worst combo",
            hovertemplate="%{x}<br>Worst combo: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["max"] * 100,
            mode="markers",
            marker={"symbol": "triangle-up", "size": 11, "color": "#2f855a"},
            name="Best combo",
            hovertemplate="%{x}<br>Best combo: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=overall * 100, line_dash="dash", line_color="#b22222", annotation_text="Overall average")
    fig.update_layout(title="Regional Divergence Across Tag x Severity Combinations", xaxis_title="CMS Region", yaxis_title="Mean Similarity (%)")
    return fig


def fig_state_league_table(state_stats: pd.DataFrame) -> go.Figure:
    if state_stats.empty:
        return go.Figure()
    df = state_stats.copy().sort_values("alignment_mean", ascending=True)
    df["state_region"] = df.apply(lambda row: f"{row['state']} (R{int(row['region'])})", axis=1)
    color_map = {
        "Gold (50%+)": "#d4af37",
        "Silver (45-50%)": "#a7a7ad",
        "Bronze (40-45%)": "#b87333",
        "Red (<40%)": "#b22222",
    }
    fig = px.bar(
        df,
        x=df["alignment_mean"] * 100,
        y="state_region",
        orientation="h",
        color="tier",
        color_discrete_map=color_map,
        custom_data=["citation_count"],
        title="State League Table",
        labels={"x": "Mean Similarity (%)", "state_region": "State (Region)", "tier": "Tier"},
        text=df["alignment_mean"].map(lambda x: f"{x * 100:.1f}%"),
    )
    fig.update_traces(
        hovertemplate="%{y}<br>Mean similarity: %{x:.1f}%<br>Citations: %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(height=max(450, min(2200, 24 * len(df) + 120)))
    return fig


def fig_region_tag_severity_heatmap(pivot: pd.DataFrame, region: int) -> go.Figure:
    if pivot.empty:
        return go.Figure()
    x_vals = [str(c) for c in pivot.columns.tolist()]
    y_vals = [str(i) for i in pivot.index.tolist()]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_vals,
            y=y_vals,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar={"title": "Mean sim"},
            hovertemplate="Tag %{y}<br>Severity %{x}<br>Mean similarity %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Region {int(region)}: Mean Similarity by Tag and Severity",
        xaxis_title="Severity",
        yaxis_title="Deficiency Tag",
    )
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=y_vals)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=x_vals)
    return fig


def fig_keyword_profile(words: pd.DataFrame, *, title: str, color: str) -> go.Figure:
    if words.empty:
        return go.Figure()
    df = words.head(20).iloc[::-1].copy()
    fig = px.bar(
        df,
        x="tfidf_score",
        y="word",
        orientation="h",
        title=title,
        labels={"tfidf_score": "Mean TF-IDF", "word": "Word / phrase"},
    )
    fig.update_traces(marker_color=color, hovertemplate="%{y}<br>Mean TF-IDF: %{x:.4f}<extra></extra>")
    return fig


def fig_keyword_difference(words: pd.DataFrame, *, title: str) -> go.Figure:
    if words.empty:
        return go.Figure()
    df = words.head(20).iloc[::-1].copy()
    fig = px.bar(
        df,
        x="difference",
        y="word",
        orientation="h",
        title=title,
        labels={"difference": "TF-IDF advantage", "word": "Shared word / phrase"},
    )
    fig.update_traces(marker_color="#8b4513", hovertemplate="%{y}<br>TF-IDF advantage: %{x:.4f}<extra></extra>")
    return fig


def fig_domain_indicator_profile(domain_df: pd.DataFrame) -> go.Figure:
    if domain_df.empty:
        return go.Figure()
    df = domain_df.copy()
    df["tag_label"] = df["tag"].apply(lambda x: f"F-0{int(x)}")
    return px.bar(
        df,
        x="tag_label",
        y="count",
        color="domain",
        barmode="group",
        title="Domain Indicator Counts in Top 30 Terms",
        labels={"tag_label": "Deficiency Tag", "count": "Matched top terms", "domain": "Indicator family"},
    )
