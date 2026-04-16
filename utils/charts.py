"""
Chart utilities for the TeraWave Capital Valuation dashboard.
All charts use Plotly for interactivity.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


COLORS = {
    "primary": "#4A9EFF",
    "secondary": "#7AB8FF",
    "accent": "#D4A574",
    "positive": "#52C7A0",
    "negative": "#E07B7B",
    "neutral": "#6C7686",
    "muted": "#8B95A7",
    "bg": "#0E1117",
    "text": "#D1D5DB",
    "title": "#E5E7EB",
    "grid": "#1F2937",
}

CATEGORICAL = [
    "#4A9EFF", "#7AB8FF", "#52C7A0", "#D4A574",
    "#B794F4", "#8B95A7", "#F6AD55", "#9FB1C7",
]


def _base_layout(title: str, height: int = 420) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=COLORS["title"], family="Inter, system-ui, sans-serif"),
            x=0.0, xanchor="left", pad=dict(l=8, t=4),
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12, family="Inter, system-ui, sans-serif"),
        xaxis=dict(
            gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
            linecolor=COLORS["grid"], tickfont=dict(color=COLORS["muted"], size=11),
            title=dict(font=dict(color=COLORS["muted"], size=11)),
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
            linecolor=COLORS["grid"], tickfont=dict(color=COLORS["muted"], size=11),
            title=dict(font=dict(color=COLORS["muted"], size=11)),
        ),
        margin=dict(l=60, r=30, t=44, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"], size=11),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
        ),
    )


def cash_flow_waterfall(projection: pd.DataFrame) -> go.Figure:
    """Stacked bar chart showing CapEx, OpEx, Revenue, and FCF by year."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=projection["calendar_year"],
        y=-projection["capex_m"],
        name="CapEx",
        marker_color=COLORS["negative"],
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=projection["calendar_year"],
        y=-projection["opex_m"],
        name="OpEx",
        marker_color=COLORS["accent"],
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=projection["calendar_year"],
        y=projection["revenue_m"],
        name="Revenue",
        marker_color=COLORS["positive"],
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=projection["calendar_year"],
        y=projection["free_cash_flow_m"],
        name="Free Cash Flow",
        line=dict(color=COLORS["secondary"], width=3),
        mode="lines+markers",
    ))

    layout = _base_layout("Annual Cash Flow Projection ($M)")
    layout["barmode"] = "relative"
    layout["yaxis"]["title"] = "$M"
    fig.update_layout(**layout)
    return fig


def cumulative_investment_chart(projection: pd.DataFrame) -> go.Figure:
    """Cumulative investment vs cumulative FCF — shows payback visually."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=projection["calendar_year"],
        y=projection["cumulative_investment_m"],
        name="Cumulative Investment",
        fill="tozeroy",
        fillcolor="rgba(224, 123, 123, 0.12)",
        line=dict(color=COLORS["negative"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=projection["calendar_year"],
        y=projection["cumulative_fcf_m"],
        name="Cumulative FCF",
        fill="tozeroy",
        fillcolor="rgba(82, 199, 160, 0.12)",
        line=dict(color=COLORS["positive"], width=2),
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)

    layout = _base_layout("Cumulative Investment vs. Free Cash Flow ($M)")
    layout["yaxis"]["title"] = "$M"
    fig.update_layout(**layout)
    return fig


def npv_distribution_chart(npv_dist: np.ndarray, wacc: float) -> go.Figure:
    """Histogram of NPV outcomes from Monte Carlo simulation."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=npv_dist,
        nbinsx=60,
        marker_color=COLORS["secondary"],
        opacity=0.75,
        name="NPV Distribution",
    ))

    # Add percentile lines
    for pct, style in [(10, "dash"), (50, "solid"), (90, "dash")]:
        val = np.percentile(npv_dist, pct)
        fig.add_vline(
            x=val, line_dash=style,
            line_color=COLORS["accent"] if pct == 50 else COLORS["neutral"],
            annotation_text=f"P{pct}: ${val:,.0f}M",
            annotation_position="top",
        )

    # Zero line
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["negative"],
                  annotation_text="Break-even", annotation_position="bottom")

    layout = _base_layout(f"NPV Distribution — {len(npv_dist):,} Simulations (WACC={wacc:.0%})")
    layout["xaxis"]["title"] = "NPV ($M)"
    layout["yaxis"]["title"] = "Frequency"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def tornado_chart(tornado_df: pd.DataFrame) -> go.Figure:
    """Tornado chart showing sensitivity of NPV to each variable."""
    fig = go.Figure()

    base_npv = tornado_df.iloc[0]["base_npv_m"]

    fig.add_trace(go.Bar(
        y=tornado_df["variable"],
        x=tornado_df["low_npv_m"] - base_npv,
        orientation="h",
        name="Downside (-25%)",
        marker_color=COLORS["negative"],
        opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        y=tornado_df["variable"],
        x=tornado_df["high_npv_m"] - base_npv,
        orientation="h",
        name="Upside (+25%)",
        marker_color=COLORS["positive"],
        opacity=0.8,
    ))

    layout = _base_layout("Sensitivity Analysis — NPV Impact ($M)")
    layout["xaxis"]["title"] = "Change in NPV ($M)"
    layout["barmode"] = "relative"
    fig.update_layout(**layout)
    return fig


def progress_per_dollar_chart(progress_df: pd.DataFrame) -> go.Figure:
    """Bubble chart: x=progress per $B, y=risk retired per $B, size=total capex."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=progress_df["progress_per_bn"],
        y=progress_df["risk_retired_per_bn"],
        mode="markers+text",
        text=progress_df["workstream"].str.replace("Satellite Manufacturing ", "Sat Mfg "),
        textposition="top center",
        textfont=dict(size=10, color=COLORS["text"]),
        marker=dict(
            size=progress_df["total_capex_m"] / progress_df["total_capex_m"].max() * 60 + 15,
            color=progress_df["composite_score"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Composite<br>Score"),
            line=dict(width=1, color=COLORS["text"]),
        ),
    ))

    layout = _base_layout("Capital Efficiency: Progress vs. Risk Retirement per $B Deployed", height=500)
    layout["xaxis"]["title"] = "Progress per $B Deployed"
    layout["yaxis"]["title"] = "Risk Retired per $B Deployed"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def capex_by_workstream_chart(capex_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart showing CapEx deployment by workstream over time."""
    pivot = capex_df.groupby(["year", "workstream"])["capex_m"].sum().unstack(fill_value=0)
    pivot.index = [2025 + yr for yr in pivot.index]

    fig = go.Figure()

    for i, col in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[col],
            name=col.replace("Satellite Manufacturing ", "Sat Mfg "),
            stackgroup="one",
            fillcolor=CATEGORICAL[i % len(CATEGORICAL)],
            line=dict(width=0.5, color=CATEGORICAL[i % len(CATEGORICAL)]),
        ))

    layout = _base_layout("Capital Deployment by Workstream ($M)")
    layout["yaxis"]["title"] = "$M"
    fig.update_layout(**layout)
    return fig


def scenario_comparison_chart(scenarios: list[dict]) -> go.Figure:
    """Bar chart comparing key metrics across scenarios."""
    names = [s["name"] for s in scenarios]
    npvs = [s["npv_m"] for s in scenarios]
    irrs = [s["irr_pct"] for s in scenarios]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=npvs, name="NPV ($M)",
        marker_color=COLORS["primary"], opacity=0.85,
    ))

    layout = _base_layout("Scenario Comparison — NPV ($M)")
    layout["yaxis"]["title"] = "NPV ($M)"
    fig.update_layout(**layout)
    return fig


# ── Variance Dashboard Charts ────────────────────────────────────────────────

def variance_waterfall(ytd_df: pd.DataFrame) -> go.Figure:
    """Waterfall chart of YTD variance by workstream."""
    fig = go.Figure()

    ws_names = ytd_df["Workstream"].str.replace("Satellite Manufacturing ", "Sat Mfg ")
    variances = ytd_df["YTD Variance ($M)"].values

    colors_bar = [COLORS["negative"] if v > 0 else COLORS["positive"] for v in variances]

    fig.add_trace(go.Bar(
        x=ws_names,
        y=variances,
        marker_color=colors_bar,
        opacity=0.85,
        text=[f"{'+'if v > 0 else ''}{v:.1f}M" for v in variances],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)

    layout = _base_layout("YTD Variance by Workstream ($M) — Over / (Under)")
    layout["yaxis"]["title"] = "Variance ($M)"
    layout["xaxis"]["tickangle"] = -30
    fig.update_layout(**layout)
    return fig


def variance_heatmap(var_table: pd.DataFrame, through_month: int = 6) -> go.Figure:
    """Heatmap of monthly variance % by workstream."""
    filtered = var_table[var_table["Month_Idx"] < through_month].copy()
    pivot = filtered.pivot_table(index="Workstream", columns="Month", values="Variance (%)")
    # Reorder columns by month
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    pivot.index = pivot.index.str.replace("Satellite Manufacturing ", "Sat Mfg ")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn_r",  # reversed — red = over budget
        zmid=0,
        text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Var %"),
    ))

    layout = _base_layout("Monthly Variance Heatmap (%) — Red = Over Budget", height=380)
    fig.update_layout(**layout)
    return fig


def budget_vs_actual_trend(trend_df: pd.DataFrame, ws_name: str) -> go.Figure:
    """Line chart of budget vs actual for a single workstream."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df["Month"], y=trend_df["Budget ($M)"],
        name="Budget", line=dict(color=COLORS["neutral"], dash="dash", width=2),
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["Month"], y=trend_df["Actual ($M)"],
        name="Actual", line=dict(color=COLORS["secondary"], width=3),
        mode="lines+markers",
    ))

    # Shade variance
    fig.add_trace(go.Scatter(
        x=list(trend_df["Month"]) + list(trend_df["Month"])[::-1],
        y=list(trend_df["Actual ($M)"]) + list(trend_df["Budget ($M)"])[::-1],
        fill="toself",
        fillcolor="rgba(224, 123, 123, 0.08)",
        line=dict(width=0),
        showlegend=False,
    ))

    short_name = ws_name.replace("Satellite Manufacturing ", "Sat Mfg ")
    layout = _base_layout(f"Budget vs. Actual — {short_name} ($M)")
    layout["yaxis"]["title"] = "$M"
    fig.update_layout(**layout)
    return fig


def cumulative_variance_chart(trend_df: pd.DataFrame, ws_name: str) -> go.Figure:
    """Cumulative budget vs actual."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df["Month"], y=trend_df["Cum Budget ($M)"],
        name="Cum Budget", line=dict(color=COLORS["neutral"], dash="dash", width=2),
        fill="tozeroy", fillcolor="rgba(108, 118, 134, 0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["Month"], y=trend_df["Cum Actual ($M)"],
        name="Cum Actual", line=dict(color=COLORS["primary"], width=3),
    ))

    short_name = ws_name.replace("Satellite Manufacturing ", "Sat Mfg ")
    layout = _base_layout(f"Cumulative Spend — {short_name} ($M)")
    layout["yaxis"]["title"] = "$M"
    fig.update_layout(**layout)
    return fig
