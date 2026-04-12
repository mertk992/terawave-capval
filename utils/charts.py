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
    "primary": "#0066CC",
    "secondary": "#00A3E0",
    "accent": "#FF6B35",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
    "bg": "#0E1117",
    "text": "#FAFAFA",
    "grid": "#1E2530",
}


def _base_layout(title: str, height: int = 450) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        margin=dict(l=60, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
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
        fillcolor="rgba(231, 76, 60, 0.15)",
        line=dict(color=COLORS["negative"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=projection["calendar_year"],
        y=projection["cumulative_fcf_m"],
        name="Cumulative FCF",
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.15)",
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
    colors = px.colors.qualitative.Set2

    for i, col in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[col],
            name=col.replace("Satellite Manufacturing ", "Sat Mfg "),
            stackgroup="one",
            fillcolor=colors[i % len(colors)],
            line=dict(width=0.5),
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


# ── M&A Charts ───────────────────────────────────────────────────────────────

def mna_radar_chart(result: dict) -> go.Figure:
    """Radar chart comparing Build vs Acquire vs Partner across scoring dimensions."""
    categories = ["Cost", "Time", "Risk", "Dependency"]
    fig = go.Figure()

    for option, color in [("build", COLORS["primary"]), ("acquire", COLORS["accent"]), ("partner", COLORS["positive"])]:
        scores = result[option]["scores"]
        values = [scores["cost_score"], scores["time_score"], scores["risk_score"], scores["dependency_score"]]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=option.title(),
            line=dict(color=color),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS["grid"]),
            angularaxis=dict(gridcolor=COLORS["grid"]),
        ),
        title=dict(text=f"Build vs Buy vs Partner — {result['target']}", font=dict(size=16, color=COLORS["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=50, b=40),
    )
    return fig


def mna_cost_comparison_chart(result: dict) -> go.Figure:
    """Bar chart comparing total cost across options."""
    options = ["Build", "Acquire", "Partner"]
    costs = [
        result["build"]["npv_cost_m"],
        result["acquire"]["upfront_cost_m"],
        result["partner"]["npv_cost_m"],
    ]
    times = [
        result["build"]["time_to_capability_yrs"],
        result["acquire"]["time_to_capability_yrs"],
        result["partner"]["time_to_capability_yrs"],
    ]
    composites = [result[o.lower()]["scores"]["composite"] for o in options]

    colors_list = [COLORS["primary"], COLORS["accent"], COLORS["positive"]]
    recommended = result["recommended_option"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=options, y=costs, name="NPV Cost ($M)",
        marker_color=colors_list, opacity=0.85,
        text=[f"{'★ ' if o == recommended else ''}{c:,.0f}M<br>{t:.1f} yrs" for o, c, t in zip(options, costs, times)],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))

    layout = _base_layout(f"Cost Comparison — {result['target']}")
    layout["yaxis"]["title"] = "NPV Cost ($M)"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


# ── Portfolio / Scenario Charts ──────────────────────────────────────────────

def portfolio_heatmap(summary_df: pd.DataFrame) -> go.Figure:
    """Heatmap of program NPVs across scenarios."""
    pivot = summary_df.pivot_table(index="Program", columns="Scenario", values="NPV ($M)")
    # Reorder columns
    col_order = ["Bull Case", "Base Case", "Bear Case", "Stress Case"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn",
        text=[[f"${v:,.0f}M" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="NPV ($M)"),
    ))

    layout = _base_layout("Program NPV Across Macro Scenarios ($M)", height=380)
    fig.update_layout(**layout)
    return fig


def portfolio_allocation_chart(alloc_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing optimal capital allocation."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=alloc_df["Program"],
        x=alloc_df["Total CapEx ($M)"],
        orientation="h",
        name="Total CapEx Required",
        marker_color=COLORS["secondary"],
        opacity=0.85,
        text=alloc_df.apply(lambda r: f"${r['Total CapEx ($M)']:,.0f}M — {r['Funding Allocation']}", axis=1),
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))

    layout = _base_layout("Capital Allocation by Program ($M)")
    layout["xaxis"]["title"] = "$M"
    layout["showlegend"] = False
    layout["margin"]["l"] = 180
    fig.update_layout(**layout)
    return fig


def portfolio_stacked_capex(programs_data: pd.DataFrame) -> go.Figure:
    """Stacked area chart of portfolio-level capex by program over time."""
    pivot = programs_data.groupby(["calendar_year", "program"])["capex_m"].sum().unstack(fill_value=0)

    fig = go.Figure()
    colors = [COLORS["primary"], COLORS["accent"], COLORS["positive"], COLORS["secondary"]]

    for i, col in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[col],
            name=col,
            stackgroup="one",
            line=dict(width=0.5),
            fillcolor=colors[i % len(colors)],
        ))

    layout = _base_layout("Portfolio Capital Deployment Over Time ($M)")
    layout["yaxis"]["title"] = "$M"
    fig.update_layout(**layout)
    return fig
