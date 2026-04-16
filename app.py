"""
TeraWave Capital Valuation Engine
=================================
Capital allocation analysis for Blue Origin's TeraWave satellite constellation.

All financial data is synthetic and for demonstration purposes only.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import config
from models.financial_model import (
    ScenarioAssumptions,
    build_full_projection,
    compute_npv,
    compute_capex_schedule,
    compute_progress_metrics,
)
from models.monte_carlo import run_monte_carlo, tornado_analysis
from models.capex_workflow import (
    CapExRequest,
    run_capex_workflow,
    get_historical_df,
    get_budget_summary,
    BUDGET_POOLS,
    PRIORITY_TAGS,
    URGENCY_LEVELS,
)
from models.rag_engine import (
    search_documents,
    get_all_documents,
    format_context_for_llm,
)
from agents.agentic_engine import (
    run_capex_analysis,
    run_general_query,
    AgentStep,
)
from models.variance_engine import (
    build_variance_table,
    detect_anomalies,
    get_ytd_summary,
    get_monthly_trend,
    WORKSTREAMS as VARIANCE_WORKSTREAMS,
)
from utils.charts import (
    cash_flow_waterfall,
    cumulative_investment_chart,
    npv_distribution_chart,
    tornado_chart,
    progress_per_dollar_chart,
    capex_by_workstream_chart,
    variance_waterfall,
    variance_heatmap,
    budget_vs_actual_trend,
    cumulative_variance_chart,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TeraWave Capital Valuation Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Layout */
    .block-container { padding-top: 2.25rem; padding-bottom: 4rem; max-width: 1400px; }

    /* Typography */
    html, body, [class*="st-"], .stMarkdown, .stText {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 1.9rem; margin-bottom: 0; color: #E5E7EB; }
    h2 { font-weight: 500; letter-spacing: -0.01em; font-size: 1.3rem; color: #E5E7EB; margin-top: 0.5rem; }
    h3 { font-weight: 500; letter-spacing: -0.01em; font-size: 1.05rem; color: #E5E7EB; }

    .subtitle { color: #8B95A7; font-size: 0.95rem; font-weight: 400; margin-top: 2px; }
    .context-chips { color: #6C7686; font-size: 0.78rem; margin-top: 6px; letter-spacing: 0.02em; }
    .context-chips span { margin-right: 14px; }
    .context-chips .dot { color: #3A4452; margin: 0 8px; }

    .disclaimer-note {
        color: #8B95A7; font-size: 0.78rem; margin: 18px 0 24px 0;
        border-left: 2px solid #2A3441; padding: 4px 0 4px 12px;
    }

    .section-caption { color: #8B95A7; font-size: 0.88rem; margin-bottom: 1rem; max-width: 760px; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 14px 18px;
        border-radius: 6px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #8B95A7; font-size: 0.72rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; color: #E5E7EB; }
    div[data-testid="stMetricDelta"] { font-size: 0.78rem; }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.9rem; font-weight: 500; color: #8B95A7;
        padding-top: 14px; padding-bottom: 14px;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: #E5E7EB; }
    div[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.06); }
    div[data-baseweb="tab-highlight"] { background-color: #4A9EFF; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B1119;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] h2 {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #8B95A7;
    }
    [data-testid="stSidebar"] .stCaption { color: #6C7686; font-size: 0.78rem; }

    /* Dividers */
    hr { opacity: 0.12; margin: 1rem 0; border-color: #2A3441; }

    /* Inline code and tags */
    code { background: rgba(74, 158, 255, 0.08); color: #9FC5FF;
        padding: 1px 6px; border-radius: 3px; font-size: 0.82em;
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    }

    /* Tool reference pill used in agent reasoning output */
    .tool-ref {
        display: inline-block; color: #9FC5FF; background: rgba(74, 158, 255, 0.08);
        border: 1px solid rgba(74, 158, 255, 0.2); padding: 1px 8px; border-radius: 3px;
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
        font-size: 0.78rem; margin-right: 4px;
    }

    /* Footer */
    .footer {
        text-align: center; color: #6C7686; font-size: 0.75rem;
        margin-top: 2rem; letter-spacing: 0.02em;
    }

    /* Dataframe */
    .stDataFrame { border: 1px solid rgba(255,255,255,0.05); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<h1>TeraWave Capital Valuation Engine</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Capital allocation analysis for Blue Origin\'s TeraWave satellite constellation</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="context-chips">'
    '<span>12-year horizon</span><span class="dot">·</span>'
    '<span>12% WACC base</span><span class="dot">·</span>'
    '<span>5,408-satellite LEO/MEO constellation</span><span class="dot">·</span>'
    '<span>Monte Carlo + agentic review</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer-note">'
    'All financial figures shown here are synthetic and illustrative. '
    'Nothing in this tool reflects actual Blue Origin financials, projections, or internal data. '
    'Built as a technical demonstration.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Scenario")
    st.caption("Adjust sensitivities relative to the base case.")

    capex_mult = st.slider("CapEx multiplier", 0.7, 1.5, 1.0, 0.05,
                           help="1.0 = base. Values >1 model cost overrun, <1 model cost savings.")
    rev_mult = st.slider("Revenue multiplier", 0.5, 1.5, 1.0, 0.05,
                         help="Scales the full revenue ramp schedule.")
    opex_mult = st.slider("OpEx multiplier", 0.7, 1.3, 1.0, 0.05)
    wacc = st.slider("WACC (%)", 8, 18, int(config.WACC * 100), 1) / 100.0

    st.markdown("## Timeline")
    timeline_shift = st.slider("Timeline shift (years)", -1, 3, 0,
                               help="Positive = delay. Negative = acceleration.")
    cadence = st.selectbox("Deployment cadence",
                           ["baseline", "aggressive", "conservative"])

    st.markdown("## Monte Carlo")
    n_sims = st.select_slider("Simulations", [500, 1000, 2500, 5000, 10000], value=2500)
    run_mc = st.button("Run simulation", type="primary", use_container_width=True)

    with st.expander("Edit underlying assumptions", expanded=False):
        st.caption("Multipliers above still apply on top of these values.")

        st.markdown("**CapEx by workstream ($M)**")
        capex_overrides = {}
        for item_name, item in config.CAPEX_ITEMS.items():
            default = item.get("total_cost_m", item.get("unit_cost_m", 0) * item.get("units", 0))
            short_name = item_name.replace("Satellite Manufacturing ", "Sat Mfg ")
            val = st.number_input(
                short_name, min_value=0.0, value=float(default), step=50.0,
                key=f"capex_{item_name}", format="%.0f",
            )
            if val != default:
                capex_overrides[item_name] = val

        st.markdown("**Annual OpEx ($M/yr)**")
        opex_val = st.number_input(
            "Base", min_value=0.0, value=float(config.ANNUAL_OPEX_M),
            step=25.0, key="opex_base", format="%.0f",
        )
        opex_override = opex_val if opex_val != config.ANNUAL_OPEX_M else None

        st.markdown("**Revenue ramp ($M by year)**")
        revenue_overrides = {}
        for yr, rev in config.REVENUE_RAMP.items():
            val = st.number_input(
                f"Year {yr} ({2025 + yr})", min_value=0.0, value=float(rev),
                step=100.0, key=f"rev_{yr}", format="%.0f",
            )
            if val != rev:
                revenue_overrides[yr] = val

    st.caption(" ")
    st.caption("Synthetic data · Built by Mert Kocyigit · Python / Streamlit / Anthropic")

# ── Build scenario ───────────────────────────────────────────────────────────
assumptions = ScenarioAssumptions(
    name="Interactive Scenario",
    capex_multiplier=capex_mult,
    revenue_multiplier=rev_mult,
    timeline_shift_years=timeline_shift,
    wacc=wacc,
    opex_multiplier=opex_mult,
    deployment_cadence=cadence,
    capex_overrides=capex_overrides,
    revenue_overrides=revenue_overrides,
    opex_override=opex_override,
)

projection = build_full_projection(assumptions)
metrics = compute_npv(projection, assumptions)
capex_df = compute_capex_schedule(assumptions)
progress_df = compute_progress_metrics(assumptions)

# ── Executive Summary ────────────────────────────────────────────────────────
st.markdown("### Base case — deterministic")
st.markdown(
    '<p class="section-caption">Discounted cash flow over the 12-year projection horizon, '
    'applying the current sidebar scenario.</p>',
    unsafe_allow_html=True,
)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"NPV @ {wacc:.0%}", f"${metrics['npv_m']:,.0f}M",
          delta="Positive" if metrics['npv_m'] > 0 else "Negative",
          delta_color="normal" if metrics['npv_m'] > 0 else "inverse")
m2.metric("IRR", f"{metrics['irr_pct']:.1f}%" if metrics['irr_pct'] else "—")
m3.metric("Payback",
          f"{metrics['payback_calendar_year']}" if metrics['payback_year'] else "Beyond horizon")
m4.metric("Total CapEx", f"${metrics['total_capex_m']:,.0f}M")
m5.metric("Peak investment", f"${metrics['peak_investment_m']:,.0f}M")

st.write("")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_val, tab_risk, tab_var, tab_wf, tab_ai = st.tabs([
    "Valuation",
    "Risk & sensitivity",
    "FP&A variance",
    "CapEx workflow",
    "Research & AI",
])

# ── Valuation ────────────────────────────────────────────────────────────────
with tab_val:
    st.markdown(
        '<p class="section-caption">Annual cash flow decomposition, cumulative investment '
        'vs. returns, capital deployment by workstream, and a capital-efficiency view '
        'showing which workstreams generate the most program progress and risk retirement '
        'per dollar spent.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(cash_flow_waterfall(projection), use_container_width=True)
    with col2:
        st.plotly_chart(cumulative_investment_chart(projection), use_container_width=True)

    st.plotly_chart(capex_by_workstream_chart(capex_df), use_container_width=True)
    st.plotly_chart(progress_per_dollar_chart(progress_df), use_container_width=True)

    # ── Scenario comparison ──────────────────────────────────────────────────
    st.markdown("#### Scenario comparison")
    st.markdown(
        '<p class="section-caption">Canonical scenarios benchmarked against the current '
        'sidebar configuration. Each scenario runs the same 12-year DCF with a different '
        'set of sensitivities on CapEx, revenue, OpEx, timeline, and deployment cadence.</p>',
        unsafe_allow_html=True,
    )

    def _run_scenario(name, **overrides):
        sa = ScenarioAssumptions(name=name, **{
            "capex_multiplier": 1.0, "revenue_multiplier": 1.0,
            "opex_multiplier": 1.0, "wacc": config.WACC,
            "timeline_shift_years": 0, "deployment_cadence": "baseline",
            **overrides,
        })
        proj = build_full_projection(sa)
        met = compute_npv(proj, sa)
        return {
            "Scenario": name,
            "NPV ($M)": met["npv_m"],
            "IRR (%)": met["irr_pct"] if met["irr_pct"] is not None else np.nan,
            "Payback": (met["payback_calendar_year"]
                        if met["payback_year"] else "Beyond horizon"),
            "Total CapEx ($M)": met["total_capex_m"],
            "Peak Investment ($M)": met["peak_investment_m"],
        }

    scenarios_raw = [
        _run_scenario("Base case"),
        _run_scenario("Cost overrun", capex_multiplier=1.25, opex_multiplier=1.10),
        _run_scenario("Accelerated", timeline_shift_years=-1,
                      deployment_cadence="aggressive", capex_multiplier=1.05),
        _run_scenario("Bear case", revenue_multiplier=0.70, capex_multiplier=1.15),
    ]
    scenarios_raw.append({
        "Scenario": "Current (sidebar)",
        "NPV ($M)": metrics["npv_m"],
        "IRR (%)": metrics["irr_pct"] if metrics["irr_pct"] is not None else np.nan,
        "Payback": (metrics["payback_calendar_year"]
                    if metrics["payback_year"] else "Beyond horizon"),
        "Total CapEx ($M)": metrics["total_capex_m"],
        "Peak Investment ($M)": metrics["peak_investment_m"],
    })
    scenarios_df = pd.DataFrame(scenarios_raw)

    sc_col1, sc_col2 = st.columns([3, 2])
    with sc_col1:
        import plotly.graph_objects as go
        from utils.charts import COLORS as CHART_COLORS, _base_layout
        bar_colors = [
            CHART_COLORS["neutral"],
            CHART_COLORS["negative"],
            CHART_COLORS["positive"],
            CHART_COLORS["muted"],
            CHART_COLORS["primary"],
        ]
        fig_sc = go.Figure(go.Bar(
            x=scenarios_df["Scenario"],
            y=scenarios_df["NPV ($M)"],
            marker_color=bar_colors,
            text=[f"${v:,.0f}M" for v in scenarios_df["NPV ($M)"]],
            textposition="outside",
            textfont=dict(color=CHART_COLORS["text"], size=11),
        ))
        fig_sc.add_hline(y=0, line_dash="dash",
                         line_color=CHART_COLORS["grid"], opacity=0.8)
        layout = _base_layout("NPV by scenario ($M)", height=360)
        layout["yaxis"]["title"] = "NPV ($M)"
        layout["showlegend"] = False
        fig_sc.update_layout(**layout)
        st.plotly_chart(fig_sc, use_container_width=True)

    with sc_col2:
        display_sc = scenarios_df.copy()
        display_sc["NPV ($M)"] = display_sc["NPV ($M)"].apply(lambda v: f"${v:,.0f}")
        display_sc["IRR (%)"] = display_sc["IRR (%)"].apply(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
        )
        display_sc["Total CapEx ($M)"] = display_sc["Total CapEx ($M)"].apply(
            lambda v: f"${v:,.0f}"
        )
        display_sc["Peak Investment ($M)"] = display_sc["Peak Investment ($M)"].apply(
            lambda v: f"${v:,.0f}"
        )
        st.dataframe(display_sc, use_container_width=True, hide_index=True,
                     height=220)
        st.caption(
            "Base case uses sidebar defaults. Cost overrun models +25% CapEx and +10% OpEx. "
            "Accelerated compresses the timeline by one year with aggressive cadence. "
            "Bear case applies -30% revenue and +15% CapEx."
        )

    with st.expander("Key assumptions"):
        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown(f"""
- **Discount rate (WACC):** {wacc:.1%}
- **Terminal growth:** {config.TERMINAL_GROWTH:.1%}
- **Projection horizon:** {config.PROJECTION_YEARS} years (Year 0 = 2025)
- **Annual OpEx (base):** ${config.ANNUAL_OPEX_M:,.0f}M growing at {config.OPEX_GROWTH_RATE:.0%} / yr
- **Revenue start:** Year {config.REVENUE_START_YEAR}
""")
        with kc2:
            st.markdown(f"""
- **Constellation:** {config.LEO_SATELLITES:,} LEO + {config.MEO_SATELLITES} MEO = {config.TOTAL_SATELLITES:,} satellites
- **Launch cadence:** ~60 satellites per New Glenn launch
- **CapEx uncertainty (per workstream):** ±10–30% modeled in Monte Carlo
- **Revenue uncertainty:** ±{config.REVENUE_UNCERTAINTY_PCT:.0%} lognormal
- **Scenario applied:** CapEx ×{capex_mult}, Revenue ×{rev_mult}, OpEx ×{opex_mult}
""")

    with st.expander("Detailed projection"):
        display_df = projection.copy()
        for col in ["capex_m", "opex_m", "revenue_m", "free_cash_flow_m",
                    "cumulative_investment_m", "cumulative_fcf_m", "dcf_m", "cumulative_dcf_m"]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.1f}M")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Risk & Sensitivity ───────────────────────────────────────────────────────
with tab_risk:
    st.markdown(
        '<p class="section-caption">Probabilistic outcomes via Monte Carlo — correlated draws '
        'on CapEx, revenue, and launch cadence — plus one-way sensitivity (tornado) of '
        'NPV to each input.</p>',
        unsafe_allow_html=True,
    )

    if run_mc or "mc_results" in st.session_state:
        if run_mc:
            with st.spinner(f"Running {n_sims:,} simulations..."):
                mc_results = run_monte_carlo(assumptions, n_simulations=n_sims)
                st.session_state["mc_results"] = mc_results
        else:
            mc_results = st.session_state["mc_results"]

        summary = mc_results.summary

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("P(NPV > 0)", f"{summary['prob_positive_npv']:.0f}%")
        mc2.metric("NPV P50", f"${summary['npv_p50_m']:,.0f}M")
        mc3.metric("NPV P10 / P90",
                   f"${summary['npv_p10_m']:,.0f}M / ${summary['npv_p90_m']:,.0f}M")
        mc4.metric("Peak investment P90", f"${summary['peak_investment_p90_m']:,.0f}M")

        st.plotly_chart(
            npv_distribution_chart(mc_results.npv_distribution, assumptions.wacc),
            use_container_width=True,
        )

        col_irr, col_pay = st.columns(2)
        import plotly.graph_objects as go
        from utils.charts import COLORS as CHART_COLORS, _base_layout

        with col_irr:
            valid_irr = mc_results.irr_distribution[mc_results.irr_distribution > -50]
            fig_irr = go.Figure(go.Histogram(
                x=valid_irr, nbinsx=50, marker_color=CHART_COLORS["primary"], opacity=0.8,
            ))
            fig_irr.add_vline(
                x=assumptions.wacc * 100, line_dash="dash",
                line_color=CHART_COLORS["accent"],
                annotation_text=f"WACC {assumptions.wacc:.0%}",
                annotation_font=dict(color=CHART_COLORS["muted"]),
            )
            layout = _base_layout("IRR distribution (%)", height=340)
            layout["xaxis"]["title"] = "IRR (%)"
            layout["showlegend"] = False
            fig_irr.update_layout(**layout)
            st.plotly_chart(fig_irr, use_container_width=True)

        with col_pay:
            fig_pay = go.Figure(go.Histogram(
                x=mc_results.payback_distribution, nbinsx=30,
                marker_color=CHART_COLORS["positive"], opacity=0.8,
            ))
            layout = _base_layout("Payback distribution (years from start)", height=340)
            layout["xaxis"]["title"] = "Years"
            layout["showlegend"] = False
            fig_pay.update_layout(**layout)
            st.plotly_chart(fig_pay, use_container_width=True)

        tornado_df = tornado_analysis(assumptions)
        st.plotly_chart(tornado_chart(tornado_df), use_container_width=True)

        with st.expander("Simulation methodology"):
            st.markdown(f"""
- **{len(mc_results.npv_distribution):,} simulations**, correlated draws across CapEx workstreams,
  revenue ramp, and launch cadence.
- **CapEx:** ±10–30% lognormal by workstream (see `config.CAPEX_ITEMS` for per-item σ).
- **Revenue:** ±{config.REVENUE_UNCERTAINTY_PCT:.0%} lognormal applied to the ramp schedule.
- **Correlation structure:** positive correlation across satellite manufacturing lines
  (shared supply chain) and negative correlation between delay and revenue timing.
- **Tornado:** one-way ±25% perturbation of each input; NPV re-computed per perturbation.
""")
        with st.expander("Full simulation summary"):
            st.json(summary)

    else:
        st.info("Press **Run simulation** in the sidebar to generate Monte Carlo outputs.")

# ── FP&A Variance ────────────────────────────────────────────────────────────
with tab_var:
    st.markdown(
        '<p class="section-caption">Monthly budget-versus-actual across workstreams with '
        'automated threshold anomaly detection and optional AI-generated variance commentary. '
        'Represents the monthly FP&A close process as a single dashboard.</p>',
        unsafe_allow_html=True,
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    through_month = st.select_slider(
        "Through period",
        options=list(range(1, 13)),
        value=6,
        format_func=lambda x: f"{month_labels[x-1]} 2026",
    )

    ytd = get_ytd_summary(through_month)
    total_budget = ytd["YTD Budget ($M)"].sum()
    total_actual = ytd["YTD Actual ($M)"].sum()
    total_var = total_actual - total_budget
    total_var_pct = (total_var / total_budget * 100) if total_budget > 0 else 0

    vm1, vm2, vm3, vm4 = st.columns(4)
    vm1.metric("YTD budget", f"${total_budget:,.0f}M")
    vm2.metric("YTD actual", f"${total_actual:,.0f}M")
    vm3.metric("Variance", f"${total_var:+,.1f}M",
               delta=f"{total_var_pct:+.1f}%", delta_color="inverse")
    over_count = len(ytd[ytd["YTD Variance (%)"] > 5])
    under_count = len(ytd[ytd["YTD Variance (%)"] < -5])
    vm4.metric("Threshold breaches", f"{over_count} over / {under_count} under",
               help="Workstreams outside ±5% of YTD budget")

    vc1, vc2 = st.columns(2)
    with vc1:
        st.plotly_chart(variance_waterfall(ytd), use_container_width=True)
    with vc2:
        var_table = build_variance_table()
        st.plotly_chart(variance_heatmap(var_table, through_month), use_container_width=True)

    alerts = detect_anomalies(through_month)
    if alerts:
        st.markdown(f"#### Automated alerts ({len(alerts)})")
        for alert in alerts:
            sev_tag = "Critical" if alert.severity == "critical" else "Warning"
            with st.expander(
                f"[{sev_tag}] {alert.workstream} — {alert.month}: "
                f"{alert.variance_pct:+.0f}% (${alert.variance_m:+.1f}M)",
                expanded=(alert.severity == "critical"),
            ):
                st.markdown(f"**Commentary:** {alert.auto_commentary}")
                st.caption(
                    f"Budget ${alert.budget_m:.1f}M · Actual ${alert.actual_m:.1f}M · "
                    f"Type: {alert.alert_type.replace('_', ' ').title()}"
                )

    st.markdown("#### Workstream drill-down")
    selected_ws = st.selectbox("Workstream", VARIANCE_WORKSTREAMS, label_visibility="collapsed")
    trend = get_monthly_trend(selected_ws)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.plotly_chart(budget_vs_actual_trend(trend, selected_ws), use_container_width=True)
    with dc2:
        st.plotly_chart(cumulative_variance_chart(trend, selected_ws), use_container_width=True)

    with st.expander("Full YTD summary table"):
        st.dataframe(ytd, use_container_width=True, hide_index=True)

    if config.ANTHROPIC_API_KEY:
        st.divider()
        if st.button("Generate AI variance commentary", key="var_ai"):
            with st.spinner("Drafting variance commentary..."):
                from agents.orchestrator import _call_agent
                var_context = (
                    f"YTD Variance Summary (through {month_labels[through_month-1]} 2026):\n\n"
                    + ytd.to_string(index=False)
                    + f"\n\nTotal Program Variance: ${total_var:+.1f}M ({total_var_pct:+.1f}%)\n\n"
                    + "Alerts:\n"
                    + "\n".join(
                        f"- {a.workstream} ({a.month}): {a.variance_pct:+.0f}% — {a.auto_commentary}"
                        for a in alerts[:8]
                    )
                )
                variance_prompt = (
                    "You are a senior FP&A analyst writing the monthly variance commentary for "
                    "Blue Origin's TeraWave program. Write a concise executive summary (3-4 "
                    "paragraphs) covering: (1) overall program spend status vs. plan, (2) top "
                    "2-3 workstreams requiring attention with specific numbers, (3) reallocation "
                    "recommendations where underspend can offset overspend, (4) forward outlook — "
                    "will the program land within full-year budget. Frame overspends through the "
                    "'accelerating progress' lens where justified. All data is synthetic for "
                    "demonstration."
                )
                ai_commentary = _call_agent(variance_prompt, "Generate the monthly variance report.", var_context)
                st.markdown("#### Commentary")
                st.markdown(ai_commentary)

# ── CapEx Workflow ───────────────────────────────────────────────────────────
with tab_wf:
    st.markdown(
        '<p class="section-caption">Submit a CapEx request. An agent evaluates it against '
        'budget availability, historical precedent, financial impact, policy, and YTD variance, '
        'then returns a recommendation with the full reasoning chain exposed.</p>',
        unsafe_allow_html=True,
    )

    wf_col1, wf_col2 = st.columns([2, 1])

    with wf_col1:
        st.markdown("#### New request")
        with st.form("capex_form"):
            req_title = st.text_input(
                "Title",
                placeholder="e.g., Additional optical terminal test units",
            )
            req_desc = st.text_area(
                "Description",
                placeholder="Brief description of what is being requested",
                height=80,
            )
            fc1, fc2 = st.columns(2)
            req_pool = fc1.selectbox("Budget pool", list(BUDGET_POOLS.keys()))
            req_amount = fc2.number_input("Amount ($M)", min_value=0.1, max_value=500.0,
                                          value=10.0, step=0.5)
            fc3, fc4, fc5 = st.columns(3)
            req_priority = fc3.selectbox("Priority", PRIORITY_TAGS)
            req_urgency = fc4.selectbox("Urgency", URGENCY_LEVELS)
            req_months = fc5.number_input("Completion (mo.)", min_value=1, max_value=60, value=12)
            req_justification = st.text_area(
                "Justification — how does this accelerate progress or retire risk?",
                placeholder="Explain the investment rationale, progress contribution, and risk retired...",
                height=100,
            )
            req_requestor = st.text_input("Requestor", value="Demo User")
            submitted = st.form_submit_button("Run analysis", type="primary",
                                              use_container_width=True)

    with wf_col2:
        st.markdown("#### Budget pools")
        budget_df = get_budget_summary()
        for _, row in budget_df.iterrows():
            pool_name = row["Budget Pool"].replace("TeraWave — ", "")
            utilization = float(row["Utilization"].strip("%"))
            if utilization < 70:
                status = "OK"
            elif utilization < 90:
                status = "Watch"
            else:
                status = "At risk"
            st.markdown(
                f"**{pool_name}**  \n"
                f"<span style='color:#8B95A7; font-size:0.82rem;'>"
                f"${row['Available ($M)']:,.0f}M available · {row['Utilization']} used · {status}"
                f"</span>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown("#### Agent toolset")
        tool_list = [
            ("check_budget", "Budget pool status"),
            ("search_documents", "Policy and precedent lookup"),
            ("get_comparable_requests", "Historical precedent"),
            ("run_financial_impact", "NPV / ROI analysis"),
            ("check_approval_routing", "Approval tier and SLA"),
            ("get_variance_status", "YTD variance check"),
        ]
        for name, desc in tool_list:
            st.markdown(
                f"<span class='tool-ref'>{name}</span> "
                f"<span style='color:#8B95A7; font-size:0.82rem;'>{desc}</span>",
                unsafe_allow_html=True,
            )

    if submitted:
        st.divider()
        if not config.ANTHROPIC_API_KEY:
            st.warning(
                "`ANTHROPIC_API_KEY` not set. Showing the rule-based workflow instead."
            )
            request = CapExRequest(
                id=f"CR-2026-{np.random.randint(100, 999)}",
                title=req_title,
                description=req_desc,
                requestor=req_requestor,
                department="TeraWave Program",
                budget_pool=req_pool,
                amount_m=req_amount,
                priority_tag=req_priority,
                urgency=req_urgency,
                justification=req_justification,
                expected_completion_months=req_months,
                submission_date="2026-04-12",
            )
            result = run_capex_workflow(request)
            st.session_state["wf_result_fallback"] = result
        else:
            agent_status = st.status("Analysis in progress...", expanded=True)
            with agent_status:
                st.caption(
                    f"Request: {req_title} · ${req_amount}M · {req_pool} · "
                    f"{req_priority} · {req_urgency}"
                )

                def on_agent_step(step: AgentStep):
                    if step.step_type == "thinking":
                        preview = step.content[:500] + ("..." if len(step.content) > 500 else "")
                        st.markdown(f"**Reasoning**  \n{preview}")
                    elif step.step_type == "tool_call":
                        st.markdown(
                            f"**Tool call** <span class='tool-ref'>{step.tool_name}</span>",
                            unsafe_allow_html=True,
                        )
                        st.code(step.content, language="json")
                    elif step.step_type == "tool_result":
                        with st.expander(f"Result — {step.tool_name}", expanded=False):
                            try:
                                st.json(json.loads(step.content))
                            except Exception:
                                st.text(step.content)

                agent_result = run_capex_analysis(
                    title=req_title,
                    description=req_desc,
                    amount_m=req_amount,
                    budget_pool=req_pool,
                    priority_tag=req_priority,
                    urgency=req_urgency,
                    justification=req_justification,
                    completion_months=req_months,
                    requestor=req_requestor,
                    on_step=on_agent_step,
                )

                if agent_result.error:
                    agent_status.update(label="Error during analysis", state="error")
                else:
                    agent_status.update(
                        label=f"Complete — {agent_result.total_tool_calls} tool calls "
                              f"in {agent_result.total_duration_ms / 1000:.1f}s",
                        state="complete",
                    )
            st.session_state["agent_result"] = agent_result

    if "agent_result" in st.session_state:
        agent_result = st.session_state["agent_result"]
        if not agent_result.error:
            ar1, ar2, ar3 = st.columns(3)
            ar1.metric("Tool calls", agent_result.total_tool_calls)
            ar2.metric("Reasoning rounds",
                       sum(1 for s in agent_result.steps if s.step_type == "thinking"))
            ar3.metric("Duration", f"{agent_result.total_duration_ms / 1000:.1f}s")

            st.markdown("#### Recommendation")
            st.markdown(agent_result.final_answer)

            with st.expander("Full reasoning chain"):
                for step in agent_result.steps:
                    if step.step_type == "tool_call":
                        st.markdown(
                            f"**Tool call** <span class='tool-ref'>{step.tool_name}</span>",
                            unsafe_allow_html=True,
                        )
                        st.code(step.content, language="json")
                    elif step.step_type == "tool_result":
                        st.markdown(f"**Result** — {step.tool_name}")
                        try:
                            st.json(json.loads(step.content))
                        except Exception:
                            st.text(step.content[:500])
                    elif step.step_type == "thinking":
                        st.markdown(f"**Reasoning**  \n{step.content[:300]}"
                                    + ("..." if len(step.content) > 300 else ""))
                    st.divider()
        else:
            st.error(f"Agent error: {agent_result.error}")

    if "wf_result_fallback" in st.session_state:
        result = st.session_state["wf_result_fallback"]
        st.divider()
        st.markdown(f"#### Recommendation: {result.recommendation}")
        st.markdown(f"*{result.recommendation_rationale}*")

        if result.conditions:
            st.markdown("**Conditions**")
            for cond in result.conditions:
                st.markdown(f"- {cond}")

        wr1, wr2, wr3, wr4, wr5 = st.columns(5)
        wr1.metric("NPV impact", f"${result.npv_impact_m:,.1f}M")
        wr2.metric("ROI", f"{result.roi_pct:.0f}%")
        wr3.metric("Payback", f"{result.payback_months} mo")
        wr4.metric("Approval tier", f"T{result.approval_tier}")
        wr5.metric("SLA", f"{result.sla_days} days")

        with st.expander("Workflow audit trail"):
            for step in result.workflow_steps:
                st.markdown(f"**{step['step']}** — {step['status']}")
                st.caption(step["details"])

    with st.expander("Historical CapEx requests"):
        st.dataframe(get_historical_df(), use_container_width=True, hide_index=True)

# ── Research & AI ────────────────────────────────────────────────────────────
with tab_ai:
    st.markdown(
        '<p class="section-caption">Ask questions grounded in the program\'s document corpus '
        '(contracts, vendor memos, policy, technical reports) and live scenario data. The '
        'agent retrieves evidence, calls analytical tools as needed, and cites sources.</p>',
        unsafe_allow_html=True,
    )

    all_docs = get_all_documents()
    with st.expander(f"Document library ({len(all_docs)} documents)", expanded=False):
        doc_type_labels = {
            "contract": "Contract", "memo": "Memo", "report": "Report",
            "policy": "Policy", "proposal": "Proposal",
        }
        doc_rows = []
        for doc in all_docs:
            label = doc_type_labels.get(doc.doc_type, "Document")
            meta_str = ""
            if doc.metadata:
                meta_str = " · ".join(f"{k}: {v}" for k, v in list(doc.metadata.items())[:3])
            doc_rows.append({
                "Type": label,
                "Title": doc.title,
                "Source": doc.source,
                "Date": doc.date,
                "Chars": f"{len(doc.content):,}",
                "Metadata": meta_str,
            })
        st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "tool_calls" in msg and msg["tool_calls"] > 0:
                st.caption(f"{msg['tool_calls']} tool calls · {msg.get('duration', '')}")
            st.markdown(msg["content"])

    if not st.session_state.messages:
        st.markdown("**Try asking**")
        suggestions = [
            "What's the budget status for Ground Segment & Gateways, and are there any variance concerns?",
            "Which workstreams should we accelerate investment in based on progress-per-dollar?",
            "Summarize the key SLA terms in the DataLink ground services contract.",
            "What approval tier would a $25M critical-path request route to?",
        ]
        row1 = st.columns(2)
        row2 = st.columns(2)
        for i, sug in enumerate(suggestions):
            row = row1 if i < 2 else row2
            if row[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["pending_query"] = sug
                st.rerun()

    if "pending_query" in st.session_state:
        prompt = st.session_state.pop("pending_query")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if prompt := st.chat_input("Ask about the program — the agent will gather data autonomously"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if (st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and (len(st.session_state.messages) < 2
                 or st.session_state.messages[-2].get("role") != "assistant")):
        user_msg = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant"):
            if not config.ANTHROPIC_API_KEY:
                st.warning(
                    "Set `ANTHROPIC_API_KEY` to enable the AI agent. "
                    "Example: `export ANTHROPIC_API_KEY=sk-ant-...`"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "_API key not configured._",
                })
            else:
                agent_status = st.status("Analyzing...", expanded=True)
                with agent_status:
                    def on_console_step(step):
                        if step.step_type == "tool_call":
                            st.markdown(
                                f"Calling <span class='tool-ref'>{step.tool_name}</span>",
                                unsafe_allow_html=True,
                            )
                            st.code(step.content, language="json")
                        elif step.step_type == "tool_result":
                            with st.expander(f"Result — {step.tool_name}", expanded=False):
                                try:
                                    st.json(json.loads(step.content))
                                except Exception:
                                    st.text(step.content[:300])

                    agent_result = run_general_query(user_msg, on_step=on_console_step)

                if agent_result.error:
                    agent_status.update(label="Error", state="error")
                    st.error(agent_result.error)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {agent_result.error}",
                    })
                else:
                    agent_status.update(
                        label=f"Complete — {agent_result.total_tool_calls} tool calls · "
                              f"{agent_result.total_duration_ms / 1000:.1f}s",
                        state="complete",
                    )
                    st.caption(
                        f"{agent_result.total_tool_calls} tool calls · "
                        f"{agent_result.total_duration_ms / 1000:.1f}s"
                    )
                    st.markdown(agent_result.final_answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": agent_result.final_answer,
                        "tool_calls": agent_result.total_tool_calls,
                        "duration": f"{agent_result.total_duration_ms / 1000:.1f}s",
                    })

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Synthetic data · Built with Python, Streamlit, and Anthropic Claude'
    '</div>',
    unsafe_allow_html=True,
)
