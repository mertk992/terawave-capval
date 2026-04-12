"""
TeraWave Capital Project Valuation Engine
==========================================
An agentic AI system for capital allocation analysis on Blue Origin's
TeraWave satellite constellation program.

⚠️ ALL FINANCIAL DATA IS SYNTHETIC AND FOR DEMONSTRATION PURPOSES ONLY.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
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
from utils.charts import (
    cash_flow_waterfall,
    cumulative_investment_chart,
    npv_distribution_chart,
    tornado_chart,
    progress_per_dollar_chart,
    capex_by_workstream_chart,
    scenario_comparison_chart,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TeraWave CapVal Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background-color: #1E2530; padding: 12px; border-radius: 8px; }
    .disclaimer {
        background-color: #2C1810; border: 1px solid #FF6B35; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 16px; font-size: 0.85em; color: #FFB088;
    }
    .agent-badge {
        display: inline-block; background: #0066CC; color: white; padding: 2px 10px;
        border-radius: 12px; font-size: 0.8em; font-weight: 600; margin-bottom: 8px;
    }
    .header-subtitle { color: #95A5A6; font-size: 1.1em; margin-top: -10px; }
    div[data-testid="stSidebar"] { background-color: #0a0e14; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("TeraWave Capital Valuation Engine")
st.markdown('<p class="header-subtitle">Agentic AI System for Capital Allocation Analysis &nbsp;|&nbsp; Blue Origin</p>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">⚠️ <strong>SYNTHETIC DATA DISCLAIMER:</strong> '
    'All financial figures in this tool are illustrative and for demonstration purposes only. '
    'They do not represent actual Blue Origin financials, projections, or internal data. '
    'Built as a technical demo of agentic AI applied to capital project valuation.</div>',
    unsafe_allow_html=True,
)

# ── Sidebar: Scenario Controls ───────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Parameters")
    st.caption("Adjust assumptions to explore scenarios")

    st.subheader("📊 Financial Assumptions")
    capex_mult = st.slider("CapEx Multiplier", 0.7, 1.5, 1.0, 0.05,
                           help="1.0 = base case. >1 = cost overrun, <1 = cost savings")
    rev_mult = st.slider("Revenue Multiplier", 0.5, 1.5, 1.0, 0.05,
                         help="1.0 = base case. Scales all revenue projections")
    opex_mult = st.slider("OpEx Multiplier", 0.7, 1.3, 1.0, 0.05)
    wacc = st.slider("WACC / Discount Rate (%)", 8, 18, int(config.WACC * 100), 1)
    wacc = wacc / 100.0

    st.subheader("⏱️ Timeline & Deployment")
    timeline_shift = st.slider("Timeline Shift (years)", -1, 3, 0,
                               help="Positive = delay, negative = acceleration")
    cadence = st.selectbox("Deployment Cadence",
                           ["baseline", "aggressive", "conservative"],
                           help="Aggressive = front-load spend, compress timelines")

    st.subheader("🎲 Monte Carlo")
    n_sims = st.select_slider("Simulations", [500, 1000, 2500, 5000, 10000], value=2500)

    st.divider()
    run_mc = st.button("▶ Run Full Analysis", type="primary", use_container_width=True)

# ── Build Base Scenario ──────────────────────────────────────────────────────
assumptions = ScenarioAssumptions(
    name="Interactive Scenario",
    capex_multiplier=capex_mult,
    revenue_multiplier=rev_mult,
    timeline_shift_years=timeline_shift,
    wacc=wacc,
    opex_multiplier=opex_mult,
    deployment_cadence=cadence,
)

projection = build_full_projection(assumptions)
metrics = compute_npv(projection, assumptions)
capex_df = compute_capex_schedule(assumptions)
progress_df = compute_progress_metrics(assumptions)

# ── Key Metrics Row ──────────────────────────────────────────────────────────
st.subheader("Key Metrics — Deterministic Base Case")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("NPV", f"${metrics['npv_m']:,.0f}M",
          delta=f"{'Positive' if metrics['npv_m'] > 0 else 'Negative'}")
m2.metric("IRR", f"{metrics['irr_pct']:.1f}%" if metrics['irr_pct'] else "N/A")
m3.metric("Payback Year", f"{metrics['payback_calendar_year']}" if metrics['payback_year'] else "Beyond horizon")
m4.metric("Total CapEx", f"${metrics['total_capex_m']:,.0f}M")
m5.metric("Peak Investment", f"${metrics['peak_investment_m']:,.0f}M")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_cashflow, tab_capital, tab_mc, tab_agent = st.tabs([
    "💰 Cash Flow Model",
    "🎯 Capital Efficiency",
    "🎲 Monte Carlo & Risk",
    "🤖 AI Agent Console",
])

# ── Tab 1: Cash Flow ─────────────────────────────────────────────────────────
with tab_cashflow:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(cash_flow_waterfall(projection), use_container_width=True)
    with col2:
        st.plotly_chart(cumulative_investment_chart(projection), use_container_width=True)

    st.plotly_chart(capex_by_workstream_chart(capex_df), use_container_width=True)

    with st.expander("📋 Detailed Projection Table"):
        display_df = projection.copy()
        for col in ["capex_m", "opex_m", "revenue_m", "free_cash_flow_m",
                     "cumulative_investment_m", "cumulative_fcf_m", "dcf_m", "cumulative_dcf_m"]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.1f}M")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Tab 2: Capital Efficiency ────────────────────────────────────────────────
with tab_capital:
    st.markdown("""
    ### Capital Allocation Framework
    > *"The goal is not cost suppression — it's accelerating the deployment of capital where
    > we earn the most progress or retire the most risk per dollar."*

    The chart below maps each workstream by its **progress contribution** and **risk retirement value**
    per billion dollars deployed. Bubble size = total CapEx. Color = composite efficiency score.
    """)

    st.plotly_chart(progress_per_dollar_chart(progress_df), use_container_width=True)

    st.subheader("Workstream Efficiency Rankings")
    ranking_df = progress_df[[
        "workstream", "description", "total_capex_m",
        "progress_per_bn", "risk_retired_per_bn", "composite_score"
    ]].copy()
    ranking_df.columns = [
        "Workstream", "Description", "Total CapEx ($M)",
        "Progress / $B", "Risk Retired / $B", "Composite Score"
    ]
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Sensitivity Analysis")
    tornado_df = tornado_analysis(assumptions)
    st.plotly_chart(tornado_chart(tornado_df), use_container_width=True)

# ── Tab 3: Monte Carlo ──────────────────────────────────────────────────────
with tab_mc:
    if run_mc or "mc_results" in st.session_state:
        if run_mc:
            with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
                mc_results = run_monte_carlo(assumptions, n_simulations=n_sims)
                st.session_state["mc_results"] = mc_results
        else:
            mc_results = st.session_state["mc_results"]

        summary = mc_results.summary

        st.subheader("Probabilistic Outcome Distribution")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("P(NPV > 0)", f"{summary['prob_positive_npv']:.0f}%")
        mc2.metric("NPV P50", f"${summary['npv_p50_m']:,.0f}M")
        mc3.metric("NPV P10 / P90",
                    f"${summary['npv_p10_m']:,.0f}M / ${summary['npv_p90_m']:,.0f}M")
        mc4.metric("Peak Investment P90", f"${summary['peak_investment_p90_m']:,.0f}M")

        st.plotly_chart(
            npv_distribution_chart(mc_results.npv_distribution, assumptions.wacc),
            use_container_width=True,
        )

        col_irr, col_pay = st.columns(2)
        with col_irr:
            valid_irr = mc_results.irr_distribution[mc_results.irr_distribution > -50]
            import plotly.graph_objects as go
            fig_irr = go.Figure(go.Histogram(x=valid_irr, nbinsx=50,
                                              marker_color="#00A3E0", opacity=0.75))
            fig_irr.add_vline(x=assumptions.wacc * 100, line_dash="dash",
                              line_color="#FF6B35",
                              annotation_text=f"WACC ({assumptions.wacc:.0%})")
            fig_irr.update_layout(title="IRR Distribution (%)",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#FAFAFA"),
                                  height=350)
            st.plotly_chart(fig_irr, use_container_width=True)

        with col_pay:
            fig_pay = go.Figure(go.Histogram(x=mc_results.payback_distribution, nbinsx=30,
                                              marker_color="#2ECC71", opacity=0.75))
            fig_pay.update_layout(title="Payback Period Distribution (Years from Start)",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#FAFAFA"),
                                  height=350)
            st.plotly_chart(fig_pay, use_container_width=True)

        with st.expander("📊 Full Simulation Summary"):
            st.json(summary)

    else:
        st.info("Click **▶ Run Full Analysis** in the sidebar to execute Monte Carlo simulations.")

# ── Tab 4: AI Agent Console ──────────────────────────────────────────────────
with tab_agent:
    st.markdown("""
    ### AI Agent Console
    Ask questions and the system will route to the appropriate specialized agent:
    - **Capital Allocation Analyst** — workstream prioritization, deployment strategy
    - **Risk & Scenario Agent** — Monte Carlo interpretation, tail risk analysis
    - **Investment Memo Writer** — generate board-ready investment memos

    *Powered by Claude (Anthropic)*
    """)

    # Build model data context for agents
    model_data = {
        "projection": projection,
        "metrics": metrics,
        "progress_metrics": progress_df,
        "tornado": tornado_df if 'tornado_df' in dir() else tornado_analysis(assumptions),
    }
    if "mc_results" in st.session_state:
        model_data["mc_summary"] = st.session_state["mc_results"].summary

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "agent" in msg:
                st.markdown(f'<span class="agent-badge">{msg["agent"]}</span>',
                            unsafe_allow_html=True)
            st.markdown(msg["content"])

    # Suggested prompts
    if not st.session_state.messages:
        st.markdown("**Suggested queries:**")
        cols = st.columns(3)
        suggestions = [
            "Which workstreams should we accelerate investment in to maximize progress?",
            "What are the top risk factors and how should we invest to retire them?",
            "Generate an investment memo for the TeraWave program.",
        ]
        for i, sug in enumerate(suggestions):
            if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["pending_query"] = sug
                st.rerun()

    # Process pending suggestion
    if "pending_query" in st.session_state:
        prompt = st.session_state.pop("pending_query")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        _process_agent_query(prompt, model_data) if False else None  # handled below

    # Chat input
    if prompt := st.chat_input("Ask about TeraWave capital allocation, risk, or request a memo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Process latest user message
    if (st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and (len(st.session_state.messages) < 2
                 or st.session_state.messages[-2].get("role") != "assistant")):
        user_msg = st.session_state.messages[-1]["content"]

        with st.chat_message("assistant"):
            if not config.ANTHROPIC_API_KEY:
                st.warning(
                    "Set your `ANTHROPIC_API_KEY` environment variable to enable the AI agents. "
                    "Example: `export ANTHROPIC_API_KEY=sk-ant-...`"
                )
                st.markdown("**Demo mode:** Showing agent routing logic. "
                            f"This query would be routed to the appropriate agent based on keywords.")
                # Show which agent would be selected
                from agents.orchestrator import query_router
                question_lower = user_msg.lower()
                if any(kw in question_lower for kw in ["memo", "report", "write", "document", "board", "summary"]):
                    agent_name = "Investment Memo Writer"
                elif any(kw in question_lower for kw in ["risk", "monte carlo", "probability", "tail", "p90", "p10", "simulation", "uncertainty"]):
                    agent_name = "Risk & Scenario Agent"
                else:
                    agent_name = "Capital Allocation Analyst"
                st.markdown(f'<span class="agent-badge">→ {agent_name}</span>', unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"[API key required — would route to **{agent_name}**]",
                    "agent": agent_name,
                })
            else:
                with st.spinner("Agent analyzing..."):
                    from agents.orchestrator import query_router
                    result = query_router(user_msg, model_data)
                    st.markdown(f'<span class="agent-badge">{result["agent"]}</span>',
                                unsafe_allow_html=True)
                    st.markdown(result["response"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "agent": result["agent"],
                    })

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.85em;">'
    '⚠️ All financial data is synthetic and for demonstration purposes only.<br>'
    'TeraWave Capital Valuation Engine — Built with Python, Streamlit, Claude (Anthropic)<br>'
    'Agentic AI × Corporate Finance'
    '</div>',
    unsafe_allow_html=True,
)
