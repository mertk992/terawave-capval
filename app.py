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
    scenario_comparison_chart,
    variance_waterfall,
    variance_heatmap,
    budget_vs_actual_trend,
    cumulative_variance_chart,
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
tab_cashflow, tab_capital, tab_mc, tab_variance, tab_workflow, tab_docs, tab_agent = st.tabs([
    "💰 Cash Flow",
    "🎯 Capital Efficiency",
    "🎲 Monte Carlo",
    "📈 Variance",
    "📋 CapEx Workflow",
    "📄 Doc Intel",
    "🤖 AI Agents",
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

# ── Tab 4: Variance Dashboard ────────────────────────────────────────────────
with tab_variance:
    st.markdown("""
    ### FP&A Variance Dashboard
    Monthly budget vs. actual with **automated anomaly detection** and AI-generated
    variance commentary. This is the monthly close process — automated.
    """)

    through_month = st.select_slider(
        "Reporting Period (through month)",
        options=list(range(1, 13)),
        value=6,
        format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1] + " 2026",
    )

    # YTD Summary
    ytd = get_ytd_summary(through_month)
    total_budget = ytd["YTD Budget ($M)"].sum()
    total_actual = ytd["YTD Actual ($M)"].sum()
    total_var = total_actual - total_budget
    total_var_pct = (total_var / total_budget * 100) if total_budget > 0 else 0

    vm1, vm2, vm3, vm4 = st.columns(4)
    vm1.metric("YTD Budget", f"${total_budget:,.0f}M")
    vm2.metric("YTD Actual", f"${total_actual:,.0f}M")
    vm3.metric("Total Variance", f"${total_var:+,.1f}M", delta=f"{total_var_pct:+.1f}%",
               delta_color="inverse")
    over_count = len(ytd[ytd["YTD Variance (%)"] > 5])
    under_count = len(ytd[ytd["YTD Variance (%)"] < -5])
    vm4.metric("Alerts", f"{over_count} over / {under_count} under")

    # Charts
    vc1, vc2 = st.columns(2)
    with vc1:
        st.plotly_chart(variance_waterfall(ytd), use_container_width=True)
    with vc2:
        var_table = build_variance_table()
        st.plotly_chart(variance_heatmap(var_table, through_month), use_container_width=True)

    # Anomaly alerts
    alerts = detect_anomalies(through_month)
    if alerts:
        st.subheader(f"Automated Variance Alerts ({len(alerts)})")
        for alert in alerts:
            icon = "🔴" if alert.severity == "critical" else "🟡"
            with st.expander(
                f"{icon} {alert.workstream} — {alert.month}: "
                f"{alert.variance_pct:+.0f}% (${alert.variance_m:+.1f}M)",
                expanded=(alert.severity == "critical"),
            ):
                st.markdown(f"**{alert.auto_commentary}**")
                st.markdown(f"Budget: ${alert.budget_m:.1f}M | Actual: ${alert.actual_m:.1f}M | "
                            f"Type: {alert.alert_type.replace('_', ' ').title()}")

    # Workstream drill-down
    st.divider()
    st.subheader("Workstream Drill-Down")
    selected_ws = st.selectbox("Select Workstream", VARIANCE_WORKSTREAMS)
    trend = get_monthly_trend(selected_ws)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.plotly_chart(budget_vs_actual_trend(trend, selected_ws), use_container_width=True)
    with dc2:
        st.plotly_chart(cumulative_variance_chart(trend, selected_ws), use_container_width=True)

    # YTD summary table
    with st.expander("Full YTD Summary Table"):
        st.dataframe(ytd, use_container_width=True, hide_index=True)

    # AI variance commentary
    if config.ANTHROPIC_API_KEY:
        if st.button("🤖 Generate AI Variance Commentary", key="var_ai"):
            with st.spinner("Agent generating variance report..."):
                from agents.orchestrator import _call_agent
                var_context = f"YTD Variance Summary (through {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][through_month-1]} 2026):\n\n"
                var_context += ytd.to_string(index=False)
                var_context += f"\n\nTotal Program Variance: ${total_var:+.1f}M ({total_var_pct:+.1f}%)"
                var_context += "\n\nAlerts:\n"
                for a in alerts[:8]:
                    var_context += f"- {a.workstream} ({a.month}): {a.variance_pct:+.0f}% — {a.auto_commentary}\n"

                variance_prompt = """You are a senior FP&A analyst writing the monthly variance commentary for Blue Origin's TeraWave program.
Write a concise executive summary (3-4 paragraphs) covering:
1. Overall program spend status vs. plan
2. Top 2-3 workstreams requiring attention (with specific numbers)
3. Reallocation recommendations — where can underspend offset overspend?
4. Forward outlook — will the program land within full-year budget?
Frame overspends through the 'accelerating progress' lens where justified.
⚠️ All data is synthetic for demonstration purposes."""

                ai_commentary = _call_agent(variance_prompt, "Generate the monthly variance report.", var_context)
                st.markdown("### AI-Generated Variance Commentary")
                st.markdown(ai_commentary)

# ── Tab 5: CapEx Workflow ────────────────────────────────────────────────────
with tab_workflow:
    st.markdown("""
    ### Capital Expenditure Request Workflow
    Submit a CapEx request and the system automatically **validates → analyzes →
    compares → routes → recommends** — replacing the typical spreadsheet + email chain.
    """)

    wf_col1, wf_col2 = st.columns([2, 1])

    with wf_col1:
        st.subheader("Submit New Request")
        with st.form("capex_form"):
            req_title = st.text_input("Request Title", placeholder="e.g., Additional Optical Terminal Test Units")
            req_desc = st.text_area("Description", placeholder="Brief description of what's being requested", height=80)
            fc1, fc2 = st.columns(2)
            req_pool = fc1.selectbox("Budget Pool", list(BUDGET_POOLS.keys()))
            req_amount = fc2.number_input("Amount ($M)", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
            fc3, fc4, fc5 = st.columns(3)
            req_priority = fc3.selectbox("Priority", PRIORITY_TAGS)
            req_urgency = fc4.selectbox("Urgency", URGENCY_LEVELS)
            req_months = fc5.number_input("Completion (months)", min_value=1, max_value=60, value=12)
            req_justification = st.text_area(
                "Justification — How does this accelerate progress or retire risk?",
                placeholder="Explain why this investment is needed and what progress or risk retirement it enables...",
                height=100,
            )
            req_requestor = st.text_input("Requestor", value="Demo User")
            submitted = st.form_submit_button("▶ Run Workflow", type="primary", use_container_width=True)

    with wf_col2:
        st.subheader("Budget Status")
        budget_df = get_budget_summary()
        for _, row in budget_df.iterrows():
            pool_name = row["Budget Pool"].replace("TeraWave — ", "")
            utilization = float(row["Utilization"].strip("%"))
            color = "🟢" if utilization < 70 else "🟡" if utilization < 90 else "🔴"
            st.markdown(f"{color} **{pool_name}** — ${row['Available ($M)']:,.0f}M avail ({row['Utilization']})")

    # Process workflow
    if submitted:
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
        st.session_state["wf_result"] = result

    if "wf_result" in st.session_state:
        result = st.session_state["wf_result"]
        st.divider()

        # Recommendation banner
        rec = result.recommendation
        rec_colors = {"Approve": "🟢", "Approve with Conditions": "🟡", "Defer": "🟠", "Reject": "🔴"}
        st.subheader(f"{rec_colors.get(rec, '⚪')} Recommendation: {rec}")
        st.markdown(f"*{result.recommendation_rationale}*")

        if result.conditions:
            st.markdown("**Conditions:**")
            for cond in result.conditions:
                st.markdown(f"- {cond}")
        if result.risk_flags:
            st.markdown("**Risk Flags:**")
            for flag in result.risk_flags:
                st.markdown(f"- ⚠️ {flag}")

        # Metrics row
        wr1, wr2, wr3, wr4, wr5 = st.columns(5)
        wr1.metric("NPV Impact", f"${result.npv_impact_m:,.1f}M")
        wr2.metric("ROI", f"{result.roi_pct:.0f}%")
        wr3.metric("Payback", f"{result.payback_months} mo")
        wr4.metric("Approval Tier", f"T{result.approval_tier}: {result.approval_tier_label}")
        wr5.metric("SLA", f"{result.sla_days} days")

        # Progress scores
        ps1, ps2, ps3 = st.columns(3)
        ps1.metric("Progress Score", f"{result.progress_score:.1f}")
        ps2.metric("Risk Retirement Score", f"{result.risk_retirement_score:.1f}")
        ps3.metric("Budget Status", result.budget_status)

        # Workflow audit trail
        st.subheader("Workflow Audit Trail")
        for step in result.workflow_steps:
            icon = "✅" if step["status"] in ["Passed", "Complete", "Within Budget"] else "⚠️" if "Condition" in step.get("status", "") or "Near" in step.get("status", "") else "❌" if step["status"] == "Failed" else "📋"
            with st.expander(f"{icon} {step['step']} — {step['status']}"):
                st.markdown(step["details"])

        # Comparable requests
        if result.comparables:
            st.subheader("Comparable Past Requests")
            comp_df = pd.DataFrame(result.comparables)
            display_cols = ["id", "title", "amount_m", "priority_tag", "status", "outcome"]
            available_cols = [c for c in display_cols if c in comp_df.columns]
            st.dataframe(comp_df[available_cols], use_container_width=True, hide_index=True)

        # AI analysis button
        if config.ANTHROPIC_API_KEY:
            if st.button("🤖 Get AI Analysis of This Request", key="wf_ai"):
                with st.spinner("Agent analyzing..."):
                    from agents.orchestrator import query_capex_workflow
                    wf_context = f"""
Request: {result.request.title}
Amount: ${result.request.amount_m}M | Pool: {result.request.budget_pool}
Priority: {result.request.priority_tag} | Urgency: {result.request.urgency}
Justification: {result.request.justification}
Automated Recommendation: {result.recommendation}
NPV Impact: ${result.npv_impact_m}M | ROI: {result.roi_pct}% | Payback: {result.payback_months} months
Budget Status: {result.budget_status} | Available: ${result.budget_available_m}M
Risk Flags: {'; '.join(result.risk_flags) if result.risk_flags else 'None'}
"""
                    model_data_wf = {"capex_workflow_context": wf_context}
                    ai_response = query_capex_workflow(
                        f"Review this CapEx request and provide your analysis:\n{wf_context}",
                        model_data_wf
                    )
                    st.markdown(ai_response)

    # Historical requests table
    with st.expander("📋 Historical CapEx Requests"):
        st.dataframe(get_historical_df(), use_container_width=True, hide_index=True)

# ── Tab 7: Document Intelligence ─────────────────────────────────────────────
with tab_docs:
    st.markdown("""
    ### Document Intelligence (RAG)
    Search and query across TeraWave program documents — contracts, vendor memos,
    policy docs, technical reports, and pipeline reviews. AI answers are **grounded
    in source documents** with citations.
    """)

    # Document library
    all_docs = get_all_documents()
    doc_col1, doc_col2 = st.columns([1, 2])

    with doc_col1:
        st.subheader("Document Library")
        doc_type_icons = {"contract": "📝", "memo": "📋", "report": "📊", "policy": "📜", "proposal": "💼"}
        for doc in all_docs:
            icon = doc_type_icons.get(doc.doc_type, "📄")
            with st.expander(f"{icon} {doc.title[:50]}..."):
                st.markdown(f"**Type:** {doc.doc_type.title()} | **Source:** {doc.source} | **Date:** {doc.date}")
                if doc.metadata:
                    meta_str = " | ".join(f"{k}: {v}" for k, v in doc.metadata.items())
                    st.markdown(f"**Metadata:** {meta_str}")
                st.markdown(f"*{len(doc.content)} characters*")

    with doc_col2:
        st.subheader("Query Documents")

        # Suggested searches
        if "doc_query" not in st.session_state:
            st.markdown("**Try these queries:**")
            doc_suggestions = [
                "What are the SLA requirements in the DataLink ground services contract?",
                "Which vendor was selected for the MEO satellite bus and why?",
                "What is the capital allocation policy for emergency requests?",
                "What are the key risks for the optical inter-satellite link system?",
                "What is the enterprise customer pipeline value and which segments are largest?",
                "How many launches are needed for the full constellation?",
            ]
            dcols = st.columns(2)
            for i, sug in enumerate(doc_suggestions):
                if dcols[i % 2].button(sug, key=f"doc_sug_{i}", use_container_width=True):
                    st.session_state["doc_query"] = sug
                    st.rerun()

        query = st.text_input("Search documents...", value=st.session_state.get("doc_query", ""),
                              placeholder="Ask anything about TeraWave program documents...")

        if query:
            # Retrieve relevant documents
            results = search_documents(query, top_k=3)

            if results:
                st.markdown(f"**Found {len(results)} relevant document(s)**")

                # Show retrieved chunks with relevance scores
                for doc, score in results:
                    icon = doc_type_icons.get(doc.doc_type, "📄")
                    with st.expander(f"{icon} {doc.title} — Relevance: {score:.0%}", expanded=(score > 0.2)):
                        st.markdown(f"**{doc.doc_type.title()}** | {doc.source} | {doc.date}")
                        # Show content preview (first 1000 chars)
                        preview = doc.content[:1500] + ("..." if len(doc.content) > 1500 else "")
                        st.text(preview)

                # AI-powered Q&A
                st.divider()
                if config.ANTHROPIC_API_KEY:
                    with st.spinner("AI analyzing documents..."):
                        from agents.orchestrator import query_document_qa
                        doc_context = format_context_for_llm(results)
                        answer = query_document_qa(query, doc_context)
                        st.markdown("### AI Answer (Grounded in Documents)")
                        st.markdown(answer)
                else:
                    st.info("Set `ANTHROPIC_API_KEY` to enable AI-powered document Q&A. "
                            "The retrieval system above works without the API key.")
            else:
                st.warning("No relevant documents found. Try a different query.")

# ── Tab 8: AI Agent Console ──────────────────────────────────────────────────
with tab_agent:
    st.markdown("""
    ### AI Agent Console
    Ask questions and the system routes to the appropriate specialized agent:
    - **Capital Allocation Analyst** — workstream prioritization, deployment strategy
    - **Risk & Scenario Agent** — Monte Carlo interpretation, tail risk analysis
    - **CapEx Workflow Analyst** — reviews capital expenditure requests
    - **Document Intelligence** — answers questions grounded in source documents
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
        row1 = st.columns(3)
        suggestions = [
            "Which workstreams should we accelerate investment in?",
            "What are the top risk factors driving NPV variance?",
            "Generate an investment memo for the TeraWave program.",
        ]
        for i, sug in enumerate(suggestions):
            if row1[i].button(sug, key=f"sug_{i}", use_container_width=True):
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
                question_lower = user_msg.lower()
                if any(kw in question_lower for kw in ["memo", "report", "write", "document", "board"]):
                    agent_name = "Investment Memo Writer"
                elif any(kw in question_lower for kw in ["risk", "monte carlo", "probability", "tail", "p90", "p10", "simulation", "uncertainty"]):
                    agent_name = "Risk & Scenario Agent"
                elif any(kw in question_lower for kw in ["capex", "request", "approval", "workflow"]):
                    agent_name = "CapEx Workflow Analyst"
                elif any(kw in question_lower for kw in ["contract", "vendor", "policy", "document", "sla"]):
                    agent_name = "Document Intelligence"
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
