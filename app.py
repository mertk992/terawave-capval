"""
CapExFlow AI
============

Focused Streamlit prototype for finance-led CapEx decision support.
All data is synthetic and for demonstration only.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st


APP_TITLE = "CapExFlow AI"
APP_SUBTITLE = "Finance workflow automation for evidence-backed capital allocation"
DATA_PATH = Path(__file__).parent / "data" / "synthetic_enterprise_data.json"

PRIORITY_TAGS = [
    "Critical Path",
    "Risk Retirement",
    "Cost Reduction",
    "Capacity Expansion",
    "R&D / Innovation",
    "Maintenance",
]
URGENCY_LEVELS = ["Standard", "Expedited", "Emergency"]

DEMO_DAY_SCENARIO = {
    "title": "Critical-Path Launch Acceleration Package",
    "description": "Add integration shifts and readiness work to pull the first operational deployment window forward.",
    "budget_pool": "TeraWave — Launch Services",
    "amount_m": 45.0,
    "priority_tag": "Critical Path",
    "urgency": "Expedited",
    "completion_months": 9,
    "justification": (
        "Funding accelerates launch readiness, reduces schedule risk for initial operational capability, "
        "and creates more progress per dollar than holding the launch cadence flat."
    ),
}

VARIANCE_STATUS = {
    "Launch Services": {
        "workstream": "Launch Services",
        "ytd_budget_m": 115.0,
        "ytd_actual_m": 116.8,
        "ytd_variance_m": 1.8,
        "ytd_variance_pct": 1.6,
        "status": "On plan",
        "commentary": "Minor month-to-month timing shifts; no material overspend signal.",
    },
    "Ground Segment & Gateways": {
        "workstream": "Ground Segment & Gateways",
        "ytd_budget_m": 178.0,
        "ytd_actual_m": 192.5,
        "ytd_variance_m": 14.5,
        "ytd_variance_pct": 8.1,
        "status": "Watch",
        "commentary": "Site acquisition and backhaul timing are creating YTD pressure.",
    },
    "Optical Inter-Satellite Links": {
        "workstream": "Optical Inter-Satellite Links",
        "ytd_budget_m": 98.0,
        "ytd_actual_m": 91.3,
        "ytd_variance_m": -6.7,
        "ytd_variance_pct": -6.8,
        "status": "Underrun",
        "commentary": "Prototype hardware receipts moved later in the year.",
    },
    "Software & Network Ops": {
        "workstream": "Software & Network Ops",
        "ytd_budget_m": 60.0,
        "ytd_actual_m": 63.4,
        "ytd_variance_m": 3.4,
        "ytd_variance_pct": 5.7,
        "status": "Watch",
        "commentary": "Cloud test environments are running above plan.",
    },
    "Satellite Manufacturing (LEO)": {
        "workstream": "Satellite Manufacturing (LEO)",
        "ytd_budget_m": 200.0,
        "ytd_actual_m": 188.2,
        "ytd_variance_m": -11.8,
        "ytd_variance_pct": -5.9,
        "status": "Underrun",
        "commentary": "Supplier NRE milestones are slightly behind billing plan.",
    },
}

st.set_page_config(page_title=APP_TITLE, page_icon="CF", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --panel: #111d2e;
        --panel2: #17263d;
        --border: rgba(143, 179, 229, 0.25);
        --cyan: #35c2ff;
        --blue: #2d7ff9;
        --green: #35d07f;
        --amber: #f5b84b;
        --muted: #a9b8cc;
    }
    .block-container { padding-top: 2rem; max-width: 1240px; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #07111f, #101827); }
    .hero {
        padding: 30px 34px;
        border-radius: 22px;
        background: radial-gradient(circle at top right, rgba(53,194,255,.20), transparent 28%),
                    linear-gradient(135deg, #07111f, #122641 58%, #1b3352);
        border: 1px solid var(--border);
        box-shadow: 0 22px 80px rgba(0,0,0,.30);
        margin-bottom: 22px;
    }
    .hero h1 { font-size: 3rem; letter-spacing: -.055em; line-height: 1.02; margin: 0 0 10px; }
    .hero p { color: #d8e6fb; font-size: 1.05rem; max-width: 850px; margin: 0; }
    .pill {
        display: inline-flex; padding: 5px 10px; border-radius: 999px;
        border: 1px solid rgba(53,194,255,.30); background: rgba(53,194,255,.09);
        color: #cfeeff; font-size: .78rem; font-weight: 750; letter-spacing: .06em;
        text-transform: uppercase; margin-bottom: 14px;
    }
    .step-row { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin: 16px 0 8px; }
    .card, .step-card, .evidence-card {
        border: 1px solid var(--border); border-radius: 16px;
        background: linear-gradient(180deg, rgba(23,35,55,.96), rgba(12,22,36,.96));
        padding: 15px; min-height: 100%;
    }
    .step-num, .small-heading { color: var(--cyan); font-size: .75rem; font-weight: 850; letter-spacing: .08em; text-transform: uppercase; }
    .step-title { margin-top: 4px; color: #f7fbff; font-weight: 780; }
    .step-copy, .muted { color: var(--muted); font-size: .88rem; }
    .tool-chip {
        display: inline-block; color: #d7efff; background: rgba(45,127,249,.12);
        border: 1px solid rgba(45,127,249,.28); border-radius: 999px;
        padding: 4px 9px; font-size: .78rem; font-family: ui-monospace, Menlo, Consolas, monospace;
        margin: 0 5px 6px 0;
    }
    .recommendation {
        border-radius: 20px; padding: 24px 26px; border: 1px solid rgba(245,184,75,.35);
        background: radial-gradient(circle at top right, rgba(245,184,75,.18), transparent 32%),
                    linear-gradient(135deg, rgba(38,27,10,.92), rgba(19,26,40,.96));
        margin: 12px 0 18px;
    }
    .recommendation h2 { margin: 4px 0; font-size: 2rem; letter-spacing: -.03em; }
    .rec-label { color: var(--amber); font-size: .78rem; font-weight: 850; text-transform: uppercase; letter-spacing: .08em; }
    .citation { color: #b8d9ff; font-size: .82rem; font-family: ui-monospace, Menlo, Consolas, monospace; }
    .disclaimer { border: 1px solid rgba(245,184,75,.32); background: rgba(245,184,75,.09); color: #ffe2aa; border-radius: 14px; padding: 12px 14px; font-size: .86rem; }
    @media (max-width: 900px) { .step-row { grid-template-columns: 1fr; } .hero h1 { font-size: 2.15rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    with st.sidebar:
        st.markdown("### Workflow Guide")
        st.caption("One finance process, from intake to memo.")
        page = st.radio(
            "Sections",
            ["Overview", "CapEx Workflow", "Evaluation Evidence", "Implementation Notes"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Recommended live path**")
        st.markdown("1. Explain the finance bottleneck.")
        st.markdown("2. Load the example CapEx request.")
        st.markdown("3. Run the finance tool workflow.")
        st.markdown("4. Show evidence, recommendation, memo.")
        st.markdown("5. Close with evaluation and safeguards.")
        st.divider()
        st.caption("All records are synthetic demo data.")

    if page == "Overview":
        render_overview()
    elif page == "CapEx Workflow":
        render_workflow()
    elif page == "Evaluation Evidence":
        render_evaluation()
    else:
        render_implementation_notes()


def hero(title: str, subtitle: str, pill: str = "Finance Workflow Prototype") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">{pill}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    hero(
        APP_TITLE,
        f"{APP_SUBTITLE}. The demo turns a capital request into an auditable recommendation, evidence pack, and executive memo.",
    )
    st.markdown(
        """
        <div class="disclaimer"><strong>Demo disclaimer:</strong> This prototype uses synthetic enterprise finance data.
        It does not represent real company financials, supplier records, approval records, or internal documents.</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Business Problem")
    st.markdown(
        """
        Finance teams evaluating major CapEx requests have to pull evidence from ERP budget pools,
        planning models, approval policies, historical requests, variance reports, and supporting documents.
        The work is usually spread across spreadsheets, email, workflow tools, and shared drives.

        **CapExFlow AI compresses that work into an auditable finance workflow**: intake the request,
        gather evidence, evaluate financial merit, determine approval routing, and produce a memo a CFO
        or investment committee can actually use.
        """
    )

    st.markdown(
        """
        <div class="step-row">
            <div class="step-card"><div class="step-num">Step 01</div><div class="step-title">CapEx request</div><div class="step-copy">Structured intake captures amount, pool, urgency, timeline, and rationale.</div></div>
            <div class="step-card"><div class="step-num">Step 02</div><div class="step-title">Finance tool workflow</div><div class="step-copy">The workflow calls budget, valuation, document, precedent, variance, and routing tools.</div></div>
            <div class="step-card"><div class="step-num">Step 03</div><div class="step-title">Evidence pack</div><div class="step-copy">Outputs become auditable facts with source IDs and citations.</div></div>
            <div class="step-card"><div class="step-num">Step 04</div><div class="step-title">Recommendation</div><div class="step-copy">Approve, approve with conditions, defer, or reject.</div></div>
            <div class="step-card"><div class="step-num">Step 05</div><div class="step-title">Executive memo</div><div class="step-copy">A short investment memo summarizes the ask, evidence, controls, and next step.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Example Case")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown(
            """
            The workflow is industry-agnostic, but the example case uses a capital-intensive program:
            **a critical-path launch acceleration request**.

            This is a strong finance scenario because the right answer is not simply “cut cost.”
            The finance question is whether the request creates enough schedule acceleration, risk reduction,
            and progress per dollar to justify conditional approval.
            """
        )
    with c2:
        render_key_value_card(
            "Scenario Snapshot",
            [
                ("Title", DEMO_DAY_SCENARIO["title"]),
                ("Amount", money(DEMO_DAY_SCENARIO["amount_m"])),
                ("Budget Pool", clean_dash(DEMO_DAY_SCENARIO["budget_pool"])),
                ("Priority", DEMO_DAY_SCENARIO["priority_tag"]),
                ("Urgency", DEMO_DAY_SCENARIO["urgency"]),
                ("Completion", f"{DEMO_DAY_SCENARIO['completion_months']} months"),
            ],
        )


def render_workflow() -> None:
    hero(
        "Capital Request Review",
        "Submit a CapEx request, run the finance evidence workflow, and generate an investment-committee memo.",
        "Main Workflow",
    )
    defaults = st.session_state.get("request_defaults", DEMO_DAY_SCENARIO.copy())

    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        st.markdown("### 1. Submit CapEx Request")
        if st.button("Load Example Request", type="secondary", use_container_width=True):
            st.session_state["request_defaults"] = DEMO_DAY_SCENARIO.copy()
            st.rerun()
        request = render_request_form(defaults)
    with right:
        st.markdown("### What Finance Checks")
        render_tool_catalog()
        st.markdown("### Budget Pools")
        render_budget_snapshot()

    if request:
        st.session_state["latest_request"] = request
        with st.spinner("Running finance evidence workflow..."):
            tool_runs = run_tool_workflow(request)
            decision = make_decision(request, tool_runs)
            memo = build_executive_memo(request, tool_runs, decision)
        st.session_state["latest_tool_runs"] = tool_runs
        st.session_state["latest_decision"] = decision
        st.session_state["latest_memo"] = memo

    if "latest_decision" in st.session_state:
        st.divider()
        render_results(
            st.session_state["latest_request"],
            st.session_state["latest_tool_runs"],
            st.session_state["latest_decision"],
            st.session_state["latest_memo"],
        )


def render_request_form(defaults: dict[str, Any]) -> dict[str, Any] | None:
    pools = [row["name"] for row in load_data()["budget_pools"] if row["name"].startswith("TeraWave")]
    with st.form("capex_request_form", border=True):
        title = st.text_input("Request title", value=defaults.get("title", ""))
        description = st.text_area("Description", value=defaults.get("description", ""), height=92)
        c1, c2 = st.columns(2)
        budget_pool = c1.selectbox(
            "Budget pool",
            pools,
            index=index_or_zero(pools, defaults.get("budget_pool")),
            format_func=clean_dash,
        )
        amount_m = c2.number_input("Amount ($M)", min_value=0.1, max_value=1000.0, value=float(defaults.get("amount_m", 45.0)), step=0.5)
        c3, c4, c5 = st.columns(3)
        priority_tag = c3.selectbox("Priority tag", PRIORITY_TAGS, index=index_or_zero(PRIORITY_TAGS, defaults.get("priority_tag")))
        urgency = c4.selectbox("Urgency", URGENCY_LEVELS, index=index_or_zero(URGENCY_LEVELS, defaults.get("urgency")))
        completion_months = c5.number_input("Completion timeline (months)", min_value=1, max_value=60, value=int(defaults.get("completion_months", 9)), step=1)
        justification = st.text_area("Justification", value=defaults.get("justification", ""), height=118)
        submitted = st.form_submit_button("Run Agentic Evaluation", type="primary", use_container_width=True)
    if not submitted:
        return None
    return {
        "title": title,
        "description": description,
        "budget_pool": budget_pool,
        "amount_m": float(amount_m),
        "priority_tag": priority_tag,
        "urgency": urgency,
        "completion_months": int(completion_months),
        "justification": justification,
    }


def render_tool_catalog() -> None:
    tools = [
        ("check_budget", "ERP-style budget pool availability"),
        ("run_financial_impact", "NPV, ROI, payback, progress, and risk scores"),
        ("search_documents", "Policy, planning, risk, and business-case evidence"),
        ("get_comparable_requests", "Historical capital request precedent"),
        ("get_variance_status", "YTD budget vs. actual context"),
        ("check_approval_routing", "Approval tier and SLA policy"),
    ]
    for name, desc in tools:
        st.markdown(f'<span class="tool-chip">{name}</span> {desc}', unsafe_allow_html=True)


def render_budget_snapshot() -> None:
    for row in load_data()["budget_pools"]:
        if not row["name"].startswith("TeraWave"):
            continue
        available = row["budget_m"] - row["spent_m"] - row["committed_m"]
        util = (row["spent_m"] + row["committed_m"]) / row["budget_m"]
        st.markdown(f"**{row['name'].replace('TeraWave — ', '')}** - {money(available)} available ({util:.0%} utilized)")


def run_tool_workflow(request: dict[str, Any]) -> list[dict[str, Any]]:
    query = (
        f"{request['title']} {request['priority_tag']} launch manifest capital allocation policy "
        "speed over savings acceleration schedule risk initial operational capability"
    )
    plan = [
        ("check_budget", "Budget Availability", "Confirm whether the request fits inside the selected budget pool.", {"budget_pool": request["budget_pool"]}),
        ("run_financial_impact", "Financial Impact", "Estimate NPV, ROI, payback, progress contribution, and risk retirement.", {"amount_m": request["amount_m"], "priority_tag": request["priority_tag"], "completion_months": request["completion_months"], "title": request["title"]}),
        ("search_documents", "Document Evidence", "Retrieve policy, planning, and business-case evidence relevant to the request.", {"query": query}),
        ("get_comparable_requests", "Precedent Review", "Find similar historical requests and outcomes.", {"budget_pool": request["budget_pool"], "priority_tag": request["priority_tag"], "amount_m": request["amount_m"]}),
        ("get_variance_status", "Variance Context", "Check whether the workstream has current budget-vs-actual pressure.", {"workstream": budget_pool_to_workstream(request["budget_pool"])}),
        ("check_approval_routing", "Approval Routing", "Determine approval tier and SLA from policy.", {"amount_m": request["amount_m"], "urgency": request["urgency"]}),
    ]
    runs = []
    for name, label, rationale, inputs in plan:
        runs.append({"name": name, "label": label, "rationale": rationale, "input": inputs, "output": execute_demo_tool(name, inputs)})
    return runs


def execute_demo_tool(name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if name == "check_budget":
        return tool_check_budget(inputs["budget_pool"])
    if name == "run_financial_impact":
        return tool_financial_impact(inputs)
    if name == "search_documents":
        return tool_search_documents(inputs["query"])
    if name == "get_comparable_requests":
        return tool_comparables(inputs)
    if name == "get_variance_status":
        return VARIANCE_STATUS.get(inputs["workstream"], VARIANCE_STATUS["Launch Services"])
    if name == "check_approval_routing":
        return tool_approval_routing(inputs["amount_m"], inputs.get("urgency", "Standard"))
    return {"error": f"Unknown tool: {name}"}


def tool_check_budget(pool_name: str) -> dict[str, Any]:
    for row in load_data()["budget_pools"]:
        if row["name"] == pool_name:
            available = row["budget_m"] - row["spent_m"] - row["committed_m"]
            util = (row["spent_m"] + row["committed_m"]) / row["budget_m"] * 100
            return {
                "budget_pool": row["name"],
                "total_budget_m": row["budget_m"],
                "spent_m": row["spent_m"],
                "committed_m": row["committed_m"],
                "available_m": available,
                "utilization_pct": round(util, 1),
                "status": "healthy" if util < 70 else "caution" if util < 90 else "critical",
                "source_record_id": row["id"],
                "source_system": row.get("system", "Synthetic ERP"),
            }
    return {"error": f"Budget pool not found: {pool_name}"}


def tool_financial_impact(inputs: dict[str, Any]) -> dict[str, Any]:
    amount = float(inputs.get("amount_m", 0))
    priority = inputs.get("priority_tag", "Maintenance")
    months = int(inputs.get("completion_months", 12))
    multipliers = {"Critical Path": 3.5, "Risk Retirement": 4.0, "Cost Reduction": 2.5, "Capacity Expansion": 2.0, "R&D / Innovation": 3.0, "Maintenance": 1.2}
    progress_map = {"Critical Path": 0.9, "Capacity Expansion": 0.7, "R&D / Innovation": 0.6, "Cost Reduction": 0.4, "Risk Retirement": 0.5, "Maintenance": 0.2}
    risk_map = {"Risk Retirement": 0.9, "Critical Path": 0.6, "R&D / Innovation": 0.7, "Cost Reduction": 0.3, "Capacity Expansion": 0.4, "Maintenance": 0.1}
    adjustment = stable_adjustment(str(inputs.get("title", priority)))
    npv = round(amount * multipliers.get(priority, 1.5) * adjustment, 1)
    roi = round((npv / amount - 1) * 100, 1) if amount else 0
    payback = max(3, min(60, int(months * (amount / max(npv, 0.01)) * 12)))
    return {
        "amount_m": amount,
        "npv_impact_m": npv,
        "roi_pct": roi,
        "payback_months": payback,
        "progress_score": progress_map.get(priority, 0.5),
        "risk_retirement_score": risk_map.get(priority, 0.3),
        "assessment": "strong" if roi > 100 else "moderate" if roi > 30 else "marginal",
        "source_record_id": "SYNTH-FINANCIAL-IMPACT-MODEL",
    }


def tool_search_documents(query: str, top_k: int = 3) -> dict[str, Any]:
    q_tokens = tokenize(query)
    scored = []
    for doc in load_data()["documents"]:
        text = f"{doc['title']} {doc['content']}"
        tokens = tokenize(text)
        if not tokens:
            continue
        overlap = len(q_tokens.intersection(tokens))
        score = overlap / max(8, len(q_tokens))
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    docs = []
    for score, doc in scored[:top_k]:
        docs.append(
            {
                "title": doc["title"],
                "doc_id": doc["id"],
                "type": doc["doc_type"],
                "date": doc["date"],
                "relevance": round(min(0.99, score), 2),
                "excerpt": doc["content"][:1100],
                "source_record_id": doc["id"],
            }
        )
    return {"query": query, "results_found": len(docs), "documents": docs}


def tool_comparables(inputs: dict[str, Any]) -> dict[str, Any]:
    pool = inputs.get("budget_pool", "")
    priority = inputs.get("priority_tag", "")
    amount = float(inputs.get("amount_m", 0) or 0)
    matches = []
    for req in load_data()["historical_requests"]:
        score = 0
        if req["budget_pool"] == pool:
            score += 3
        if req["priority_tag"] == priority:
            score += 2
        if amount and abs(req["amount_m"] - amount) / amount < 0.5:
            score += 2
        if score >= 2:
            row = dict(req)
            row["relevance_score"] = score
            row["source_record_id"] = req["id"]
            matches.append(row)
    matches.sort(key=lambda item: item["relevance_score"], reverse=True)
    return {"comparables_found": len(matches[:5]), "requests": matches[:5]}


def tool_approval_routing(amount: float, urgency: str) -> dict[str, Any]:
    for tier in load_data()["approval_tiers"]:
        if amount <= tier["max_amount_m"]:
            sla = tier["sla_days"]
            if urgency == "Emergency":
                sla = max(1, sla // 2)
            elif urgency == "Expedited":
                sla = max(2, int(sla * 0.7))
            return {"tier": tier["tier"], "approver": tier["label"], "sla_days": sla, "urgency": urgency, "amount_m": amount, "source_record_id": tier["source_record_id"]}
    return {"error": "Amount exceeds approval tiers"}


def make_decision(request: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = by_tool(runs)
    budget = evidence["check_budget"]
    financial = evidence["run_financial_impact"]
    routing = evidence["check_approval_routing"]
    variance = evidence["get_variance_status"]
    amount = request["amount_m"]
    available = float(budget.get("available_m", 0) or 0)
    roi = float(financial.get("roi_pct", 0) or 0)
    progress = float(financial.get("progress_score", 0) or 0)
    risk = float(financial.get("risk_retirement_score", 0) or 0)
    tier = int(routing.get("tier", 0) or 0)
    variance_pct = abs(float(variance.get("ytd_variance_pct", 0) or 0))

    score = 0
    score += 30 if available >= amount else -30
    score += 25 if roi >= 100 else 16 if roi >= 50 else 6 if roi > 0 else -10
    score += 20 if request["priority_tag"] in {"Critical Path", "Risk Retirement"} else 8
    score += round((progress + risk) * 10)
    score += 8 if variance_pct <= 5 else -5
    score -= 4 if tier >= 3 else 0
    score = int(max(0, min(100, score)))

    conditions = []
    risk_flags = []
    if available < amount:
        risk_flags.append("Requested amount exceeds available budget headroom.")
    if tier >= 3:
        conditions.append("Route to CFO approval before supplier or internal commitments are made.")
    if request["urgency"] in {"Expedited", "Emergency"}:
        conditions.append("Use milestone release: staffing kickoff, pad-readiness completion, and readiness-review pull-forward evidence.")
    if amount >= 25:
        conditions.append("Require monthly finance check-ins until the 9-month completion window is retired.")
    if variance_pct > 5:
        conditions.append("Confirm variance recovery plan or reallocation source before final release.")
    if roi < 30:
        risk_flags.append("Financial impact is below the normal threshold for strategic acceleration funding.")

    if available < amount * 0.9 or roi < 0:
        recommendation = "Reject"
        rationale = "The request does not clear minimum budget or financial merit thresholds."
    elif score < 45:
        recommendation = "Defer"
        rationale = "The request needs stronger financial evidence or budget mitigation before approval."
    elif conditions:
        recommendation = "Approve with Conditions"
        rationale = "The request has strong strategic and financial merit, but the dollar size, expedited timing, and CFO routing require explicit controls."
    else:
        recommendation = "Approve"
        rationale = "The request fits budget, clears financial thresholds, and aligns with program priorities."
    return {"recommendation": recommendation, "rationale": rationale, "score": score, "conditions": conditions, "risk_flags": risk_flags}


def render_results(request: dict[str, Any], runs: list[dict[str, Any]], decision: dict[str, Any], memo: str) -> None:
    evidence = by_tool(runs)
    budget = evidence["check_budget"]
    financial = evidence["run_financial_impact"]
    routing = evidence["check_approval_routing"]
    docs = evidence["search_documents"].get("documents", [])
    comps = evidence["get_comparable_requests"].get("requests", [])
    variance = evidence["get_variance_status"]

    st.markdown("### 2. Recommendation")
    st.markdown(
        f"""
        <div class="recommendation"><div class="rec-label">Recommendation</div><h2>{decision['recommendation']}</h2><div class="muted">{decision['rationale']}</div></div>
        """,
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Decision Score", f"{decision['score']}/100")
    k2.metric("Budget Headroom", money(budget.get("available_m", 0)))
    k3.metric("NPV Impact", money(financial.get("npv_impact_m", 0)))
    k4.metric("ROI", f"{financial.get('roi_pct', 0):,.0f}%")
    k5.metric("Approval", f"T{routing.get('tier')}: {routing.get('approver')}")

    if decision["conditions"]:
        st.markdown("**Approval conditions**")
        for condition in decision["conditions"]:
            st.markdown(f"- {condition}")

    st.markdown("### 3. Finance Evidence Pack")
    cols = st.columns(3)
    with cols[0]:
        render_evidence_card("Budget", [f"Pool: {clean_dash(budget.get('budget_pool', 'N/A'))}", f"Total: {money(budget.get('total_budget_m', 0))}", f"Available: {money(budget.get('available_m', 0))}", f"Utilization: {budget.get('utilization_pct', 'N/A')}%"], budget.get("source_record_id", "budget source"))
    with cols[1]:
        render_evidence_card("Financial", [f"NPV impact: {money(financial.get('npv_impact_m', 0))}", f"Payback: {financial.get('payback_months', 'N/A')} months", f"Progress score: {financial.get('progress_score', 'N/A')}", f"Risk-retirement score: {financial.get('risk_retirement_score', 'N/A')}"], financial.get("source_record_id", "financial model"))
    with cols[2]:
        render_evidence_card("Approval Policy", [f"Tier: {routing.get('tier', 'N/A')}", f"Approver: {routing.get('approver', 'N/A')}", f"SLA: {routing.get('sla_days', 'N/A')} business days", f"Urgency: {routing.get('urgency', request['urgency'])}"], routing.get("source_record_id", "approval policy"))

    d1, d2 = st.columns([1, 1], gap="large")
    with d1:
        st.markdown("#### Source Documents")
        for doc in docs[:3]:
            with st.expander(f"{doc.get('title')} [{doc.get('doc_id')}]"):
                st.caption(f"{doc.get('type', 'document').title()} | relevance {doc.get('relevance', 0):.0%}")
                st.write(doc.get("excerpt", "")[:900] + ("..." if len(doc.get("excerpt", "")) > 900 else ""))
        st.markdown("#### Historical Capital Requests")
        for item in comps[:4]:
            st.markdown(f"- **{item.get('id')}**: {item.get('title')} - {money(item.get('amount_m', 0))}, {item.get('status')}; {item.get('outcome')}")
    with d2:
        st.markdown("#### Scorecard")
        render_scorecard(financial, budget, variance, decision)
        st.markdown("#### Variance Context")
        render_key_value_card("Current Workstream", [("Workstream", variance.get("workstream", "N/A")), ("YTD Budget", money(variance.get("ytd_budget_m", 0))), ("YTD Actual", money(variance.get("ytd_actual_m", 0))), ("YTD Variance", f"{money(variance.get('ytd_variance_m', 0))} ({variance.get('ytd_variance_pct', 0):+.1f}%)"), ("Status", variance.get("status", "N/A"))])

    with st.expander("Full Tool Trace", expanded=False):
        for run in runs:
            st.markdown(f"**{run['label']}**")
            st.caption(run["rationale"])
            st.markdown(f'<span class="tool-chip">{run["name"]}</span>', unsafe_allow_html=True)
            st.json({"input": run["input"], "output": run["output"]})
            st.divider()

    st.markdown("### 4. Investment Committee Memo")
    st.markdown(memo)
    st.download_button("Download memo (.md)", data=memo, file_name="terawave_capval_investment_memo.md", mime="text/markdown", use_container_width=True)


def render_scorecard(financial: dict[str, Any], budget: dict[str, Any], variance: dict[str, Any], decision: dict[str, Any]) -> None:
    budget_fit = 100 if budget.get("status") in {"healthy", "caution"} else 35
    roi = float(financial.get("roi_pct", 0) or 0)
    financial_merit = min(100, max(0, roi / 2))
    progress = float(financial.get("progress_score", 0) or 0) * 100
    risk = float(financial.get("risk_retirement_score", 0) or 0) * 100
    variance_health = max(0, 100 - abs(float(variance.get("ytd_variance_pct", 0) or 0)) * 6)
    rows = [("Budget Fit", budget_fit), ("Financial Merit", financial_merit), ("Progress", progress), ("Risk Retirement", risk), ("Variance Health", variance_health), ("Overall", decision["score"])]
    for label, value in rows:
        st.caption(f"{label}: {int(value)}/100")
        st.progress(int(value))


def build_executive_memo(request: dict[str, Any], runs: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    evidence = by_tool(runs)
    budget = evidence["check_budget"]
    financial = evidence["run_financial_impact"]
    routing = evidence["check_approval_routing"]
    docs = evidence["search_documents"].get("documents", [])
    comps = evidence["get_comparable_requests"].get("requests", [])
    variance = evidence["get_variance_status"]
    doc_cites = ", ".join(doc.get("doc_id", "DOC") for doc in docs[:3]) or "no document citations"
    comp_cites = ", ".join(item.get("id", "CR") for item in comps[:3]) or "no comparable records"
    conditions = decision["conditions"] or ["No additional conditions required."]
    risk_flags = decision["risk_flags"] or ["No blocking risk flags identified."]
    return "\n".join([
        f"# Executive Investment Memo: {request['title']}",
        "",
        f"**Date:** {date.today().isoformat()}",
        "**Prepared by:** CapExFlow AI",
        "**Decision status:** Decision support only; synthetic demo data.",
        "",
        "## Recommendation",
        "",
        f"**{decision['recommendation']}** - {decision['rationale']}",
        "",
        "## Request Summary",
        "",
        f"- Amount: {money(request['amount_m'])}",
        f"- Budget pool: {clean_dash(request['budget_pool'])}",
        f"- Priority / urgency: {request['priority_tag']} / {request['urgency']}",
        f"- Completion timeline: {request['completion_months']} months",
        f"- Justification: {request['justification']}",
        "",
        "## Evidence Cited",
        "",
        f"- Budget: {clean_dash(budget.get('budget_pool'))} has {money(budget.get('available_m', 0))} available against a {money(budget.get('total_budget_m', 0))} pool [{budget.get('source_record_id')}].",
        f"- Financial impact: estimated NPV impact is {money(financial.get('npv_impact_m', 0))}, ROI is {financial.get('roi_pct', 0):,.1f}%, and payback is {financial.get('payback_months')} months [{financial.get('source_record_id')}].",
        f"- Approval policy: the request routes to Tier {routing.get('tier')} ({routing.get('approver')}) with a {routing.get('sla_days')}-business-day SLA [{routing.get('source_record_id')}].",
        f"- Document search: retrieved {doc_cites}, including policy, planning, and capital allocation evidence for schedule acceleration.",
        f"- Historical precedent: comparable requests include {comp_cites}.",
        f"- Variance context: {variance.get('workstream')} is {variance.get('status')} with {variance.get('ytd_variance_pct', 0):+.1f}% YTD variance.",
        "",
        "## Conditions And Controls",
        "",
        *[f"- {condition}" for condition in conditions],
        "",
        "## Risk Flags",
        "",
        *[f"- {flag}" for flag in risk_flags],
        "",
        "## Next Step",
        "",
        "Send the memo, evidence pack, and tool trace to the required approver. Finance should keep the release milestone-based because the value case depends on measurable operational acceleration, not only budget availability.",
    ])


def render_evaluation() -> None:
    hero("Evaluation Evidence", "Concrete tests mapped to the final-project rubric: correctness, repeatability, grounding, and clarity.", "Controls")
    truth_rows = run_ground_truth_evaluation()
    repeatability = run_repeatability_check()
    pass_count = sum(1 for row in truth_rows if row["Pass"] == "Yes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ground Truth Pass Rate", f"{pass_count}/{len(truth_rows)}")
    c2.metric("Repeatability", f"{repeatability['recommendation_consistency_pct']:.0f}%")
    c3.metric("NPV Range", money(repeatability["npv_range_m"]))
    c4.metric("ROI Range", f"{repeatability['roi_range_pct']:.2f} pts")
    st.markdown("### Ground-Truth Decision Tests")
    for row in truth_rows:
        st.markdown(f"- **{row['Scenario']}**: expected `{row['Expected']}`, observed `{row['Observed']}` - **{row['Pass']}**. Evidence: {row['Evidence']}.")
    st.markdown("### Repeatability Check")
    st.json(repeatability)
    st.markdown("### Rubric Alignment")
    rubric = [
        ("Problem Choice & Business Value", "Compresses multi-system finance evidence assembly into one capital allocation workflow.", "30%"),
        ("Technical Depth", "Structured tool calls, deterministic financial scoring, document retrieval, variance checks, and memo generation.", "35%"),
        ("Evaluation & Evidence", "Ground-truth tests, repeatability check, source citations, full tool trace, and numerical consistency.", "15%"),
        ("Clarity & Communication", "Guided finance narrative, clear recommendation labels, evidence cards, and an investment memo.", "20%"),
    ]
    for area, evidence, weight in rubric:
        st.markdown(f"- **{area} ({weight})**: {evidence}")


def run_ground_truth_evaluation() -> list[dict[str, Any]]:
    scenarios = [
        ("Clearly approve", "Approve", {"title": "OISL Automated Alignment Station", "description": "Improve optical terminal yield.", "budget_pool": "TeraWave — OISL & Comms", "amount_m": 8.2, "priority_tag": "Risk Retirement", "urgency": "Standard", "completion_months": 6, "justification": "Retires a known OISL manufacturing bottleneck and protects deployment schedule confidence."}, "Strong risk-retirement fit, in-budget request, and high expected financial impact."),
        ("Clearly reject", "Reject", {"title": "Noncritical Admin Hardware Refresh", "description": "Replace office hardware.", "budget_pool": "TeraWave — Software & Network Ops", "amount_m": 650.0, "priority_tag": "Maintenance", "urgency": "Standard", "completion_months": 24, "justification": "Refreshes aging support assets but does not directly accelerate deployment or retire critical program risk."}, "Low strategic fit and request exceeds available budget."),
        ("Ambiguous / conditional", "Approve with Conditions", {"title": "Noncritical Spares Replenishment", "description": "Increase spare hardware inventory.", "budget_pool": "TeraWave — Satellite Manufacturing", "amount_m": 25.0, "priority_tag": "Maintenance", "urgency": "Standard", "completion_months": 24, "justification": "Improves resilience but has limited direct timeline acceleration, so finance should evaluate it with milestone controls."}, "Budget is available, but strategic fit is weaker and controls are appropriate."),
    ]
    rows = []
    for name, expected, request, why in scenarios:
        runs = run_tool_workflow(request)
        decision = make_decision(request, runs)
        financial = by_tool(runs)["run_financial_impact"]
        budget = by_tool(runs)["check_budget"]
        rows.append({"Scenario": name, "Expected": expected, "Observed": decision["recommendation"], "Pass": "Yes" if decision["recommendation"] == expected else "No", "NPV Impact ($M)": financial["npv_impact_m"], "ROI (%)": financial["roi_pct"], "Budget Status": budget["status"], "Evidence": why})
    return rows


def run_repeatability_check(n_runs: int = 5) -> dict[str, Any]:
    recommendations = []
    npvs = []
    rois = []
    for _ in range(n_runs):
        runs = run_tool_workflow(DEMO_DAY_SCENARIO)
        decision = make_decision(DEMO_DAY_SCENARIO, runs)
        financial = by_tool(runs)["run_financial_impact"]
        recommendations.append(decision["recommendation"])
        npvs.append(financial["npv_impact_m"])
        rois.append(financial["roi_pct"])
    return {"request": DEMO_DAY_SCENARIO["title"], "runs": n_runs, "recommendations": recommendations, "unique_recommendations": sorted(set(recommendations)), "recommendation_consistency_pct": round(recommendations.count(recommendations[0]) / n_runs * 100, 1), "npv_range_m": round(max(npvs) - min(npvs), 2), "roi_range_pct": round(max(rois) - min(rois), 2), "result": "PASS" if len(set(recommendations)) == 1 else "REVIEW"}


def render_implementation_notes() -> None:
    hero("Implementation Notes", "How the prototype demonstrates finance workflow automation while staying demoable and auditable.", "Architecture")
    st.markdown(
        """
        ### Technical Architecture

        The app is intentionally scoped to one finance workflow. The agentic layer is represented by structured tools for budget
        checks, financial impact, document retrieval, comparable precedents, variance status, and approval routing. Tool outputs
        are then normalized into a recommendation, evidence pack, scorecard, and memo.

        ### Production Safeguards

        This should remain decision support, not autonomous approval. A production version would need source-system permissions,
        audit logging, prompt-injection defenses for retrieved documents, model/version governance, human approval gates, and
        monitoring for numerical consistency.
        """
    )
    st.code("python -m streamlit run app.py", language="bash")


def render_evidence_card(title: str, facts: list[str], citation: str) -> None:
    facts_html = "".join(f"<div class='step-copy'>{fact}</div>" for fact in facts)
    st.markdown(f"<div class='evidence-card'><div class='small-heading'>{title}</div>{facts_html}<div style='height:10px'></div><div class='citation'>Citation: {citation}</div></div>", unsafe_allow_html=True)


def render_key_value_card(title: str, rows: list[tuple[str, Any]]) -> None:
    body = "".join(f"<div class='step-copy'><strong>{key}:</strong> {value}</div>" for key, value in rows)
    st.markdown(f"<div class='card'><div class='small-heading'>{title}</div>{body}</div>", unsafe_allow_html=True)


def by_tool(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {run["name"]: run["output"] for run in runs}


def budget_pool_to_workstream(pool: str) -> str:
    if "Launch Services" in pool:
        return "Launch Services"
    if "Ground Segment" in pool:
        return "Ground Segment & Gateways"
    if "OISL" in pool or "Comms" in pool:
        return "Optical Inter-Satellite Links"
    if "Software" in pool:
        return "Software & Network Ops"
    if "Manufacturing" in pool:
        return "Satellite Manufacturing (LEO)"
    return "Launch Services"


def stable_adjustment(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:4], 16) % 31
    return 0.85 + bucket / 100


def tokenize(text: str) -> set[str]:
    stops = {"the", "and", "for", "with", "from", "that", "this", "into", "must", "shall", "are", "was", "were", "has", "have", "will", "may", "per", "not", "only", "than"}
    return {tok for tok in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(tok) > 2 and tok not in stops}


def clean_dash(value: Any) -> str:
    return str(value).replace("TeraWave — ", "").replace("TeraWave - ", "").replace("—", "-")


def money(value: Any) -> str:
    try:
        return f"${float(value):,.1f}M" if abs(float(value)) < 100 else f"${float(value):,.0f}M"
    except (TypeError, ValueError):
        return "$0M"


def index_or_zero(options: list[str], value: Any) -> int:
    return options.index(value) if value in options else 0


main()
