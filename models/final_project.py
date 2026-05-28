"""
Final project helpers for the concise TeraWave CapVal demo.

These functions turn workflow evidence into a short investment memo and expose
the evaluation assets needed for the final report/demo.

ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from models.capex_workflow import CapExRequest, run_capex_workflow
from models.synthetic_data import get_source_metadata


SUCCESS_METRICS = [
    {
        "metric": "Tool-selection accuracy",
        "target": "Agent calls the relevant enterprise-style tools for each request.",
        "rubric": "Technical Depth / Evaluation",
    },
    {
        "metric": "Recommendation correctness",
        "target": "GO, CONDITIONAL, DEFER, or NO-GO matches the scenario expectation.",
        "rubric": "Evaluation & Evidence",
    },
    {
        "metric": "Evidence grounding",
        "target": "Recommendation and memo cite retrieved policy, contract, variance, or precedent evidence.",
        "rubric": "Evaluation & Evidence",
    },
    {
        "metric": "Numerical consistency",
        "target": "Memo figures match tool outputs for NPV, ROI, payback, budget, and approval tier.",
        "rubric": "Technical Depth / Clarity",
    },
    {
        "metric": "Business usability",
        "target": "A finance/business reader can understand the action without reading raw tool logs.",
        "rubric": "Clarity & Communication",
    },
    {
        "metric": "Cycle-time compression",
        "target": "Prototype produces a first-pass memo in minutes instead of a multi-day manual assembly cycle.",
        "rubric": "Problem Choice & Business Value",
    },
]


EVALUATION_SCENARIOS = [
    {
        "scenario": "Routine satellite component reorder",
        "input": "$8M maintenance / manufacturing request",
        "expected": "Short tool chain, GO if budget is healthy",
        "evidence": "Budget check, comparable requests, financial impact",
    },
    {
        "scenario": "Critical-path launch acceleration",
        "input": "$45M launch services request, expedited",
        "expected": "CONDITIONAL GO with CFO routing and milestone controls",
        "evidence": "Budget, NPV/ROI, approval routing, comparables, launch document search",
    },
    {
        "scenario": "Request in an overspending workstream",
        "input": "$15M ground segment request during YTD variance pressure",
        "expected": "DEFER or CONDITIONAL with reallocation requirement",
        "evidence": "Variance status, budget status, policy citation",
    },
    {
        "scenario": "Risk-retirement investment",
        "input": "$12M optical inter-satellite link risk mitigation",
        "expected": "GO if risk-retired-per-dollar is high",
        "evidence": "Document search, risk score, financial impact",
    },
    {
        "scenario": "Incomplete or weak request",
        "input": "Low-justification request with limited strategic link",
        "expected": "Clarifying question, DEFER, or conditions",
        "evidence": "Validation flags and missing rationale",
    },
    {
        "scenario": "Over-threshold board approval",
        "input": "$125M program expansion request",
        "expected": "Board routing and explicit deployment trade-offs",
        "evidence": "Approval policy, budget availability, memo decision points",
    },
]


REPORT_OUTLINE = [
    ("Business problem and value", "Analysts spend days assembling CapEx decisions from finance, procurement, workflow, and document systems."),
    ("System overview", "Streamlit prototype with an agentic CapEx workflow, transparent tool trace, valuation dashboard, RAG, and memo output."),
    ("Data and implementation", "Synthetic budget pools, historical requests, policy/contracts corpus, DCF model, Monte Carlo engine, and Claude tool-use loop."),
    ("Technical depth", "Autonomous tool selection, structured tool schemas, correlated Monte Carlo, TF-IDF retrieval, approval routing, variance detection."),
    ("Evaluation evidence", "Six scenario tests scored on tool choice, recommendation quality, grounding, numerical consistency, and usability."),
    ("Deployment trade-offs", "Human-in-the-loop governance, synthetic-data limits, privacy, prompt injection, model reliability, and enterprise integration risk."),
]


DEMO_DAY_SCENARIO = {
    "title": "Critical-Path Launch Acceleration Package",
    "description": "Add launch integration shifts and pad-readiness work to pull the first operational TeraWave deployment window forward.",
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

DEMO_NARRATIVE_PATHS = {
    "Recommended: Critical-path acceleration": {
        "why": "Best Demo Day path. It shows financial return, approval routing, precedent, schedule risk, and memo generation in one story.",
        "talk_track": [
            "Open on the business problem: capital decisions require evidence from many systems.",
            "Submit the critical-path request and watch the agent call tools.",
            "Show the recommendation and explain why conditions matter.",
            "Open the memo and point to the tool evidence behind the recommendation.",
        ],
        "request": DEMO_DAY_SCENARIO,
    },
    "Clear approve: risk-retirement investment": {
        "why": "Shows the system approving a high-value technical risk retirement case.",
        "talk_track": [
            "Frame this as an investment that retires a known technical risk.",
            "Show budget fit, strong ROI, and document evidence.",
            "Use it as the clean positive-control scenario in the report.",
        ],
        "request": {
            "title": "OISL Automated Alignment Station",
            "description": "Fund automated optical terminal alignment equipment to improve manufacturing yield and retire OISL schedule risk.",
            "budget_pool": "TeraWave — OISL & Comms",
            "amount_m": 8.2,
            "priority_tag": "Risk Retirement",
            "urgency": "Standard",
            "completion_months": 6,
            "justification": "The investment improves terminal yield, retires a known OISL manufacturing bottleneck, and protects deployment schedule confidence.",
        },
    },
    "Clear reject: low-merit over-budget request": {
        "why": "Shows the system can say no when a request has poor strategic fit and exceeds available budget.",
        "talk_track": [
            "Frame this as the negative-control scenario.",
            "Show that budget pressure and weak strategic alignment drive the recommendation.",
            "Use it to prove the system is not a rubber stamp.",
        ],
        "request": {
            "title": "Noncritical Admin Hardware Refresh",
            "description": "Replace general office hardware and noncritical support equipment across the network operations organization.",
            "budget_pool": "TeraWave — Software & Network Ops",
            "amount_m": 650.0,
            "priority_tag": "Maintenance",
            "urgency": "Standard",
            "completion_months": 24,
            "justification": "The request refreshes aging support assets but does not directly accelerate deployment or retire a critical program risk.",
        },
    },
}

GROUND_TRUTH_SCENARIOS = [
    {
        "name": "Clearly approve",
        "expected": "Approve",
        "request": DEMO_NARRATIVE_PATHS["Clear approve: risk-retirement investment"]["request"],
        "why": "Strong risk-retirement fit, in-budget request, and high expected financial impact.",
    },
    {
        "name": "Clearly reject",
        "expected": "Reject",
        "request": DEMO_NARRATIVE_PATHS["Clear reject: low-merit over-budget request"]["request"],
        "why": "Low strategic fit and request exceeds available budget.",
    },
    {
        "name": "Ambiguous / conditional",
        "expected": "Approve with Conditions",
        "request": {
            "title": "Noncritical Spares Replenishment",
            "description": "Increase noncritical spare hardware inventory to improve operational resilience without changing the deployment critical path.",
            "budget_pool": "TeraWave — Satellite Manufacturing",
            "amount_m": 25.0,
            "priority_tag": "Maintenance",
            "urgency": "Standard",
            "completion_months": 24,
            "justification": "The request improves resilience but has limited direct timeline acceleration, so finance should evaluate it with milestone controls.",
        },
        "why": "Budget is available, but strategic fit is weaker and controls are appropriate.",
    },
]


def run_repeatability_check(n_runs: int = 5) -> dict[str, Any]:
    """Run the same request repeatedly and summarize recommendation variance."""
    request = _request_from_dict(DEMO_DAY_SCENARIO)
    results = [run_capex_workflow(request) for _ in range(n_runs)]
    recommendations = [r.recommendation for r in results]
    npv_values = [r.npv_impact_m for r in results]
    roi_values = [r.roi_pct for r in results]

    return {
        "request": DEMO_DAY_SCENARIO["title"],
        "runs": n_runs,
        "recommendations": recommendations,
        "unique_recommendations": sorted(set(recommendations)),
        "recommendation_consistency_pct": round(recommendations.count(recommendations[0]) / n_runs * 100, 1),
        "npv_range_m": round(max(npv_values) - min(npv_values), 2),
        "roi_range_pct": round(max(roi_values) - min(roi_values), 2),
        "result": "PASS" if len(set(recommendations)) == 1 else "REVIEW",
    }


def run_ground_truth_evaluation() -> list[dict[str, Any]]:
    """Run approve/reject/ambiguous cases and compare against expected labels."""
    rows = []
    for scenario in GROUND_TRUTH_SCENARIOS:
        result = run_capex_workflow(_request_from_dict(scenario["request"]))
        passed = result.recommendation == scenario["expected"]
        rows.append({
            "Scenario": scenario["name"],
            "Expected": scenario["expected"],
            "Observed": result.recommendation,
            "Pass": "Yes" if passed else "No",
            "NPV Impact ($M)": result.npv_impact_m,
            "ROI (%)": result.roi_pct,
            "Budget Status": result.budget_status,
            "Approval Tier": f"T{result.approval_tier}: {result.approval_tier_label}",
            "Evidence": scenario["why"],
        })
    return rows


def evaluation_summary() -> dict[str, Any]:
    """Return compact metrics for the report and app evidence panel."""
    truth_rows = run_ground_truth_evaluation()
    repeatability = run_repeatability_check()
    pass_count = sum(1 for row in truth_rows if row["Pass"] == "Yes")
    return {
        "ground_truth_pass_rate": f"{pass_count}/{len(truth_rows)}",
        "repeatability_result": repeatability["result"],
        "recommendation_consistency_pct": repeatability["recommendation_consistency_pct"],
        "npv_range_m": repeatability["npv_range_m"],
        "roi_range_pct": repeatability["roi_range_pct"],
    }


def _request_from_dict(data: dict[str, Any]) -> CapExRequest:
    return CapExRequest(
        id="EVAL",
        title=data["title"],
        description=data["description"],
        requestor="Evaluation Harness",
        department="TeraWave Program",
        budget_pool=data["budget_pool"],
        amount_m=float(data["amount_m"]),
        priority_tag=data["priority_tag"],
        urgency=data["urgency"],
        justification=data["justification"],
        expected_completion_months=int(data["completion_months"]),
        submission_date="2026-04-12",
    )


def extract_agent_evidence(agent_result: Any) -> dict[str, Any]:
    """Convert an AgentResult tool trace into a compact evidence pack."""
    source = get_source_metadata()
    tool_results: dict[str, list[dict[str, Any]]] = {}
    tool_calls = []

    for step in getattr(agent_result, "steps", []):
        if getattr(step, "step_type", "") == "tool_call":
            tool_calls.append(getattr(step, "tool_name", "unknown"))
        if getattr(step, "step_type", "") != "tool_result":
            continue

        name = getattr(step, "tool_name", "unknown") or "unknown"
        content = getattr(step, "content", "")
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw": content}
        tool_results.setdefault(name, []).append(parsed)

    return {
        "source": "agentic_tool_trace",
        "data_status": "synthetic_demo_data_only",
        "source_file": source["source_file"],
        "dataset_id": source["dataset_id"],
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "final_answer": getattr(agent_result, "final_answer", ""),
        "duration_ms": getattr(agent_result, "total_duration_ms", 0),
        "total_tool_calls": getattr(agent_result, "total_tool_calls", len(tool_calls)),
    }


def extract_workflow_evidence(workflow_result: Any) -> dict[str, Any]:
    """Convert the deterministic fallback workflow result into the same evidence shape."""
    source = get_source_metadata()
    req = workflow_result.request
    return {
        "source": "rule_based_workflow",
        "data_status": "synthetic_demo_data_only",
        "source_file": source["source_file"],
        "dataset_id": source["dataset_id"],
        "request": {
            "title": req.title,
            "amount_m": req.amount_m,
            "budget_pool": req.budget_pool,
            "priority_tag": req.priority_tag,
            "urgency": req.urgency,
            "completion_months": req.expected_completion_months,
            "justification": req.justification,
        },
        "financial": {
            "npv_impact_m": workflow_result.npv_impact_m,
            "roi_pct": workflow_result.roi_pct,
            "payback_months": workflow_result.payback_months,
            "progress_score": workflow_result.progress_score,
            "risk_retirement_score": workflow_result.risk_retirement_score,
        },
        "budget": {
            "status": workflow_result.budget_status,
            "available_m": workflow_result.budget_available_m,
            "remaining_pct": workflow_result.budget_remaining_pct,
        },
        "routing": {
            "tier": workflow_result.approval_tier,
            "label": workflow_result.approval_tier_label,
            "sla_days": workflow_result.sla_days,
        },
        "comparables": workflow_result.comparables,
        "recommendation": workflow_result.recommendation,
        "rationale": workflow_result.recommendation_rationale,
        "conditions": workflow_result.conditions,
        "risk_flags": workflow_result.risk_flags,
        "workflow_steps": workflow_result.workflow_steps,
    }


def build_agent_memo(request: dict[str, Any], evidence: dict[str, Any]) -> str:
    """Build a concise memo from an agentic tool trace."""
    tool_results = evidence.get("tool_results", {})
    budget = _first(tool_results.get("check_budget", []))
    financial = _first(tool_results.get("run_financial_impact", []))
    routing = _first(tool_results.get("check_approval_routing", []))
    comparables = _first(tool_results.get("get_comparable_requests", []))
    docs = _first(tool_results.get("search_documents", []))
    variance = _first(tool_results.get("get_variance_status", []))

    return _render_memo(
        request=request,
        recommendation=_recommendation_from_text(evidence.get("final_answer", "")),
        rationale=evidence.get("final_answer", "").strip(),
        financial=financial,
        budget=budget,
        routing=routing,
        comparables=comparables.get("requests", []),
        documents=docs.get("documents", []),
        variance=variance,
        conditions=[],
        risk_flags=[],
        tool_calls=evidence.get("tool_calls", []),
    )


def build_workflow_memo(evidence: dict[str, Any]) -> str:
    """Build a concise memo from deterministic workflow evidence."""
    return _render_memo(
        request=evidence.get("request", {}),
        recommendation=evidence.get("recommendation", "Recommendation Pending"),
        rationale=evidence.get("rationale", ""),
        financial=evidence.get("financial", {}),
        budget=evidence.get("budget", {}),
        routing=evidence.get("routing", {}),
        comparables=evidence.get("comparables", []),
        documents=[],
        variance={},
        conditions=evidence.get("conditions", []),
        risk_flags=evidence.get("risk_flags", []),
        tool_calls=[step.get("step", "") for step in evidence.get("workflow_steps", [])],
    )


def _render_memo(
    request: dict[str, Any],
    recommendation: str,
    rationale: str,
    financial: dict[str, Any],
    budget: dict[str, Any],
    routing: dict[str, Any],
    comparables: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    variance: dict[str, Any],
    conditions: list[str],
    risk_flags: list[str],
    tool_calls: list[str],
) -> str:
    title = request.get("title", "CapEx Request")
    amount = request.get("amount_m", request.get("amount", "N/A"))
    pool = request.get("budget_pool", "N/A")
    priority = request.get("priority_tag", "N/A")
    urgency = request.get("urgency", "N/A")
    completion = request.get("completion_months", request.get("expected_completion_months", "N/A"))

    memo = [
        f"# Investment Memo: {title}",
        "",
        f"**Date:** {date.today().isoformat()}",
        "**Prepared by:** TeraWave CapVal Agent",
        "**Data status:** Demo data only - not an official approval record.",
        "",
        "## Executive Recommendation",
        "",
        f"**Recommendation:** {recommendation}",
        "",
        _short_rationale(rationale),
        "",
        "## Request Snapshot",
        "",
        f"- **Amount:** ${amount}M",
        f"- **Budget pool:** {pool}",
        f"- **Priority:** {priority}",
        f"- **Urgency:** {urgency}",
        f"- **Expected completion:** {completion} months",
        "",
        "## Tool Evidence Driving The Recommendation",
        "",
        _tools_called_line(tool_calls),
        _budget_line(budget),
        _financial_line(financial),
        _routing_line(routing),
        _comparables_line(comparables),
        _documents_line(documents),
        _variance_line(variance),
        "",
        "## Conditions And Risk Controls",
        "",
        _list_or_default(conditions, "No additional conditions surfaced by the workflow."),
        _list_or_default(risk_flags, "No blocking risk flags surfaced by the workflow."),
        "",
        "## Audit Trail",
        "",
        _list_or_default(tool_calls, "No tool calls captured."),
        "",
        "## Deployment Note",
        "",
        (
            "This output is decision support, not automated approval. In production, the system would require "
            "human review, source-system permissions, prompt-injection controls, monitoring for numerical "
            "consistency, and integration with ERP, planning, document, procurement, and workflow systems."
        ),
    ]
    return "\n".join(memo)


def _first(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    return items[0] if items else {}


def _short_rationale(text: str) -> str:
    if not text:
        return "The recommendation is based on budget fit, financial impact, approval routing, precedent, and risk evidence."
    cleaned = " ".join(text.split())
    return cleaned[:800] + ("..." if len(cleaned) > 800 else "")


def _recommendation_from_text(text: str) -> str:
    upper = text.upper()
    if "APPROVE WITH CONDITIONS" in upper or "CONDITIONAL" in upper:
        return "CONDITIONAL GO"
    if "DEFER" in upper:
        return "DEFER"
    if "REJECT" in upper or "NO-GO" in upper or "NO GO" in upper:
        return "NO-GO"
    if "APPROVE" in upper or " GO" in upper:
        return "GO"
    return "Recommendation Pending"


def _financial_line(financial: dict[str, Any]) -> str:
    if not financial:
        return "- **Financial impact tool:** Not captured."
    return (
        f"- **Financial impact tool:** Estimated NPV impact of ${financial.get('npv_impact_m', 'N/A')}M, "
        f"ROI of {financial.get('roi_pct', 'N/A')}%, payback of {financial.get('payback_months', 'N/A')} months, "
        f"progress score {financial.get('progress_score', 'N/A')}, and risk-retirement score {financial.get('risk_retirement_score', 'N/A')}."
    )


def _budget_line(budget: dict[str, Any]) -> str:
    if not budget:
        return "- **Budget tool:** Not captured."
    available = budget.get("available_m", budget.get("budget_available_m", "N/A"))
    status = budget.get("status", "N/A")
    total = budget.get("total_budget_m")
    spent = budget.get("spent_m")
    committed = budget.get("committed_m")
    utilization = budget.get("utilization_pct")
    suffix = f", utilization {utilization}%" if utilization is not None else ""
    if total is not None and spent is not None and committed is not None:
        return (
            f"- **Budget tool:** ${available}M remaining headroom "
            f"(total ${total}M, spent ${spent}M, committed ${committed}M); status {status}{suffix}."
        )
    return f"- **Budget tool:** ${available}M remaining headroom; status {status}{suffix}."


def _routing_line(routing: dict[str, Any]) -> str:
    if not routing:
        return "- **Approval-routing tool:** Not captured."
    tier = routing.get("tier", routing.get("approval_tier", "N/A"))
    label = routing.get("label", routing.get("approver", routing.get("approval_tier_label", "N/A")))
    sla = routing.get("sla_days", "N/A")
    return f"- **Approval-routing tool:** Routes to Tier {tier} ({label}) with an SLA of {sla} business days."


def _variance_line(variance: dict[str, Any]) -> str:
    if not variance:
        return "- **Variance tool:** Not captured or not applicable."
    if "status" in variance:
        return (
            f"- **Variance tool:** Workstream status {variance.get('status')}; "
            f"YTD variance {variance.get('ytd_variance_m', 'N/A')}M "
            f"({variance.get('ytd_variance_pct', 'N/A')}%), active alerts {variance.get('active_alerts', 'N/A')}."
        )
    return f"- **Variance tool:** {json.dumps(variance)[:220]}."


def _comparables_line(comparables: list[dict[str, Any]]) -> str:
    if not comparables:
        return "- **Precedent tool:** No comparable requests captured."
    approved = sum(1 for item in comparables if "approved" in str(item.get("status", "")).lower())
    avg_roi = _average_roi(comparables)
    labels = [
        f"{item.get('id', 'N/A')} ({item.get('status', 'N/A')}, ${item.get('amount_m', 'N/A')}M, NPV ${item.get('npv_impact_m', 'N/A')}M)"
        for item in comparables[:3]
    ]
    roi_suffix = f", average historical ROI {avg_roi:.0f}%" if avg_roi is not None else ""
    return f"- **Precedent tool:** {approved} of {len(comparables)} comparable requests approved{roi_suffix}; examples: {'; '.join(labels)}."


def _documents_line(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "- **Document-search tool:** No document citations captured."
    labels = [
        f"{doc.get('title', 'Untitled')} [{doc.get('doc_id', doc.get('id', 'N/A'))}]"
        for doc in documents[:3]
    ]
    return f"- **Document-search tool:** Cited {'; '.join(labels)}."


def _tools_called_line(tool_calls: list[str]) -> str:
    if not tool_calls:
        return "- **Tools called:** No tool calls captured."
    normalized = [str(item).replace("_", " ") for item in tool_calls]
    return f"- **Tools called:** {', '.join(normalized)}."


def _average_roi(comparables: list[dict[str, Any]]) -> float | None:
    rois = []
    for item in comparables:
        amount = item.get("amount_m")
        npv = item.get("npv_impact_m")
        if amount and npv is not None:
            rois.append((npv / amount - 1) * 100)
    return sum(rois) / len(rois) if rois else None


def _list_or_default(items: list[Any], default: str) -> str:
    if not items:
        return f"- {default}"
    return "\n".join(f"- {item}" for item in items)
