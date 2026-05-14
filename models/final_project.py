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
        "## Evidence Pack",
        "",
        _financial_line(financial),
        _budget_line(budget),
        _routing_line(routing),
        _variance_line(variance),
        _comparables_line(comparables),
        _documents_line(documents),
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
        return "- **Financial impact:** Not captured."
    return (
        f"- **Financial impact:** NPV ${financial.get('npv_impact_m', 'N/A')}M, "
        f"ROI {financial.get('roi_pct', 'N/A')}%, payback {financial.get('payback_months', 'N/A')} months, "
        f"progress score {financial.get('progress_score', 'N/A')}, risk-retirement score {financial.get('risk_retirement_score', 'N/A')}."
    )


def _budget_line(budget: dict[str, Any]) -> str:
    if not budget:
        return "- **Budget:** Not captured."
    available = budget.get("available_m", budget.get("budget_available_m", "N/A"))
    status = budget.get("status", "N/A")
    utilization = budget.get("utilization_pct")
    suffix = f", utilization {utilization}%" if utilization is not None else ""
    return f"- **Budget:** {status}; available ${available}M{suffix}."


def _routing_line(routing: dict[str, Any]) -> str:
    if not routing:
        return "- **Approval routing:** Not captured."
    tier = routing.get("tier", routing.get("approval_tier", "N/A"))
    label = routing.get("label", routing.get("approver", routing.get("approval_tier_label", "N/A")))
    sla = routing.get("sla_days", "N/A")
    return f"- **Approval routing:** Tier {tier}, {label}, SLA {sla} business days."


def _variance_line(variance: dict[str, Any]) -> str:
    if not variance:
        return "- **Variance context:** Not captured or not applicable."
    if "status" in variance:
        return f"- **Variance context:** {variance.get('status')}."
    return f"- **Variance context:** {json.dumps(variance)[:220]}."


def _comparables_line(comparables: list[dict[str, Any]]) -> str:
    if not comparables:
        return "- **Precedent:** No comparable requests captured."
    labels = [f"{item.get('id', 'N/A')} ({item.get('status', 'N/A')}, ${item.get('amount_m', 'N/A')}M)" for item in comparables[:3]]
    return f"- **Precedent:** {'; '.join(labels)}."


def _documents_line(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "- **Source documents:** No document citations captured."
    labels = [
        f"{doc.get('title', 'Untitled')} [{doc.get('doc_id', doc.get('id', 'N/A'))}]"
        for doc in documents[:3]
    ]
    return f"- **Source documents:** {'; '.join(labels)}."


def _list_or_default(items: list[Any], default: str) -> str:
    if not items:
        return f"- {default}"
    return "\n".join(f"- {item}" for item in items)
