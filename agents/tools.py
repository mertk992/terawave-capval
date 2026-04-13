"""
Agent Tools — Functions that agents can autonomously call via Claude tool_use.

These are real functions the agent decides to invoke based on what
information it needs. The agent plans, selects tools, interprets results,
and chains calls — this is what makes the system agentic.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

from __future__ import annotations

import json
import numpy as np

# ── Tool Definitions (Claude API format) ─────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "check_budget",
        "description": "Check the current budget status for a specific budget pool. Returns total budget, amount spent, committed, and available. Use this to verify whether a CapEx request fits within available budget.",
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_pool": {
                    "type": "string",
                    "description": "The budget pool to check, e.g. 'TeraWave — Ground Segment & Gateways'"
                }
            },
            "required": ["budget_pool"],
        },
    },
    {
        "name": "search_documents",
        "description": "Search the TeraWave program document corpus for relevant contracts, policies, memos, and reports. Use this to find policy requirements, contract terms, vendor details, or technical risk assessments relevant to an analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query, e.g. 'capital allocation approval thresholds' or 'optical link risk assessment'"
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_comparable_requests",
        "description": "Find historical CapEx requests similar to the current one. Returns past requests from the same budget pool or with similar priority/amount, including their outcomes. Use this to establish precedent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_pool": {
                    "type": "string",
                    "description": "Budget pool to filter by"
                },
                "priority_tag": {
                    "type": "string",
                    "description": "Priority tag to match, e.g. 'Critical Path', 'Risk Retirement'"
                },
                "amount_m": {
                    "type": "number",
                    "description": "Request amount in $M to find similar-sized requests"
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_financial_impact",
        "description": "Run a financial impact analysis for a proposed capital expenditure. Estimates NPV impact, ROI, and payback period based on the workstream, amount, and priority. Use this to quantify the financial case.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount_m": {
                    "type": "number",
                    "description": "The CapEx amount in $M"
                },
                "priority_tag": {
                    "type": "string",
                    "description": "Priority category: 'Critical Path', 'Risk Retirement', 'Cost Reduction', 'Capacity Expansion', 'R&D / Innovation', 'Maintenance'"
                },
                "completion_months": {
                    "type": "integer",
                    "description": "Expected months to complete"
                }
            },
            "required": ["amount_m", "priority_tag"],
        },
    },
    {
        "name": "check_approval_routing",
        "description": "Determine which approval tier a request routes to based on its dollar amount, and the SLA for that tier. Use this to inform the requestor about the approval process.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount_m": {
                    "type": "number",
                    "description": "Request amount in $M"
                },
                "urgency": {
                    "type": "string",
                    "description": "Urgency level: 'Standard', 'Expedited', 'Emergency'"
                }
            },
            "required": ["amount_m"],
        },
    },
    {
        "name": "get_variance_status",
        "description": "Get the current budget vs. actual variance status for a workstream. Returns YTD spend, variance, and any active alerts. Use this to check if a workstream is already over/under budget before approving new spend.",
        "input_schema": {
            "type": "object",
            "properties": {
                "workstream": {
                    "type": "string",
                    "description": "Workstream name, e.g. 'Ground Segment & Gateways'"
                }
            },
            "required": ["workstream"],
        },
    },
]


# ── Tool Implementations ─────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result as a string."""
    if tool_name == "check_budget":
        return _tool_check_budget(tool_input)
    elif tool_name == "search_documents":
        return _tool_search_documents(tool_input)
    elif tool_name == "get_comparable_requests":
        return _tool_get_comparable_requests(tool_input)
    elif tool_name == "run_financial_impact":
        return _tool_run_financial_impact(tool_input)
    elif tool_name == "check_approval_routing":
        return _tool_check_approval_routing(tool_input)
    elif tool_name == "get_variance_status":
        return _tool_get_variance_status(tool_input)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _tool_check_budget(input: dict) -> str:
    from models.capex_workflow import BUDGET_POOLS
    pool_name = input.get("budget_pool", "")

    # Fuzzy match
    matched = None
    for name in BUDGET_POOLS:
        if pool_name.lower() in name.lower() or name.lower() in pool_name.lower():
            matched = name
            break

    if not matched:
        return json.dumps({"error": f"Budget pool '{pool_name}' not found. Available: {list(BUDGET_POOLS.keys())}"})

    pool = BUDGET_POOLS[matched]
    available = pool["budget_m"] - pool["spent_m"] - pool["committed_m"]
    utilization = (pool["spent_m"] + pool["committed_m"]) / pool["budget_m"] * 100

    return json.dumps({
        "budget_pool": matched,
        "total_budget_m": pool["budget_m"],
        "spent_m": pool["spent_m"],
        "committed_m": pool["committed_m"],
        "available_m": available,
        "utilization_pct": round(utilization, 1),
        "status": "healthy" if utilization < 70 else "caution" if utilization < 90 else "critical",
    })


def _tool_search_documents(input: dict) -> str:
    from models.rag_engine import search_documents
    query = input.get("query", "")
    results = search_documents(query, top_k=2)

    docs = []
    for doc, score in results:
        # Return key excerpts, not full content
        content = doc.content[:1500]
        docs.append({
            "title": doc.title,
            "doc_id": doc.id,
            "type": doc.doc_type,
            "date": doc.date,
            "relevance": round(score, 3),
            "excerpt": content,
            "metadata": doc.metadata,
        })

    return json.dumps({"query": query, "results_found": len(docs), "documents": docs})


def _tool_get_comparable_requests(input: dict) -> str:
    from models.capex_workflow import HISTORICAL_REQUESTS
    pool = input.get("budget_pool", "")
    priority = input.get("priority_tag", "")
    amount = input.get("amount_m", 0)

    matches = []
    for req in HISTORICAL_REQUESTS:
        score = 0
        if pool and pool.lower() in req["budget_pool"].lower():
            score += 3
        if priority and req["priority_tag"] == priority:
            score += 2
        if amount > 0 and abs(req["amount_m"] - amount) / max(amount, 1) < 0.5:
            score += 2
        if score >= 2:
            matches.append({**req, "relevance_score": score})

    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    return json.dumps({"comparables_found": len(matches[:5]), "requests": matches[:5]})


def _tool_run_financial_impact(input: dict) -> str:
    amount = input.get("amount_m", 0)
    priority = input.get("priority_tag", "Standard")
    months = input.get("completion_months", 12)

    priority_multipliers = {
        "Critical Path": 3.5, "Risk Retirement": 4.0, "Cost Reduction": 2.5,
        "Capacity Expansion": 2.0, "R&D / Innovation": 3.0, "Maintenance": 1.2,
    }
    mult = priority_multipliers.get(priority, 1.5)

    rng = np.random.RandomState(42)
    npv_impact = round(amount * mult * (1 + rng.uniform(-0.2, 0.2)), 1)
    roi = round((npv_impact / amount - 1) * 100, 1) if amount > 0 else 0
    payback = max(3, min(60, int(months * (amount / (npv_impact + 0.01)) * 12)))

    progress_map = {"Critical Path": 0.9, "Capacity Expansion": 0.7, "R&D / Innovation": 0.6,
                    "Cost Reduction": 0.4, "Risk Retirement": 0.5, "Maintenance": 0.2}
    risk_map = {"Risk Retirement": 0.9, "Critical Path": 0.6, "R&D / Innovation": 0.7,
                "Cost Reduction": 0.3, "Capacity Expansion": 0.4, "Maintenance": 0.1}

    return json.dumps({
        "amount_m": amount,
        "npv_impact_m": npv_impact,
        "roi_pct": roi,
        "payback_months": payback,
        "progress_score": progress_map.get(priority, 0.5),
        "risk_retirement_score": risk_map.get(priority, 0.3),
        "assessment": "strong" if roi > 100 else "moderate" if roi > 30 else "marginal",
    })


def _tool_check_approval_routing(input: dict) -> str:
    from models.capex_workflow import APPROVAL_TIERS
    amount = input.get("amount_m", 0)
    urgency = input.get("urgency", "Standard")

    for tier in APPROVAL_TIERS:
        if amount <= tier["max_amount_m"]:
            sla = tier["sla_days"]
            if urgency == "Emergency":
                sla = max(1, sla // 2)
            elif urgency == "Expedited":
                sla = max(2, int(sla * 0.7))

            return json.dumps({
                "tier": tier["tier"],
                "approver": tier["label"],
                "sla_days": sla,
                "urgency": urgency,
                "amount_m": amount,
            })

    return json.dumps({"error": "Amount exceeds all tiers"})


def _tool_get_variance_status(input: dict) -> str:
    from models.variance_engine import get_ytd_summary, detect_anomalies, WORKSTREAMS

    ws = input.get("workstream", "")

    # Fuzzy match
    matched = None
    for name in WORKSTREAMS:
        if ws.lower() in name.lower() or name.lower() in ws.lower():
            matched = name
            break

    if not matched:
        return json.dumps({"error": f"Workstream '{ws}' not found. Available: {WORKSTREAMS}"})

    ytd = get_ytd_summary(6)
    row = ytd[ytd["Workstream"] == matched].iloc[0]

    alerts = detect_anomalies(6)
    ws_alerts = [a for a in alerts if a.workstream == matched]

    return json.dumps({
        "workstream": matched,
        "ytd_budget_m": float(row["YTD Budget ($M)"]),
        "ytd_actual_m": float(row["YTD Actual ($M)"]),
        "ytd_variance_m": float(row["YTD Variance ($M)"]),
        "ytd_variance_pct": float(row["YTD Variance (%)"]),
        "status": row["Status"],
        "active_alerts": len(ws_alerts),
        "alert_details": [{"month": a.month, "variance_pct": a.variance_pct, "commentary": a.auto_commentary} for a in ws_alerts[:3]],
    })
