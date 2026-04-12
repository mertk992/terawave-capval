"""
Capital Expenditure Request Workflow Engine

Automates the CapEx approval pipeline:
  Submit → Validate → Analyze → Compare → Route → Recommend

Replaces what is typically a spreadsheet + email chain process.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json


# ── Approval Tiers ───────────────────────────────────────────────────────────

APPROVAL_TIERS = [
    {"tier": 1, "label": "Program Manager",     "max_amount_m": 5,    "sla_days": 3},
    {"tier": 2, "label": "VP of Finance",        "max_amount_m": 25,   "sla_days": 5},
    {"tier": 3, "label": "CFO",                  "max_amount_m": 100,  "sla_days": 7},
    {"tier": 4, "label": "CEO / Board",          "max_amount_m": 1e9,  "sla_days": 14},
]

BUDGET_POOLS = {
    "TeraWave — Satellite Manufacturing": {"budget_m": 4_500, "spent_m": 1_200, "committed_m": 800},
    "TeraWave — Ground Segment":          {"budget_m": 2_500, "spent_m": 600,   "committed_m": 400},
    "TeraWave — Launch Services":         {"budget_m": 2_700, "spent_m": 900,   "committed_m": 500},
    "TeraWave — OISL & Comms":            {"budget_m": 1_200, "spent_m": 300,   "committed_m": 200},
    "TeraWave — Software & Network Ops":  {"budget_m": 800,   "spent_m": 200,   "committed_m": 100},
    "TeraWave — R&D & Prototyping":       {"budget_m": 1_500, "spent_m": 700,   "committed_m": 300},
    "New Glenn — Operations":             {"budget_m": 1_200, "spent_m": 500,   "committed_m": 200},
    "Blue Moon — Development":            {"budget_m": 1_000, "spent_m": 400,   "committed_m": 150},
}

PRIORITY_TAGS = ["Critical Path", "Risk Retirement", "Cost Reduction", "Capacity Expansion", "R&D / Innovation", "Maintenance"]

URGENCY_LEVELS = ["Standard", "Expedited", "Emergency"]


@dataclass
class CapExRequest:
    """A capital expenditure request submitted for approval."""
    id: str = ""
    title: str = ""
    description: str = ""
    requestor: str = ""
    department: str = ""
    budget_pool: str = ""
    amount_m: float = 0.0
    priority_tag: str = "Standard"
    urgency: str = "Standard"
    justification: str = ""
    expected_completion_months: int = 12
    # Auto-populated
    submission_date: str = ""
    status: str = "Draft"
    approval_tier: int = 0
    approval_tier_label: str = ""


@dataclass
class WorkflowResult:
    """Output of the automated workflow analysis."""
    request: CapExRequest
    # Validation
    validation_passed: bool = True
    validation_issues: list[str] = field(default_factory=list)
    # Budget check
    budget_available_m: float = 0.0
    budget_remaining_pct: float = 0.0
    budget_status: str = ""  # "Within Budget", "Exceeds Remaining", "Over Committed"
    # Financial analysis
    npv_impact_m: float = 0.0
    payback_months: int = 0
    roi_pct: float = 0.0
    progress_score: float = 0.0
    risk_retirement_score: float = 0.0
    # Comparable requests
    comparables: list[dict] = field(default_factory=list)
    # Routing
    approval_tier: int = 0
    approval_tier_label: str = ""
    sla_days: int = 0
    # Recommendation
    recommendation: str = ""  # "Approve", "Approve with Conditions", "Defer", "Reject"
    recommendation_rationale: str = ""
    conditions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    # Audit
    workflow_steps: list[dict] = field(default_factory=list)


# ── Synthetic Historical Requests ────────────────────────────────────────────

HISTORICAL_REQUESTS = [
    {
        "id": "CR-2025-001", "title": "Q/V-Band Test Array Fabrication",
        "budget_pool": "TeraWave — Satellite Manufacturing", "amount_m": 12.5,
        "priority_tag": "Critical Path", "status": "Approved", "outcome": "On track",
        "npv_impact_m": 45.0, "payback_months": 18, "approval_tier": 2,
    },
    {
        "id": "CR-2025-002", "title": "Gateway Site Land Acquisition — Singapore",
        "budget_pool": "TeraWave — Ground Segment", "amount_m": 85.0,
        "priority_tag": "Critical Path", "status": "Approved", "outcome": "Completed",
        "npv_impact_m": 220.0, "payback_months": 36, "approval_tier": 3,
    },
    {
        "id": "CR-2025-003", "title": "Optical Terminal Prototype (Gen 2)",
        "budget_pool": "TeraWave — OISL & Comms", "amount_m": 8.2,
        "priority_tag": "Risk Retirement", "status": "Approved", "outcome": "Completed ahead of schedule",
        "npv_impact_m": 35.0, "payback_months": 12, "approval_tier": 2,
    },
    {
        "id": "CR-2025-004", "title": "Additional Launch Pad Modifications",
        "budget_pool": "TeraWave — Launch Services", "amount_m": 45.0,
        "priority_tag": "Capacity Expansion", "status": "Approved with Conditions", "outcome": "In progress",
        "npv_impact_m": 95.0, "payback_months": 24, "approval_tier": 3,
    },
    {
        "id": "CR-2025-005", "title": "Network Orchestration Software Suite",
        "budget_pool": "TeraWave — Software & Network Ops", "amount_m": 15.0,
        "priority_tag": "Critical Path", "status": "Approved", "outcome": "In progress",
        "npv_impact_m": 60.0, "payback_months": 15, "approval_tier": 2,
    },
    {
        "id": "CR-2025-006", "title": "Backup Transponder Inventory",
        "budget_pool": "TeraWave — Satellite Manufacturing", "amount_m": 3.2,
        "priority_tag": "Maintenance", "status": "Approved", "outcome": "Completed",
        "npv_impact_m": 5.0, "payback_months": 6, "approval_tier": 1,
    },
    {
        "id": "CR-2025-007", "title": "MEO Satellite Thermal Redesign Study",
        "budget_pool": "TeraWave — R&D & Prototyping", "amount_m": 2.8,
        "priority_tag": "Risk Retirement", "status": "Approved", "outcome": "Completed — retired thermal risk",
        "npv_impact_m": 120.0, "payback_months": 9, "approval_tier": 1,
    },
    {
        "id": "CR-2025-008", "title": "Ground Station Fiber Backhaul — Brazil",
        "budget_pool": "TeraWave — Ground Segment", "amount_m": 22.0,
        "priority_tag": "Capacity Expansion", "status": "Deferred", "outcome": "Resubmitted Q3",
        "npv_impact_m": 40.0, "payback_months": 30, "approval_tier": 2,
    },
    {
        "id": "CR-2025-009", "title": "Autonomous Collision Avoidance System",
        "budget_pool": "TeraWave — Software & Network Ops", "amount_m": 6.5,
        "priority_tag": "Risk Retirement", "status": "Approved", "outcome": "In progress",
        "npv_impact_m": 200.0, "payback_months": 8, "approval_tier": 2,
    },
    {
        "id": "CR-2025-010", "title": "Spectrum License Extension — EU",
        "budget_pool": "TeraWave — R&D & Prototyping", "amount_m": 35.0,
        "priority_tag": "Critical Path", "status": "Approved", "outcome": "Completed",
        "npv_impact_m": 500.0, "payback_months": 48, "approval_tier": 3,
    },
]


# ── Workflow Engine ──────────────────────────────────────────────────────────

def run_capex_workflow(request: CapExRequest) -> WorkflowResult:
    """Execute the full automated CapEx approval workflow."""
    result = WorkflowResult(request=request)
    result.workflow_steps = []

    # ── Step 1: Validate ──
    _step_validate(request, result)

    # ── Step 2: Budget Check ──
    _step_budget_check(request, result)

    # ── Step 3: Financial Analysis ──
    _step_financial_analysis(request, result)

    # ── Step 4: Find Comparables ──
    _step_find_comparables(request, result)

    # ── Step 5: Route to Approval Tier ──
    _step_route_approval(request, result)

    # ── Step 6: Generate Recommendation ──
    _step_recommend(request, result)

    return result


def _step_validate(req: CapExRequest, res: WorkflowResult):
    """Validate the request for completeness and policy compliance."""
    issues = []
    if not req.title:
        issues.append("Title is required")
    if req.amount_m <= 0:
        issues.append("Amount must be positive")
    if not req.budget_pool or req.budget_pool not in BUDGET_POOLS:
        issues.append(f"Invalid budget pool: {req.budget_pool}")
    if not req.justification or len(req.justification) < 20:
        issues.append("Justification must be at least 20 characters")
    if req.urgency == "Emergency" and req.priority_tag not in ["Critical Path", "Risk Retirement"]:
        issues.append("Emergency urgency requires Critical Path or Risk Retirement priority")

    res.validation_passed = len(issues) == 0
    res.validation_issues = issues
    res.workflow_steps.append({
        "step": "Validation",
        "status": "Passed" if res.validation_passed else "Failed",
        "details": "All checks passed" if not issues else "; ".join(issues),
    })


def _step_budget_check(req: CapExRequest, res: WorkflowResult):
    """Check request against budget pool availability."""
    if req.budget_pool not in BUDGET_POOLS:
        res.budget_status = "Invalid Pool"
        res.workflow_steps.append({"step": "Budget Check", "status": "Skipped", "details": "Invalid budget pool"})
        return

    pool = BUDGET_POOLS[req.budget_pool]
    available = pool["budget_m"] - pool["spent_m"] - pool["committed_m"]
    res.budget_available_m = available
    res.budget_remaining_pct = (available / pool["budget_m"]) * 100 if pool["budget_m"] > 0 else 0

    if req.amount_m <= available:
        res.budget_status = "Within Budget"
    elif req.amount_m <= available * 1.1:  # 10% buffer
        res.budget_status = "Near Limit (within 10% buffer)"
    else:
        res.budget_status = "Exceeds Available Budget"

    res.workflow_steps.append({
        "step": "Budget Check",
        "status": res.budget_status,
        "details": f"Requested: ${req.amount_m:.1f}M | Available: ${available:.1f}M ({res.budget_remaining_pct:.0f}% of pool remaining)",
    })


def _step_financial_analysis(req: CapExRequest, res: WorkflowResult):
    """Run automated financial impact analysis."""
    # Synthetic financial analysis based on priority and amount
    priority_multipliers = {
        "Critical Path": 3.5, "Risk Retirement": 4.0, "Cost Reduction": 2.5,
        "Capacity Expansion": 2.0, "R&D / Innovation": 3.0, "Maintenance": 1.2,
    }
    mult = priority_multipliers.get(req.priority_tag, 1.5)

    # NPV estimate (synthetic — based on priority type and amount)
    res.npv_impact_m = round(req.amount_m * mult * (1 + np.random.RandomState(hash(req.title) % 2**31).uniform(-0.3, 0.3)), 1)
    res.payback_months = max(3, int(req.expected_completion_months * (req.amount_m / (res.npv_impact_m + 0.01)) * 12))
    res.payback_months = min(res.payback_months, 60)  # cap at 5 years
    res.roi_pct = round((res.npv_impact_m / req.amount_m - 1) * 100, 1) if req.amount_m > 0 else 0

    # Progress and risk scores (0-1) based on priority tag
    progress_map = {"Critical Path": 0.9, "Capacity Expansion": 0.7, "R&D / Innovation": 0.6, "Cost Reduction": 0.4, "Risk Retirement": 0.5, "Maintenance": 0.2}
    risk_map = {"Risk Retirement": 0.9, "Critical Path": 0.6, "R&D / Innovation": 0.7, "Cost Reduction": 0.3, "Capacity Expansion": 0.4, "Maintenance": 0.1}
    res.progress_score = progress_map.get(req.priority_tag, 0.5)
    res.risk_retirement_score = risk_map.get(req.priority_tag, 0.3)

    res.workflow_steps.append({
        "step": "Financial Analysis",
        "status": "Complete",
        "details": f"NPV Impact: ${res.npv_impact_m:.1f}M | ROI: {res.roi_pct:.0f}% | Payback: {res.payback_months} months | Progress Score: {res.progress_score:.1f} | Risk Retirement: {res.risk_retirement_score:.1f}",
    })


def _step_find_comparables(req: CapExRequest, res: WorkflowResult):
    """Find similar historical requests for context."""
    comps = []
    for hist in HISTORICAL_REQUESTS:
        score = 0
        if hist["budget_pool"] == req.budget_pool:
            score += 3
        if hist["priority_tag"] == req.priority_tag:
            score += 2
        if abs(hist["amount_m"] - req.amount_m) / max(req.amount_m, 1) < 0.5:
            score += 2
        if score >= 3:
            comps.append({**hist, "relevance_score": score})

    comps.sort(key=lambda x: x["relevance_score"], reverse=True)
    res.comparables = comps[:3]

    res.workflow_steps.append({
        "step": "Comparable Analysis",
        "status": "Complete",
        "details": f"Found {len(res.comparables)} comparable past requests",
    })


def _step_route_approval(req: CapExRequest, res: WorkflowResult):
    """Route to the appropriate approval tier based on amount."""
    for tier in APPROVAL_TIERS:
        if req.amount_m <= tier["max_amount_m"]:
            res.approval_tier = tier["tier"]
            res.approval_tier_label = tier["label"]
            res.sla_days = tier["sla_days"]
            if req.urgency == "Emergency":
                res.sla_days = max(1, res.sla_days // 2)
            elif req.urgency == "Expedited":
                res.sla_days = max(2, int(res.sla_days * 0.7))
            break

    res.workflow_steps.append({
        "step": "Approval Routing",
        "status": f"Tier {res.approval_tier}",
        "details": f"Routed to: {res.approval_tier_label} | SLA: {res.sla_days} business days ({req.urgency})",
    })


def _step_recommend(req: CapExRequest, res: WorkflowResult):
    """Generate automated recommendation based on all analysis."""
    flags = []

    # Scoring
    score = 0

    # Financial merit
    if res.roi_pct > 100:
        score += 3
    elif res.roi_pct > 50:
        score += 2
    elif res.roi_pct > 0:
        score += 1

    # Budget fit
    if res.budget_status == "Within Budget":
        score += 2
    elif "Near Limit" in res.budget_status:
        score += 1
        flags.append("Near budget limit — consider phased spending")
    else:
        score -= 2
        flags.append("EXCEEDS available budget — requires reallocation or supplemental funding")

    # Strategic alignment
    if req.priority_tag in ["Critical Path", "Risk Retirement"]:
        score += 2
    elif req.priority_tag in ["R&D / Innovation", "Capacity Expansion"]:
        score += 1

    # Progress acceleration
    if res.progress_score >= 0.7:
        score += 1
    if res.risk_retirement_score >= 0.7:
        score += 1

    # Payback
    if res.payback_months > 36:
        flags.append("Long payback period (>3 years) — confirm strategic necessity")
    if res.payback_months <= 12:
        score += 1

    # Validation
    if not res.validation_passed:
        score -= 3
        flags.append("Validation issues must be resolved before approval")

    # Determine recommendation
    conditions = []
    if score >= 6:
        res.recommendation = "Approve"
        res.recommendation_rationale = "Strong financial merit, strategic alignment, and budget availability support approval."
    elif score >= 3:
        res.recommendation = "Approve with Conditions"
        if res.budget_status != "Within Budget":
            conditions.append("Identify budget reallocation source before commitment")
        if res.payback_months > 24:
            conditions.append("Establish milestone-based release of funds")
        if req.amount_m > 20:
            conditions.append("Quarterly progress reporting to finance")
        if not conditions:
            conditions.append("Confirm timeline alignment with program schedule")
        res.recommendation_rationale = "Merits approval but requires conditions to manage identified risks."
    elif score >= 0:
        res.recommendation = "Defer"
        res.recommendation_rationale = "Insufficient financial justification or budget constraints. Recommend resubmission with additional analysis."
    else:
        res.recommendation = "Reject"
        res.recommendation_rationale = "Does not meet minimum thresholds for financial merit, strategic alignment, or budget availability."

    res.conditions = conditions
    res.risk_flags = flags

    res.workflow_steps.append({
        "step": "Recommendation",
        "status": res.recommendation,
        "details": res.recommendation_rationale,
    })


def get_historical_df() -> pd.DataFrame:
    """Return historical requests as a DataFrame."""
    return pd.DataFrame(HISTORICAL_REQUESTS)


def get_budget_summary() -> pd.DataFrame:
    """Return budget pool summary."""
    records = []
    for pool, data in BUDGET_POOLS.items():
        available = data["budget_m"] - data["spent_m"] - data["committed_m"]
        records.append({
            "Budget Pool": pool,
            "Total Budget ($M)": data["budget_m"],
            "Spent ($M)": data["spent_m"],
            "Committed ($M)": data["committed_m"],
            "Available ($M)": available,
            "Utilization": f"{((data['spent_m'] + data['committed_m']) / data['budget_m'] * 100):.0f}%",
        })
    return pd.DataFrame(records)
