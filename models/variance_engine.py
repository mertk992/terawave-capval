"""
FP&A Variance Analysis Engine

Compares budget vs. actual spending by workstream and month,
auto-detects anomalies, and generates variance commentary.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ── Synthetic Monthly Budget & Actuals ───────────────────────────────────────

WORKSTREAMS = [
    "Satellite Manufacturing (LEO)",
    "Satellite Manufacturing (MEO)",
    "Launch Services",
    "Ground Segment & Gateways",
    "Optical Inter-Satellite Links",
    "Software & Network Ops",
    "R&D & Prototyping",
    "Program Management",
]

# Monthly budget plan ($M) — 12 months of FY2026
MONTHLY_BUDGET = {
    "Satellite Manufacturing (LEO)":   [25, 28, 32, 35, 38, 42, 45, 48, 50, 52, 55, 58],
    "Satellite Manufacturing (MEO)":   [18, 20, 22, 25, 28, 30, 32, 35, 35, 38, 40, 42],
    "Launch Services":                 [15, 15, 18, 20, 22, 25, 28, 30, 32, 35, 35, 38],
    "Ground Segment & Gateways":       [22, 25, 28, 30, 35, 38, 35, 32, 30, 28, 25, 22],
    "Optical Inter-Satellite Links":   [12, 14, 16, 18, 18, 20, 20, 18, 16, 14, 12, 10],
    "Software & Network Ops":          [8,  8,  10, 10, 12, 12, 14, 14, 15, 15, 16, 16],
    "R&D & Prototyping":               [15, 14, 12, 10, 10, 8,  8,  6,  6,  5,  5,  4],
    "Program Management":              [5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _generate_actuals(seed: int = 42) -> dict:
    """
    Generate realistic synthetic actual spending with plausible variance patterns.
    Some workstreams run over, some under, some have specific stories.
    """
    rng = np.random.RandomState(seed)
    actuals = {}

    # Variance patterns — each workstream has a "story"
    patterns = {
        "Satellite Manufacturing (LEO)": {
            "bias": 0.02,   # slightly over budget — production ramp challenges
            "noise": 0.08,
            "story": "Production ramp challenges in Q1; yield improvements in Q3",
            "spike_month": 4, "spike_pct": 0.25,  # April spike — tooling investment
        },
        "Satellite Manufacturing (MEO)": {
            "bias": -0.05,  # under budget — vendor delivered ahead of schedule
            "noise": 0.06,
            "story": "Northstar ahead of schedule on NRE; lower unit costs than projected",
        },
        "Launch Services": {
            "bias": 0.00,
            "noise": 0.05,
            "story": "On plan; minor timing shifts between months",
        },
        "Ground Segment & Gateways": {
            "bias": 0.18,   # significantly over — Singapore site acceleration
            "noise": 0.10,
            "story": "Singapore gateway accelerated per program directive; land costs 22% above estimate",
            "spike_month": 5, "spike_pct": 0.35,
        },
        "Optical Inter-Satellite Links": {
            "bias": 0.10,   # over — additional risk retirement investment
            "noise": 0.12,
            "story": "Incremental $28M for automated alignment station (risk retirement investment)",
            "spike_month": 3, "spike_pct": 0.40,
        },
        "Software & Network Ops": {
            "bias": -0.08,  # under — team hired slower than planned
            "noise": 0.06,
            "story": "Hiring ramp 2 months behind plan; contractor bridge in Q2",
        },
        "R&D & Prototyping": {
            "bias": -0.15,  # under budget — projects completing early
            "noise": 0.10,
            "story": "Thermal redesign study completed early; reallocated $4M to OISL",
        },
        "Program Management": {
            "bias": 0.03,
            "noise": 0.04,
            "story": "Slightly over due to additional travel for vendor oversight",
        },
    }

    for ws in WORKSTREAMS:
        budget = MONTHLY_BUDGET[ws]
        pattern = patterns[ws]
        monthly_actuals = []

        for m in range(12):
            base = budget[m]
            variance = base * (pattern["bias"] + rng.normal(0, pattern["noise"]))

            # Add spike if applicable
            if "spike_month" in pattern and m == pattern["spike_month"]:
                variance += base * pattern["spike_pct"]

            actual = max(0, base + variance)
            monthly_actuals.append(round(actual, 1))

        actuals[ws] = monthly_actuals

    return actuals, patterns


ACTUALS, VARIANCE_PATTERNS = _generate_actuals()


@dataclass
class VarianceAlert:
    workstream: str
    month: str
    month_idx: int
    budget_m: float
    actual_m: float
    variance_m: float
    variance_pct: float
    severity: str        # "info", "warning", "critical"
    alert_type: str      # "over_budget", "under_budget", "trend", "cumulative"
    auto_commentary: str


def build_variance_table() -> pd.DataFrame:
    """Build full budget vs actual table."""
    records = []
    for ws in WORKSTREAMS:
        budget = MONTHLY_BUDGET[ws]
        actual = ACTUALS[ws]
        cum_budget = 0
        cum_actual = 0
        for m in range(12):
            cum_budget += budget[m]
            cum_actual += actual[m]
            var = actual[m] - budget[m]
            var_pct = (var / budget[m] * 100) if budget[m] > 0 else 0
            cum_var = cum_actual - cum_budget
            cum_var_pct = (cum_var / cum_budget * 100) if cum_budget > 0 else 0
            records.append({
                "Workstream": ws,
                "Month": MONTHS[m],
                "Month_Idx": m,
                "Budget ($M)": budget[m],
                "Actual ($M)": actual[m],
                "Variance ($M)": round(var, 1),
                "Variance (%)": round(var_pct, 1),
                "Cum Budget ($M)": round(cum_budget, 1),
                "Cum Actual ($M)": round(cum_actual, 1),
                "Cum Variance ($M)": round(cum_var, 1),
                "Cum Variance (%)": round(cum_var_pct, 1),
            })
    return pd.DataFrame(records)


def detect_anomalies(through_month: int = 6) -> list[VarianceAlert]:
    """
    Auto-detect variance anomalies through the specified month.
    Uses threshold-based detection with context-aware commentary.
    """
    alerts = []

    for ws in WORKSTREAMS:
        budget = MONTHLY_BUDGET[ws][:through_month]
        actual = ACTUALS[ws][:through_month]
        pattern = VARIANCE_PATTERNS[ws]

        cum_budget = sum(budget)
        cum_actual = sum(actual)
        cum_var = cum_actual - cum_budget
        cum_var_pct = (cum_var / cum_budget * 100) if cum_budget > 0 else 0

        # Check each month for spikes
        for m in range(through_month):
            var = actual[m] - budget[m]
            var_pct = (var / budget[m] * 100) if budget[m] > 0 else 0

            # Single-month spike detection
            if abs(var_pct) > 20:
                severity = "critical" if abs(var_pct) > 35 else "warning"
                alert_type = "over_budget" if var > 0 else "under_budget"

                commentary = _generate_commentary(ws, MONTHS[m], var, var_pct, pattern)

                alerts.append(VarianceAlert(
                    workstream=ws,
                    month=MONTHS[m],
                    month_idx=m,
                    budget_m=budget[m],
                    actual_m=actual[m],
                    variance_m=round(var, 1),
                    variance_pct=round(var_pct, 1),
                    severity=severity,
                    alert_type=alert_type,
                    auto_commentary=commentary,
                ))

        # Cumulative trend detection
        if abs(cum_var_pct) > 10:
            severity = "critical" if abs(cum_var_pct) > 20 else "warning"
            trend_dir = "over" if cum_var > 0 else "under"

            commentary = _generate_trend_commentary(ws, cum_var, cum_var_pct, through_month, pattern)

            alerts.append(VarianceAlert(
                workstream=ws,
                month=f"YTD (through {MONTHS[through_month - 1]})",
                month_idx=through_month,
                budget_m=cum_budget,
                actual_m=cum_actual,
                variance_m=round(cum_var, 1),
                variance_pct=round(cum_var_pct, 1),
                severity=severity,
                alert_type="cumulative",
                auto_commentary=commentary,
            ))

    # Sort by severity then variance magnitude
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: (severity_order.get(a.severity, 3), -abs(a.variance_pct)))

    return alerts


def _generate_commentary(ws: str, month: str, var: float, var_pct: float, pattern: dict) -> str:
    """Generate automated variance commentary for a single month."""
    direction = "over" if var > 0 else "under"

    # Use the workstream's story for context
    story = pattern.get("story", "")

    if abs(var_pct) > 35:
        urgency = "Significant"
    elif abs(var_pct) > 20:
        urgency = "Notable"
    else:
        urgency = "Minor"

    commentary = f"{urgency} {direction}-spend of ${abs(var):.1f}M ({abs(var_pct):.0f}%) in {month}."

    if direction == "over" and "accelerat" in story.lower():
        commentary += f" Driven by program-directed acceleration. {story}"
    elif direction == "over" and "risk" in story.lower():
        commentary += f" {story} — aligns with risk retirement strategy."
    elif direction == "over":
        commentary += f" {story}" if story else " Investigate root cause."
    elif direction == "under" and ("early" in story.lower() or "ahead" in story.lower()):
        commentary += f" Favorable: {story}"
    elif direction == "under":
        commentary += f" {story}" if story else " May indicate execution delay — verify."

    return commentary


def _generate_trend_commentary(ws: str, cum_var: float, cum_var_pct: float, through_month: int, pattern: dict) -> str:
    """Generate automated cumulative variance commentary."""
    direction = "over" if cum_var > 0 else "under"
    story = pattern.get("story", "")

    commentary = f"Cumulative {direction}-spend of ${abs(cum_var):.1f}M ({abs(cum_var_pct):.0f}%) through {MONTHS[through_month - 1]}."

    if direction == "over" and pattern.get("bias", 0) > 0.1:
        commentary += f" Structural overspend — {story}"
        commentary += " Recommend budget reforecast or reallocation from underspent workstreams."
    elif direction == "over":
        commentary += f" {story}" if story else " Monitor trend — may require reforecast."
    elif direction == "under" and pattern.get("bias", 0) < -0.1:
        commentary += f" Favorable: {story}"
        commentary += " Surplus available for reallocation to accelerate other workstreams."
    elif direction == "under":
        commentary += f" {story}" if story else " Verify execution is on track despite lower spend."

    return commentary


def get_ytd_summary(through_month: int = 6) -> pd.DataFrame:
    """Get YTD budget vs actual summary by workstream."""
    records = []
    for ws in WORKSTREAMS:
        budget_ytd = sum(MONTHLY_BUDGET[ws][:through_month])
        actual_ytd = sum(ACTUALS[ws][:through_month])
        var = actual_ytd - budget_ytd
        var_pct = (var / budget_ytd * 100) if budget_ytd > 0 else 0

        full_year_budget = sum(MONTHLY_BUDGET[ws])
        forecast = actual_ytd + sum(MONTHLY_BUDGET[ws][through_month:])  # actuals + remaining plan

        records.append({
            "Workstream": ws,
            "YTD Budget ($M)": round(budget_ytd, 1),
            "YTD Actual ($M)": round(actual_ytd, 1),
            "YTD Variance ($M)": round(var, 1),
            "YTD Variance (%)": round(var_pct, 1),
            "Full Year Budget ($M)": round(full_year_budget, 1),
            "Full Year Forecast ($M)": round(forecast, 1),
            "Forecast Variance ($M)": round(forecast - full_year_budget, 1),
            "Status": "🟢" if abs(var_pct) < 5 else "🟡" if abs(var_pct) < 15 else "🔴",
        })

    return pd.DataFrame(records)


def get_monthly_trend(workstream: str) -> pd.DataFrame:
    """Get monthly budget vs actual for a single workstream."""
    budget = MONTHLY_BUDGET[workstream]
    actual = ACTUALS[workstream]

    records = []
    cum_b = 0
    cum_a = 0
    for m in range(12):
        cum_b += budget[m]
        cum_a += actual[m]
        records.append({
            "Month": MONTHS[m],
            "Budget ($M)": budget[m],
            "Actual ($M)": actual[m],
            "Variance ($M)": round(actual[m] - budget[m], 1),
            "Cum Budget ($M)": round(cum_b, 1),
            "Cum Actual ($M)": round(cum_a, 1),
        })
    return pd.DataFrame(records)
