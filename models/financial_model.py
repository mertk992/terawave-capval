"""
TeraWave Financial Model — Builds deterministic and stochastic projections.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class ScenarioAssumptions:
    """User-adjustable scenario parameters."""
    name: str = "Base Case"
    capex_multiplier: float = 1.0        # 1.0 = base, 1.2 = 20% overrun
    revenue_multiplier: float = 1.0      # 1.0 = base, 0.8 = 20% lower
    timeline_shift_years: int = 0        # +1 = 1-year delay
    wacc: float = config.WACC
    opex_multiplier: float = 1.0
    # Capital deployment acceleration levers
    parallel_workstreams: int = 3        # How many workstreams run concurrently
    deployment_cadence: str = "baseline" # "baseline", "aggressive", "conservative"


def compute_capex_schedule(assumptions: ScenarioAssumptions) -> pd.DataFrame:
    """Build year-by-year CapEx schedule across all workstreams."""
    years = list(range(config.PROJECTION_YEARS + 1))
    records = []

    for item_name, item in config.CAPEX_ITEMS.items():
        # Total cost for this item
        if "total_cost_m" in item:
            total = item["total_cost_m"]
        else:
            total = item["unit_cost_m"] * item["units"]

        total *= assumptions.capex_multiplier

        # Spread across phase years
        phase = config.PHASES[item["phase"]]
        start = phase["start"] + assumptions.timeline_shift_years
        end = phase["end"] + assumptions.timeline_shift_years

        # Deployment cadence adjustment
        if assumptions.deployment_cadence == "aggressive":
            duration = max(1, int((end - start) * 0.75))
            end = start + duration
        elif assumptions.deployment_cadence == "conservative":
            duration = int((end - start) * 1.25)
            end = start + duration

        phase_years = list(range(max(0, start), min(end + 1, config.PROJECTION_YEARS + 1)))
        if not phase_years:
            continue

        # Front-load spending for aggressive deployment
        if assumptions.deployment_cadence == "aggressive":
            weights = np.array([1.3 - 0.3 * i / len(phase_years) for i in range(len(phase_years))])
        else:
            weights = np.ones(len(phase_years))
        weights /= weights.sum()

        for i, yr in enumerate(phase_years):
            records.append({
                "year": yr,
                "workstream": item_name,
                "capex_m": total * weights[i],
                "phase": item["phase"],
            })

    df = pd.DataFrame(records)
    return df


def compute_revenue_schedule(assumptions: ScenarioAssumptions) -> pd.DataFrame:
    """Build year-by-year revenue projection."""
    records = []
    for yr, rev in config.REVENUE_RAMP.items():
        adj_yr = yr + assumptions.timeline_shift_years
        if 0 <= adj_yr <= config.PROJECTION_YEARS:
            records.append({
                "year": adj_yr,
                "revenue_m": rev * assumptions.revenue_multiplier,
            })
    return pd.DataFrame(records)


def compute_opex_schedule(assumptions: ScenarioAssumptions) -> pd.DataFrame:
    """Build year-by-year OpEx schedule (starts when operations begin)."""
    ops_start = config.REVENUE_START_YEAR + assumptions.timeline_shift_years - 1
    records = []
    for yr in range(max(0, ops_start), config.PROJECTION_YEARS + 1):
        years_active = yr - ops_start
        opex = config.ANNUAL_OPEX_M * assumptions.opex_multiplier * (
            (1 + config.OPEX_GROWTH_RATE) ** years_active
        )
        records.append({"year": yr, "opex_m": opex})
    return pd.DataFrame(records)


def build_full_projection(assumptions: ScenarioAssumptions) -> pd.DataFrame:
    """Combine CapEx, OpEx, Revenue into a single year-by-year projection."""
    years = list(range(config.PROJECTION_YEARS + 1))

    capex_df = compute_capex_schedule(assumptions)
    rev_df = compute_revenue_schedule(assumptions)
    opex_df = compute_opex_schedule(assumptions)

    # Aggregate capex by year
    capex_by_year = capex_df.groupby("year")["capex_m"].sum().reindex(years, fill_value=0)
    rev_by_year = rev_df.set_index("year")["revenue_m"].reindex(years, fill_value=0) if len(rev_df) else pd.Series(0, index=years)
    opex_by_year = opex_df.set_index("year")["opex_m"].reindex(years, fill_value=0) if len(opex_df) else pd.Series(0, index=years)

    projection = pd.DataFrame({
        "year": years,
        "calendar_year": [2025 + yr for yr in years],
        "capex_m": capex_by_year.values,
        "opex_m": opex_by_year.values,
        "revenue_m": rev_by_year.values,
    })

    projection["free_cash_flow_m"] = (
        projection["revenue_m"] - projection["opex_m"] - projection["capex_m"]
    )
    projection["cumulative_investment_m"] = (
        projection["capex_m"] + projection["opex_m"]
    ).cumsum()
    projection["cumulative_fcf_m"] = projection["free_cash_flow_m"].cumsum()

    # Discounted cash flows
    discount_factors = np.array([(1 + assumptions.wacc) ** (-yr) for yr in years])
    projection["dcf_m"] = projection["free_cash_flow_m"] * discount_factors
    projection["cumulative_dcf_m"] = projection["dcf_m"].cumsum()

    return projection


def compute_npv(projection: pd.DataFrame, assumptions: ScenarioAssumptions) -> dict:
    """Compute NPV, IRR, payback period, and other key metrics."""
    fcf = projection["free_cash_flow_m"].values
    years = projection["year"].values

    # NPV
    npv = projection["dcf_m"].sum()

    # Terminal value (Gordon Growth on last year FCF)
    terminal_fcf = fcf[-1] * (1 + config.TERMINAL_GROWTH)
    terminal_value = terminal_fcf / (assumptions.wacc - config.TERMINAL_GROWTH)
    terminal_pv = terminal_value / (1 + assumptions.wacc) ** years[-1]
    total_npv = npv + terminal_pv

    # IRR (numerical approximation)
    irr = _compute_irr(fcf)

    # Payback period
    cumulative = np.cumsum(fcf)
    payback_idx = np.where(cumulative > 0)[0]
    payback_year = int(years[payback_idx[0]]) if len(payback_idx) > 0 else None

    # Peak capital deployed
    peak_investment = projection["cumulative_investment_m"].max()

    # Total capex
    total_capex = projection["capex_m"].sum()

    return {
        "npv_m": round(total_npv, 1),
        "npv_excl_terminal_m": round(npv, 1),
        "terminal_value_pv_m": round(terminal_pv, 1),
        "irr_pct": round(irr * 100, 1) if irr else None,
        "payback_year": payback_year,
        "payback_calendar_year": 2025 + payback_year if payback_year else None,
        "peak_investment_m": round(peak_investment, 1),
        "total_capex_m": round(total_capex, 1),
    }


def _compute_irr(cashflows: np.ndarray, tol=1e-6, max_iter=1000) -> Optional[float]:
    """Newton-Raphson IRR solver."""
    rate = 0.10
    for _ in range(max_iter):
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
        dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-12:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate
        if rate < -0.5 or rate > 5.0:
            return None
    return rate if -0.5 < rate < 5.0 else None


def compute_progress_metrics(assumptions: ScenarioAssumptions) -> pd.DataFrame:
    """
    Compute the 'progress per dollar' and 'risk retired per dollar' for each workstream.
    This is the unique corporate finance lens: not 'minimize cost' but 'maximize progress
    and risk retirement per unit of capital deployed.'
    """
    capex_df = compute_capex_schedule(assumptions)
    records = []

    for ws_name, ws in config.WORKSTREAM_WEIGHTS.items():
        ws_capex = capex_df[capex_df["workstream"] == ws_name]["capex_m"].sum()
        if ws_capex == 0:
            continue

        progress_per_dollar = ws["progress_weight"] / (ws_capex / 1000)  # per $B
        risk_retired_per_dollar = ws["risk_retirement_weight"] / (ws_capex / 1000)
        composite_score = 0.5 * progress_per_dollar + 0.5 * risk_retired_per_dollar

        records.append({
            "workstream": ws_name,
            "description": ws["description"],
            "total_capex_m": round(ws_capex, 1),
            "progress_weight": ws["progress_weight"],
            "risk_retirement_weight": ws["risk_retirement_weight"],
            "progress_per_bn": round(progress_per_dollar, 3),
            "risk_retired_per_bn": round(risk_retired_per_dollar, 3),
            "composite_score": round(composite_score, 3),
        })

    df = pd.DataFrame(records).sort_values("composite_score", ascending=False)
    return df
