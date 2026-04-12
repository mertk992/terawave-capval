"""
Monte Carlo Simulation Engine for TeraWave Capital Valuation.

Runs thousands of simulations with correlated uncertainty across cost,
revenue, and timeline dimensions to produce probabilistic outcomes.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

import config
from models.financial_model import (
    ScenarioAssumptions,
    build_full_projection,
    compute_npv,
    compute_progress_metrics,
)


@dataclass
class MonteCarloResults:
    npv_distribution: np.ndarray
    irr_distribution: np.ndarray
    payback_distribution: np.ndarray
    peak_investment_distribution: np.ndarray
    total_capex_distribution: np.ndarray
    scenario_details: list
    percentiles: dict

    @property
    def summary(self) -> dict:
        return {
            "npv_p10_m": round(float(np.percentile(self.npv_distribution, 10)), 1),
            "npv_p50_m": round(float(np.percentile(self.npv_distribution, 50)), 1),
            "npv_p90_m": round(float(np.percentile(self.npv_distribution, 90)), 1),
            "npv_mean_m": round(float(np.mean(self.npv_distribution)), 1),
            "irr_p10_pct": round(float(np.percentile(self.irr_distribution[self.irr_distribution > -50], 10)), 1),
            "irr_p50_pct": round(float(np.percentile(self.irr_distribution[self.irr_distribution > -50], 50)), 1),
            "irr_p90_pct": round(float(np.percentile(self.irr_distribution[self.irr_distribution > -50], 90)), 1),
            "prob_positive_npv": round(float(np.mean(self.npv_distribution > 0) * 100), 1),
            "prob_irr_above_wacc": round(float(np.mean(self.irr_distribution > config.WACC * 100) * 100), 1),
            "peak_investment_p50_m": round(float(np.percentile(self.peak_investment_distribution, 50)), 1),
            "peak_investment_p90_m": round(float(np.percentile(self.peak_investment_distribution, 90)), 1),
        }


def run_monte_carlo(
    base_assumptions: ScenarioAssumptions,
    n_simulations: int = config.MC_SIMULATIONS,
    seed: int = 42,
) -> MonteCarloResults:
    """
    Run Monte Carlo simulation varying capex, revenue, and timeline.

    Uses correlated sampling: cost overruns are positively correlated with
    timeline delays, and negatively correlated with revenue (worse execution
    → higher costs, longer timeline, slower revenue ramp).
    """
    rng = np.random.default_rng(seed)

    # Correlation structure
    # [capex_shock, revenue_shock, timeline_shock]
    corr_matrix = np.array([
        [1.0,  -0.3,  0.5],   # capex ↔ revenue (negative), capex ↔ timeline (positive)
        [-0.3,  1.0, -0.4],   # revenue ↔ timeline (negative)
        [0.5,  -0.4,  1.0],
    ])
    L = np.linalg.cholesky(corr_matrix)

    npv_dist = []
    irr_dist = []
    payback_dist = []
    peak_inv_dist = []
    total_capex_dist = []
    details = []

    for i in range(n_simulations):
        # Generate correlated shocks
        z = rng.standard_normal(3)
        correlated = L @ z

        capex_shock = 1.0 + correlated[0] * 0.20      # ±20% std dev
        revenue_shock = 1.0 + correlated[1] * 0.25     # ±25% std dev
        timeline_shock = correlated[2]                   # continuous

        # Discretize timeline shift
        timeline_shift = 0
        if timeline_shock > 1.0:
            timeline_shift = 2
        elif timeline_shock > 0.5:
            timeline_shift = 1
        elif timeline_shock < -0.5:
            timeline_shift = -1

        # Clamp multipliers
        capex_mult = max(0.7, min(1.6, capex_shock * base_assumptions.capex_multiplier))
        rev_mult = max(0.4, min(1.5, revenue_shock * base_assumptions.revenue_multiplier))

        sim_assumptions = ScenarioAssumptions(
            name=f"MC_{i}",
            capex_multiplier=capex_mult,
            revenue_multiplier=rev_mult,
            timeline_shift_years=timeline_shift + base_assumptions.timeline_shift_years,
            wacc=base_assumptions.wacc,
            opex_multiplier=base_assumptions.opex_multiplier * (1 + (capex_shock - 1) * 0.3),
            deployment_cadence=base_assumptions.deployment_cadence,
        )

        projection = build_full_projection(sim_assumptions)
        metrics = compute_npv(projection, sim_assumptions)

        npv_dist.append(metrics["npv_m"])
        irr_dist.append(metrics["irr_pct"] if metrics["irr_pct"] is not None else -50)
        payback_dist.append(metrics["payback_year"] if metrics["payback_year"] else 15)
        peak_inv_dist.append(metrics["peak_investment_m"])
        total_capex_dist.append(metrics["total_capex_m"])

        if i < 20:  # Store first 20 for detail display
            details.append({
                "sim": i,
                "capex_mult": round(capex_mult, 2),
                "rev_mult": round(rev_mult, 2),
                "timeline_shift": timeline_shift,
                "npv_m": metrics["npv_m"],
                "irr_pct": metrics["irr_pct"],
                "payback_year": metrics["payback_year"],
            })

    npv_arr = np.array(npv_dist)
    irr_arr = np.array(irr_dist)
    payback_arr = np.array(payback_dist)
    peak_arr = np.array(peak_inv_dist)
    capex_arr = np.array(total_capex_dist)

    percentiles = {
        "npv": {p: float(np.percentile(npv_arr, p)) for p in [5, 10, 25, 50, 75, 90, 95]},
        "irr": {p: float(np.percentile(irr_arr[irr_arr > -50], p)) for p in [5, 10, 25, 50, 75, 90, 95]},
        "payback": {p: float(np.percentile(payback_arr, p)) for p in [5, 10, 25, 50, 75, 90, 95]},
    }

    return MonteCarloResults(
        npv_distribution=npv_arr,
        irr_distribution=irr_arr,
        payback_distribution=payback_arr,
        peak_investment_distribution=peak_arr,
        total_capex_distribution=capex_arr,
        scenario_details=details,
        percentiles=percentiles,
    )


def sensitivity_analysis(
    base_assumptions: ScenarioAssumptions,
    variable: str,
    range_pct: float = 0.30,
    steps: int = 11,
) -> pd.DataFrame:
    """
    One-at-a-time sensitivity: vary a single input and track NPV.
    """
    multipliers = np.linspace(1 - range_pct, 1 + range_pct, steps)
    results = []

    for mult in multipliers:
        sim = ScenarioAssumptions(
            name=f"Sensitivity_{variable}_{mult:.2f}",
            capex_multiplier=base_assumptions.capex_multiplier * (mult if variable == "capex" else 1.0),
            revenue_multiplier=base_assumptions.revenue_multiplier * (mult if variable == "revenue" else 1.0),
            timeline_shift_years=base_assumptions.timeline_shift_years + (int(round((mult - 1) * 5)) if variable == "timeline" else 0),
            wacc=base_assumptions.wacc * (mult if variable == "wacc" else 1.0),
            opex_multiplier=base_assumptions.opex_multiplier * (mult if variable == "opex" else 1.0),
            deployment_cadence=base_assumptions.deployment_cadence,
        )
        projection = build_full_projection(sim)
        metrics = compute_npv(projection, sim)
        results.append({
            "variable": variable,
            "multiplier": round(mult, 2),
            "pct_change": round((mult - 1) * 100, 1),
            "npv_m": metrics["npv_m"],
            "irr_pct": metrics["irr_pct"],
        })

    return pd.DataFrame(results)


def tornado_analysis(base_assumptions: ScenarioAssumptions) -> pd.DataFrame:
    """Run sensitivity on all key variables and return tornado-chart data."""
    variables = ["capex", "revenue", "wacc", "opex"]
    records = []

    # Base case
    base_proj = build_full_projection(base_assumptions)
    base_metrics = compute_npv(base_proj, base_assumptions)
    base_npv = base_metrics["npv_m"]

    for var in variables:
        df = sensitivity_analysis(base_assumptions, var, range_pct=0.25, steps=3)
        low_npv = df.iloc[0]["npv_m"]
        high_npv = df.iloc[-1]["npv_m"]
        records.append({
            "variable": var.title(),
            "low_npv_m": low_npv,
            "high_npv_m": high_npv,
            "range_m": abs(high_npv - low_npv),
            "base_npv_m": base_npv,
        })

    return pd.DataFrame(records).sort_values("range_m", ascending=True)
