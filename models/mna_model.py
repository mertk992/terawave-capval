"""
Strategic M&A / Make-vs-Buy Analysis Model

Evaluates build-vs-buy-vs-partner decisions for aerospace supply chain components.
Uses Blue Origin's vertical integration strategy as context.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class TargetProfile:
    name: str
    description: str
    category: str  # "component", "technology", "service"
    acquisition_price_m: float
    annual_revenue_m: float
    annual_ebitda_m: float
    employees: int
    patents: int
    key_capabilities: list[str] = field(default_factory=list)
    strategic_fit: float = 0.0       # 0-1 score
    integration_complexity: float = 0.0  # 0-1 score
    ip_overlap_pct: float = 0.0


@dataclass
class MakeVsBuyScenario:
    """Parameters for a make-vs-buy evaluation."""
    target: TargetProfile
    # Build internally
    internal_dev_cost_m: float = 0.0
    internal_dev_years: float = 0.0
    internal_annual_opex_m: float = 0.0
    internal_risk_score: float = 0.0  # 0-1
    # Partner/license
    partner_annual_cost_m: float = 0.0
    partner_contract_years: int = 5
    partner_dependency_risk: float = 0.0  # 0-1
    # Synergies from acquisition
    revenue_synergy_m: float = 0.0
    cost_synergy_m: float = 0.0
    integration_cost_m: float = 0.0
    integration_months: int = 18


# ── Synthetic Target Companies ───────────────────────────────────────────────

SYNTHETIC_TARGETS = [
    TargetProfile(
        name="Photon Dynamics",
        description="Optical inter-satellite link (OISL) specialist. Leading tech for free-space laser comms.",
        category="technology",
        acquisition_price_m=2_800,
        annual_revenue_m=420,
        annual_ebitda_m=85,
        employees=650,
        patents=47,
        key_capabilities=["Free-space optical comms", "Pointing & tracking systems", "Space-qualified transceivers"],
        strategic_fit=0.92,
        integration_complexity=0.45,
        ip_overlap_pct=0.15,
    ),
    TargetProfile(
        name="Orbital Antenna Systems",
        description="Q/V-band phased array manufacturer. Supplies antennas for LEO constellations.",
        category="component",
        acquisition_price_m=1_200,
        annual_revenue_m=280,
        annual_ebitda_m=52,
        employees=380,
        patents=23,
        key_capabilities=["Q/V-band arrays", "Beam-forming ASICs", "High-volume sat antenna production"],
        strategic_fit=0.85,
        integration_complexity=0.35,
        ip_overlap_pct=0.08,
    ),
    TargetProfile(
        name="TerraLink Networks",
        description="Ground station network operator with 40+ global gateway sites.",
        category="service",
        acquisition_price_m=3_500,
        annual_revenue_m=600,
        annual_ebitda_m=180,
        employees=1_200,
        patents=12,
        key_capabilities=["Global gateway network", "Carrier-grade NOC", "Enterprise SLA management"],
        strategic_fit=0.78,
        integration_complexity=0.60,
        ip_overlap_pct=0.05,
    ),
]

SYNTHETIC_SCENARIOS = {
    "Photon Dynamics": MakeVsBuyScenario(
        target=SYNTHETIC_TARGETS[0],
        internal_dev_cost_m=1_800,
        internal_dev_years=4.5,
        internal_annual_opex_m=120,
        internal_risk_score=0.65,
        partner_annual_cost_m=200,
        partner_contract_years=7,
        partner_dependency_risk=0.70,
        revenue_synergy_m=150,
        cost_synergy_m=80,
        integration_cost_m=350,
        integration_months=24,
    ),
    "Orbital Antenna Systems": MakeVsBuyScenario(
        target=SYNTHETIC_TARGETS[1],
        internal_dev_cost_m=900,
        internal_dev_years=3.0,
        internal_annual_opex_m=75,
        internal_risk_score=0.40,
        partner_annual_cost_m=140,
        partner_contract_years=5,
        partner_dependency_risk=0.50,
        revenue_synergy_m=60,
        cost_synergy_m=45,
        integration_cost_m=150,
        integration_months=12,
    ),
    "TerraLink Networks": MakeVsBuyScenario(
        target=SYNTHETIC_TARGETS[2],
        internal_dev_cost_m=4_200,
        internal_dev_years=6.0,
        internal_annual_opex_m=250,
        internal_risk_score=0.55,
        partner_annual_cost_m=300,
        partner_contract_years=10,
        partner_dependency_risk=0.45,
        revenue_synergy_m=200,
        cost_synergy_m=120,
        integration_cost_m=500,
        integration_months=30,
    ),
}


def evaluate_make_vs_buy(scenario: MakeVsBuyScenario, wacc: float = 0.12, horizon: int = 10) -> dict:
    """
    Evaluate build vs. buy vs. partner across financial and strategic dimensions.
    Returns a comparison dict with NPV and scoring for each option.
    """
    t = scenario.target
    discount = np.array([(1 + wacc) ** -i for i in range(horizon + 1)])

    # ── Option 1: BUILD INTERNALLY ──
    build_capex = np.zeros(horizon + 1)
    build_years = int(np.ceil(scenario.internal_dev_years))
    for yr in range(min(build_years, horizon + 1)):
        build_capex[yr] = scenario.internal_dev_cost_m / build_years
    build_opex = np.zeros(horizon + 1)
    for yr in range(build_years, horizon + 1):
        build_opex[yr] = scenario.internal_annual_opex_m
    build_total_cost = (build_capex + build_opex) * discount
    build_npv_cost = build_total_cost.sum()
    build_time_to_capability = scenario.internal_dev_years

    # ── Option 2: ACQUIRE ──
    acquire_upfront = t.acquisition_price_m + scenario.integration_cost_m
    acquire_synergies = np.zeros(horizon + 1)
    integration_years = scenario.integration_months / 12
    for yr in range(horizon + 1):
        if yr == 0:
            continue
        if yr < integration_years:
            ramp = yr / integration_years
        else:
            ramp = 1.0
        acquire_synergies[yr] = (scenario.revenue_synergy_m + scenario.cost_synergy_m) * ramp
    acquire_npv = -acquire_upfront + (acquire_synergies * discount).sum()
    acquire_time_to_capability = integration_years

    # Accretion/dilution (simplified — based on EBITDA multiple)
    ev_ebitda = t.acquisition_price_m / t.annual_ebitda_m if t.annual_ebitda_m > 0 else 0
    year1_synergy = (scenario.revenue_synergy_m + scenario.cost_synergy_m) * min(1, 12 / scenario.integration_months)
    accretive_year1 = (t.annual_ebitda_m + year1_synergy) > (acquire_upfront * wacc)

    # ── Option 3: PARTNER / LICENSE ──
    partner_cost = np.zeros(horizon + 1)
    for yr in range(min(scenario.partner_contract_years, horizon + 1)):
        partner_cost[yr] = scenario.partner_annual_cost_m
    # After contract ends, must renew at higher cost or build
    for yr in range(scenario.partner_contract_years, horizon + 1):
        partner_cost[yr] = scenario.partner_annual_cost_m * 1.15
    partner_npv_cost = (partner_cost * discount).sum()
    partner_time_to_capability = 0.5  # fastest to start

    # ── Composite Scoring ──
    def _score_option(npv_cost, time_yrs, risk, dependency=0):
        """Lower is better for cost/time/risk; normalize to 0-100 score."""
        cost_score = max(0, 100 - npv_cost / 50)  # rough normalization
        time_score = max(0, 100 - time_yrs * 15)
        risk_score = (1 - risk) * 100
        dep_score = (1 - dependency) * 100
        return {
            "cost_score": round(cost_score, 1),
            "time_score": round(time_score, 1),
            "risk_score": round(risk_score, 1),
            "dependency_score": round(dep_score, 1),
            "composite": round(cost_score * 0.30 + time_score * 0.25 + risk_score * 0.25 + dep_score * 0.20, 1),
        }

    build_scores = _score_option(build_npv_cost, build_time_to_capability, scenario.internal_risk_score, 0)
    acquire_scores = _score_option(abs(acquire_npv) if acquire_npv < 0 else 0, acquire_time_to_capability, t.integration_complexity, 0)
    partner_scores = _score_option(partner_npv_cost, partner_time_to_capability, 0.2, scenario.partner_dependency_risk)

    # Determine recommendation
    options = {"Build": build_scores["composite"], "Acquire": acquire_scores["composite"], "Partner": partner_scores["composite"]}
    recommended = max(options, key=options.get)

    return {
        "target": t.name,
        "recommended_option": recommended,
        "build": {
            "npv_cost_m": round(build_npv_cost, 1),
            "time_to_capability_yrs": build_time_to_capability,
            "scores": build_scores,
        },
        "acquire": {
            "upfront_cost_m": round(acquire_upfront, 1),
            "npv_synergies_m": round((acquire_synergies * discount).sum(), 1),
            "net_npv_m": round(acquire_npv, 1),
            "ev_ebitda_multiple": round(ev_ebitda, 1),
            "accretive_year1": accretive_year1,
            "time_to_capability_yrs": round(acquire_time_to_capability, 1),
            "scores": acquire_scores,
        },
        "partner": {
            "npv_cost_m": round(partner_npv_cost, 1),
            "annual_cost_m": scenario.partner_annual_cost_m,
            "dependency_risk": scenario.partner_dependency_risk,
            "time_to_capability_yrs": partner_time_to_capability,
            "scores": partner_scores,
        },
    }


def build_comparison_table(scenario_name: str, wacc: float = 0.12) -> pd.DataFrame:
    """Build a formatted comparison table for display."""
    scenario = SYNTHETIC_SCENARIOS[scenario_name]
    result = evaluate_make_vs_buy(scenario, wacc)

    rows = []
    for option in ["build", "acquire", "partner"]:
        data = result[option]
        scores = data["scores"]
        rows.append({
            "Option": option.title(),
            "Total Cost (NPV $M)": data.get("npv_cost_m", abs(data.get("net_npv_m", 0))),
            "Time to Capability (yrs)": data.get("time_to_capability_yrs", "—"),
            "Cost Score": scores["cost_score"],
            "Time Score": scores["time_score"],
            "Risk Score": scores["risk_score"],
            "Dependency Score": scores["dependency_score"],
            "Composite Score": scores["composite"],
            "Recommended": "✓" if option.title() == result["recommended_option"] else "",
        })

    return pd.DataFrame(rows)
