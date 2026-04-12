"""
Scenario Planning & Strategic Finance Simulator

Long-range financial planning under deep uncertainty across multiple
program bets. Evaluates portfolio-level capital allocation.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ProgramBet:
    """A major program/investment bet in the portfolio."""
    name: str
    description: str
    total_capex_m: float
    peak_annual_capex_m: float
    revenue_start_year: int
    steady_state_revenue_m: float
    probability_of_success: float  # 0-1
    strategic_priority: float       # 0-1
    risk_category: str              # "technical", "market", "regulatory", "execution"
    years_to_revenue: int = 5


@dataclass
class MacroScenario:
    """A macro environment scenario affecting all programs."""
    name: str
    description: str
    demand_multiplier: float = 1.0
    regulatory_delay_years: int = 0
    cost_inflation_pct: float = 0.0
    funding_availability: float = 1.0  # 1.0 = full access to capital
    probability: float = 0.25  # scenario probability weight


# ── Synthetic Program Portfolio ──────────────────────────────────────────────

PROGRAMS = [
    ProgramBet(
        name="TeraWave Constellation",
        description="5,408-sat LEO/MEO constellation for enterprise/govt connectivity",
        total_capex_m=16_000,
        peak_annual_capex_m=3_500,
        revenue_start_year=4,
        steady_state_revenue_m=8_000,
        probability_of_success=0.70,
        strategic_priority=0.95,
        risk_category="technical",
        years_to_revenue=4,
    ),
    ProgramBet(
        name="New Glenn Launch Services",
        description="Heavy-lift launch vehicle for commercial and government payloads",
        total_capex_m=5_000,
        peak_annual_capex_m=1_200,
        revenue_start_year=2,
        steady_state_revenue_m=3_000,
        probability_of_success=0.85,
        strategic_priority=0.90,
        risk_category="execution",
        years_to_revenue=2,
    ),
    ProgramBet(
        name="Orbital Reef Station",
        description="Commercial space station for research and tourism",
        total_capex_m=8_000,
        peak_annual_capex_m=1_800,
        revenue_start_year=6,
        steady_state_revenue_m=2_500,
        probability_of_success=0.55,
        strategic_priority=0.65,
        risk_category="market",
        years_to_revenue=6,
    ),
    ProgramBet(
        name="Blue Moon Lander",
        description="Lunar cargo and crew lander for NASA Artemis program",
        total_capex_m=4_500,
        peak_annual_capex_m=1_000,
        revenue_start_year=3,
        steady_state_revenue_m=2_000,
        probability_of_success=0.75,
        strategic_priority=0.80,
        risk_category="execution",
        years_to_revenue=3,
    ),
]

MACRO_SCENARIOS = [
    MacroScenario(
        name="Bull Case",
        description="Strong demand, favorable regulation, low inflation",
        demand_multiplier=1.3,
        regulatory_delay_years=0,
        cost_inflation_pct=-0.05,
        funding_availability=1.2,
        probability=0.20,
    ),
    MacroScenario(
        name="Base Case",
        description="Moderate growth, normal regulatory timeline",
        demand_multiplier=1.0,
        regulatory_delay_years=0,
        cost_inflation_pct=0.0,
        funding_availability=1.0,
        probability=0.45,
    ),
    MacroScenario(
        name="Bear Case",
        description="Weak demand, regulatory delays, cost pressures",
        demand_multiplier=0.7,
        regulatory_delay_years=2,
        cost_inflation_pct=0.15,
        funding_availability=0.8,
        probability=0.25,
    ),
    MacroScenario(
        name="Stress Case",
        description="Recession, funding constraints, major delays",
        demand_multiplier=0.5,
        regulatory_delay_years=3,
        cost_inflation_pct=0.25,
        funding_availability=0.5,
        probability=0.10,
    ),
]


def build_program_projection(
    program: ProgramBet,
    scenario: MacroScenario,
    horizon: int = 12,
    wacc: float = 0.12,
) -> pd.DataFrame:
    """Build year-by-year projection for a single program under a scenario."""
    years = list(range(horizon + 1))

    # CapEx schedule — bell curve peaking mid-program
    capex = np.zeros(horizon + 1)
    build_years = program.years_to_revenue + scenario.regulatory_delay_years
    for yr in range(min(build_years + 2, horizon + 1)):
        # Bell-shaped spending
        peak_yr = build_years / 2
        capex[yr] = program.peak_annual_capex_m * np.exp(-0.5 * ((yr - peak_yr) / (build_years / 3)) ** 2)
    # Apply cost inflation
    capex *= (1 + scenario.cost_inflation_pct)
    # Scale to approximately match total capex
    if capex.sum() > 0:
        capex = capex / capex.sum() * program.total_capex_m * (1 + scenario.cost_inflation_pct)

    # Revenue schedule
    revenue = np.zeros(horizon + 1)
    rev_start = program.revenue_start_year + scenario.regulatory_delay_years
    for yr in range(horizon + 1):
        if yr < rev_start:
            continue
        ramp = min(1.0, (yr - rev_start + 1) / 3)  # 3-year ramp
        revenue[yr] = program.steady_state_revenue_m * ramp * scenario.demand_multiplier

    # Apply probability of success as expected value
    revenue_expected = revenue * program.probability_of_success

    # FCF
    fcf = revenue_expected - capex
    discount = np.array([(1 + wacc) ** -yr for yr in years])

    df = pd.DataFrame({
        "year": years,
        "calendar_year": [2025 + yr for yr in years],
        "program": program.name,
        "scenario": scenario.name,
        "capex_m": np.round(capex, 1),
        "revenue_m": np.round(revenue, 1),
        "expected_revenue_m": np.round(revenue_expected, 1),
        "fcf_m": np.round(fcf, 1),
        "dcf_m": np.round(fcf * discount, 1),
    })
    return df


def build_portfolio_projection(
    scenario: MacroScenario,
    programs: list[ProgramBet] = None,
    horizon: int = 12,
    wacc: float = 0.12,
) -> pd.DataFrame:
    """Build combined portfolio projection under a single scenario."""
    if programs is None:
        programs = PROGRAMS

    dfs = []
    for prog in programs:
        dfs.append(build_program_projection(prog, scenario, horizon, wacc))

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def portfolio_summary(
    programs: list[ProgramBet] = None,
    scenarios: list[MacroScenario] = None,
    horizon: int = 12,
    wacc: float = 0.12,
) -> pd.DataFrame:
    """Summary table: each program × each scenario → NPV."""
    if programs is None:
        programs = PROGRAMS
    if scenarios is None:
        scenarios = MACRO_SCENARIOS

    records = []
    for prog in programs:
        for scen in scenarios:
            proj = build_program_projection(prog, scen, horizon, wacc)
            npv = proj["dcf_m"].sum()
            total_capex = proj["capex_m"].sum()
            total_rev = proj["expected_revenue_m"].sum()
            records.append({
                "Program": prog.name,
                "Scenario": scen.name,
                "NPV ($M)": round(npv, 0),
                "Total CapEx ($M)": round(total_capex, 0),
                "Total Expected Revenue ($M)": round(total_rev, 0),
                "Prob of Success": f"{prog.probability_of_success:.0%}",
                "Scenario Prob": f"{scen.probability:.0%}",
                "Prob-Weighted NPV ($M)": round(npv * scen.probability, 0),
            })

    return pd.DataFrame(records)


def optimal_allocation(
    budget_m: float = 30_000,
    programs: list[ProgramBet] = None,
    scenarios: list[MacroScenario] = None,
    wacc: float = 0.12,
) -> pd.DataFrame:
    """
    Simple portfolio optimization: rank programs by probability-weighted NPV
    per dollar of capex, then allocate budget greedily.
    """
    if programs is None:
        programs = PROGRAMS
    if scenarios is None:
        scenarios = MACRO_SCENARIOS

    records = []
    for prog in programs:
        weighted_npv = 0
        for scen in scenarios:
            proj = build_program_projection(prog, scen, wacc=wacc)
            npv = proj["dcf_m"].sum()
            weighted_npv += npv * scen.probability

        efficiency = weighted_npv / prog.total_capex_m if prog.total_capex_m > 0 else 0

        records.append({
            "Program": prog.name,
            "Total CapEx ($M)": prog.total_capex_m,
            "Prob-Weighted NPV ($M)": round(weighted_npv, 0),
            "NPV / CapEx Ratio": round(efficiency, 2),
            "Strategic Priority": prog.strategic_priority,
            "Risk Category": prog.risk_category,
            "Combined Score": round(efficiency * 0.6 + prog.strategic_priority * 0.4, 3),
        })

    df = pd.DataFrame(records).sort_values("Combined Score", ascending=False)

    # Greedy allocation
    remaining = budget_m
    allocations = []
    for _, row in df.iterrows():
        needed = row["Total CapEx ($M)"]
        allocated = min(needed, remaining)
        pct = allocated / needed * 100 if needed > 0 else 0
        allocations.append(f"{pct:.0f}%")
        remaining -= allocated

    df["Funding Allocation"] = allocations
    return df
