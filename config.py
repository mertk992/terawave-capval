"""
TeraWave Capital Project Valuation Engine — Configuration

⚠️ ALL FINANCIAL DATA IS SYNTHETIC AND FOR DEMONSTRATION PURPOSES ONLY.
This tool was built as a technical demo and does not reflect actual Blue Origin financials.
"""

import os

try:
    import streamlit as st
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
except Exception:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"

# ── TeraWave Program Parameters (SYNTHETIC) ─────────────────────────────────
# Based on publicly available architectural details; all costs are illustrative.

PROGRAM_NAME = "TeraWave Satellite Constellation"
PROGRAM_DESCRIPTION = (
    "5,408-satellite LEO/MEO constellation delivering 6 Tbps symmetrical capacity "
    "to ~100,000 enterprise, data center, and government sites worldwide."
)

# Constellation architecture
LEO_SATELLITES = 5_280
MEO_SATELLITES = 128
TOTAL_SATELLITES = LEO_SATELLITES + MEO_SATELLITES

# Timeline (years from program start, Year 0 = 2025)
PHASES = {
    "Phase 0 — Design & Prototyping":     {"start": 0, "end": 2,  "label": "Design"},
    "Phase 1 — Initial Deployment":        {"start": 2, "end": 4,  "label": "Initial"},
    "Phase 2 — Constellation Build-Out":   {"start": 4, "end": 7,  "label": "Build-Out"},
    "Phase 3 — Full Operational Capacity": {"start": 7, "end": 10, "label": "FOC"},
}

# ── Synthetic Cost Structure ($M) ────────────────────────────────────────────
# These are order-of-magnitude estimates for a program of this scale.

CAPEX_ITEMS = {
    "Satellite Manufacturing (LEO)": {
        "unit_cost_m": 0.8,       # $M per satellite
        "units": LEO_SATELLITES,
        "phase": "Phase 2 — Constellation Build-Out",
        "uncertainty_pct": 0.20,  # ±20%
    },
    "Satellite Manufacturing (MEO)": {
        "unit_cost_m": 25.0,
        "units": MEO_SATELLITES,
        "phase": "Phase 2 — Constellation Build-Out",
        "uncertainty_pct": 0.25,
    },
    "Launch Services": {
        "unit_cost_m": 30.0,      # per launch (New Glenn)
        "units": 90,              # ~60 sats per launch
        "phase": "Phase 1 — Initial Deployment",
        "uncertainty_pct": 0.15,
    },
    "Ground Segment & Gateways": {
        "total_cost_m": 2_500.0,
        "phase": "Phase 1 — Initial Deployment",
        "uncertainty_pct": 0.25,
    },
    "Optical Inter-Satellite Links": {
        "total_cost_m": 1_200.0,
        "phase": "Phase 0 — Design & Prototyping",
        "uncertainty_pct": 0.30,
    },
    "Software & Network Operations": {
        "total_cost_m": 800.0,
        "phase": "Phase 1 — Initial Deployment",
        "uncertainty_pct": 0.20,
    },
    "Regulatory & Spectrum Licensing": {
        "total_cost_m": 300.0,
        "phase": "Phase 0 — Design & Prototyping",
        "uncertainty_pct": 0.10,
    },
    "R&D and Prototyping": {
        "total_cost_m": 1_500.0,
        "phase": "Phase 0 — Design & Prototyping",
        "uncertainty_pct": 0.20,
    },
}

# Annual OpEx once operational ($M/year)
ANNUAL_OPEX_M = 600.0
OPEX_GROWTH_RATE = 0.03  # 3% annual growth

# ── Revenue Assumptions (SYNTHETIC) ──────────────────────────────────────────
REVENUE_START_YEAR = 4  # First revenue in Year 4 (partial)
REVENUE_RAMP = {
    4: 200,     # Early adopter contracts
    5: 800,
    6: 2_000,
    7: 3_500,
    8: 5_000,
    9: 6_500,
    10: 8_000,
    11: 9_000,
    12: 10_000,
}
REVENUE_UNCERTAINTY_PCT = 0.30

# ── Valuation Parameters ─────────────────────────────────────────────────────
WACC = 0.12              # 12% discount rate
TERMINAL_GROWTH = 0.03   # 3% terminal growth
PROJECTION_YEARS = 12

# ── Progress & Risk Metrics (unique to this framework) ───────────────────────
# Each workstream has a "progress weight" — how much it advances the program
# and a "risk retirement weight" — how much technical/market risk it retires.

WORKSTREAM_WEIGHTS = {
    "Satellite Manufacturing (LEO)": {
        "progress_weight": 0.30,
        "risk_retirement_weight": 0.15,
        "description": "Mass production of LEO bus & payload",
    },
    "Satellite Manufacturing (MEO)": {
        "progress_weight": 0.15,
        "risk_retirement_weight": 0.25,
        "description": "High-capacity optical MEO satellites — highest technical risk",
    },
    "Launch Services": {
        "progress_weight": 0.20,
        "risk_retirement_weight": 0.10,
        "description": "New Glenn launch cadence for constellation deployment",
    },
    "Ground Segment & Gateways": {
        "progress_weight": 0.15,
        "risk_retirement_weight": 0.20,
        "description": "Global gateway network for enterprise connectivity",
    },
    "Optical Inter-Satellite Links": {
        "progress_weight": 0.10,
        "risk_retirement_weight": 0.20,
        "description": "Cross-link technology enabling mesh networking",
    },
    "Software & Network Operations": {
        "progress_weight": 0.10,
        "risk_retirement_weight": 0.10,
        "description": "Network management, routing, and customer provisioning",
    },
}

# Monte Carlo
MC_SIMULATIONS = 5_000
