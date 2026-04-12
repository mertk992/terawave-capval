# TeraWave Capital Valuation Engine

An agentic AI system for capital allocation analysis, built around Blue Origin's TeraWave satellite constellation program.

> **⚠️ All financial data is synthetic and for demonstration purposes only.** This tool does not represent actual Blue Origin financials, projections, or internal data.

## What It Does

This is a multi-agent capital project valuation engine that goes beyond traditional DCF analysis by incorporating a **"progress per dollar" and "risk retired per dollar"** framework — designed for capital-intensive programs where the goal is to **accelerate deployment**, not suppress costs.

### Core Components

1. **Financial Model** — 12-year DCF projection with CapEx by workstream, OpEx ramp, revenue schedule, IRR, NPV, and payback analysis
2. **Capital Efficiency Framework** — Maps each workstream by progress contribution and risk retirement value per dollar deployed
3. **Monte Carlo Simulation** — 2,500+ correlated simulations producing probabilistic NPV/IRR/payback distributions with sensitivity analysis
4. **AI Agent Console** — Three specialized Claude agents:
   - **Capital Allocation Analyst** — Recommends optimal capital deployment across workstreams
   - **Risk & Scenario Agent** — Interprets simulation results and recommends risk-retirement investments
   - **Investment Memo Writer** — Generates board-ready investment memos

### Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit Dashboard             │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
│  │Cash Flow│ │ Capital  │ │ Monte Carlo  │  │
│  │  Model  │ │Efficiency│ │   & Risk     │  │
│  └────┬────┘ └────┬─────┘ └──────┬───────┘  │
│       └───────────┼──────────────┘           │
│                   ▼                          │
│         ┌─────────────────┐                  │
│         │   Agent Router  │                  │
│         └────────┬────────┘                  │
│    ┌─────────────┼─────────────┐             │
│    ▼             ▼             ▼             │
│ ┌──────┐   ┌─────────┐   ┌─────────┐        │
│ │CapAl │   │  Risk   │   │  Memo   │        │
│ │Analyst│   │  Agent  │   │ Writer  │        │
│ └──────┘   └─────────┘   └─────────┘        │
│         (Claude API)                         │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key (optional — app works in demo mode without it)
export ANTHROPIC_API_KEY=sk-ant-...

# Run the app
python -m streamlit run app.py
```

## Tech Stack

- **Python** — Core language
- **Streamlit** — Interactive dashboard
- **Plotly** — Data visualization
- **NumPy/Pandas** — Financial modeling & Monte Carlo simulation
- **Claude (Anthropic)** — Multi-agent AI system
- **SciPy** — Statistical analysis

## TeraWave Program Context

TeraWave is Blue Origin's satellite communications network: 5,408 optically interconnected satellites (5,280 LEO + 128 MEO) delivering 6 Tbps symmetrical capacity to ~100,000 enterprise, data center, and government sites worldwide. First deployment targeted for Q4 2027.

## License

MIT
