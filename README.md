# TeraWave CapVal: Agentic CapEx Memo Engine

An agentic CapEx decision-support system that turns a capital request into an
evidence-backed recommendation and concise investment memo.

> **⚠️ All financial data is synthetic and for demonstration purposes only.** This tool does not represent actual Blue Origin financials, projections, or internal data.

## Final Project Thesis

TeraWave CapVal compresses the finance workflow from request intake to
board-ready recommendation. A finance analyst submits a CapEx request; the
agent autonomously gathers evidence from synthetic enterprise-style systems,
runs financial and risk analysis, and produces a GO / CONDITIONAL / DEFER /
NO-GO recommendation with an audit trail and a 1-2 page memo.

The project is intentionally scoped around one high-value business workflow:

```
CapEx request -> tool-using agent -> evidence pack -> recommendation -> memo
```

This keeps the final report and Demo Day presentation concise while still
showing technical depth.

## What It Does

This is an agentic capital project valuation engine that goes beyond traditional
DCF analysis by incorporating a **"progress per dollar" and "risk retired per
dollar"** framework. It is designed for capital-intensive programs where the
goal is to **accelerate deployment**, not simply suppress costs.

### Core Components

1. **Financial Model** — 12-year DCF projection with CapEx by workstream, OpEx ramp, revenue schedule, IRR, NPV, and payback analysis
2. **Capital Efficiency Framework** — Maps each workstream by progress contribution and risk retirement value per dollar deployed
3. **Monte Carlo Simulation** — 2,500+ correlated simulations producing probabilistic NPV/IRR/payback distributions with sensitivity analysis
4. **Agentic CapEx Workflow** — Claude chooses and chains tools for budget, RAG, comparables, financial impact, approval routing, and variance evidence
5. **Evidence Pack + Memo** — The app converts tool outputs into a concise investment memo that names the tools called and the evidence driving the recommendation
6. **Demo Path + Evaluation Evidence** — The CapEx Workflow tab includes focused Demo Day narrative paths, ground-truth scenario tests, and repeatability checks

### Synthetic Evidence Corpus

The demo now uses an explicit synthetic source file at
`data/synthetic_enterprise_data.json`. Budget checks, approval routing,
historical request comparables, and RAG document search all load from that file
and return source metadata (`source_file`, `dataset_id`, `source_record_id`) in
tool outputs. This makes the prototype "real" in the important demo sense: the
agent grounds its answer in traceable source material, while the source material
is clearly labeled as synthetic.

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

## Evaluation Plan

The final report should evaluate the prototype with concrete evidence from the
app's CapEx Workflow evaluation panel:

- **Repeatability:** run the same critical-path CapEx request five times and report recommendation consistency, NPV range, and ROI range.
- **Ground-truth tests:** compare expected vs. observed recommendations for clearly approve, clearly reject, and ambiguous / conditional scenarios.
- **Memo evidence trace:** show that the memo cites budget, financial impact, approval routing, precedent, document-search, and variance evidence.
- **Business usability:** verify that the recommendation and memo can be understood without reading raw tool logs.

See `FINAL_REPORT_OUTLINE.md` and `DEMO_DAY_SCRIPT.md` for the concise final
submission structure.

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
