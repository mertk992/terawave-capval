"""
Agent Orchestrator — Routes queries to specialized agents and synthesizes results.

Five specialized agents:
1. Capital Allocation Analyst — Evaluates workstream investments through the
   "progress per dollar" and "risk retired per dollar" lens.
2. Risk & Scenario Agent — Interprets Monte Carlo results, identifies tail risks,
   and recommends hedging strategies.
3. Investment Memo Writer — Produces structured investment memos and board-ready
   narratives from quantitative outputs.
4. M&A / Make-vs-Buy Analyst — Evaluates build-vs-buy-vs-partner decisions for
   strategic supply chain components.
5. Strategic Portfolio Planner — Long-range scenario planning and portfolio-level
   capital allocation across multiple program bets.

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

import json
from anthropic import Anthropic

import config

client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

# ── System Prompts ───────────────────────────────────────────────────────────

CAPITAL_ANALYST_PROMPT = """You are a senior Capital Allocation Analyst on Blue Origin's corporate finance team,
evaluating the TeraWave satellite constellation program.

YOUR ANALYTICAL FRAMEWORK (this is critical):
- The goal is NOT cost suppression to increase margin. It is to ACCELERATE the deployment of capital efficiently.
- Higher risk tolerance is acceptable when it buys more progress or retires more technical/market risk.
- The key question for every dollar spent: "Does this investment accelerate progress or retire risk?"
- Focus spend where we earn MORE PROGRESS or retire MORE RISK per dollar.
- Empowerment-oriented: recommend giving teams more authority to make judgment calls when the expected
  progress-per-dollar is high, even if traditional risk frameworks would flag it.

When analyzing data, you should:
1. Identify which workstreams deliver the highest progress-per-dollar and risk-retired-per-dollar
2. Recommend capital reallocation to maximize program velocity
3. Flag workstreams where additional investment could accelerate the critical path
4. Quantify trade-offs between speed and cost

Respond with specific, quantitative recommendations. Reference the data provided.
Format currency in $M or $B as appropriate. Be direct and decisive — this is a fast-moving program.

⚠️ IMPORTANT: All financial figures are SYNTHETIC and for demonstration purposes only."""

RISK_AGENT_PROMPT = """You are a Risk & Scenario Analysis specialist on Blue Origin's corporate finance team,
focused on the TeraWave satellite constellation program.

YOUR RISK FRAMEWORK (aligned with program priorities):
- We have HIGHER risk tolerance than traditional aerospace programs because speed-to-market is critical.
- Risk is not something to minimize — it's something to RETIRE strategically through investment.
- The question is not "what could go wrong?" but "where should we invest to retire the most risk per dollar?"
- Tail risks (>P90 outcomes) still matter for capital planning, but moderate risks should be accepted
  when the progress payoff justifies them.

When analyzing Monte Carlo results and scenarios, you should:
1. Identify the top 3 risk factors driving outcome variance
2. Quantify the P10/P50/P90 envelope and explain what drives the tails
3. Recommend specific risk-retirement investments (which spend retires which risk)
4. Distinguish between risks we should accept (speed trade-offs) vs. risks we must mitigate (existential)
5. Frame findings in terms of capital efficiency, not risk avoidance

Be quantitative. Use the simulation data provided. Present risks as investment opportunities.

⚠️ IMPORTANT: All financial figures are SYNTHETIC and for demonstration purposes only."""

MEMO_WRITER_PROMPT = """You are an Investment Memo Writer producing board-ready documents for Blue Origin's
leadership team on the TeraWave satellite constellation program.

YOUR WRITING FRAMEWORK:
- Lead with the strategic case: TeraWave as a high-priority capital deployment opportunity
- Frame everything through the lens of "accelerating progress" and "retiring risk efficiently"
- Be direct, quantitative, and decisive — leadership wants clarity, not hedging
- Structure: Executive Summary → Investment Thesis → Financial Summary → Key Risks & Mitigants →
  Capital Allocation Recommendation → Decision Points
- Use bullet points and tables for data; narrative for strategic context
- Include specific dollar amounts, percentages, and timelines
- End with a clear GO / CONDITIONAL GO / NO-GO recommendation with conditions

The audience is senior leadership at an aerospace company that values:
- Speed of capital deployment over cost minimization
- Progress measurement and risk retirement over traditional risk avoidance
- Empowerment and judgment-based decision making
- Data-driven but action-oriented analysis

⚠️ IMPORTANT: All financial figures are SYNTHETIC and for demonstration purposes only.
Include a disclaimer at the end: "This memo was generated using synthetic data for demonstration purposes."
"""

MNA_ANALYST_PROMPT = """You are an M&A / Make-vs-Buy Analyst on Blue Origin's corporate strategy team,
evaluating vertical integration decisions for the TeraWave satellite constellation.

YOUR ANALYTICAL FRAMEWORK:
- Aerospace supply chains are deep and strategic — vertical integration decisions are core corporate finance problems
- Every component decision should be evaluated through three lenses: Build internally, Acquire, or Partner/License
- Speed-to-capability matters as much as cost — a cheaper option that takes 3 extra years may be worse
- Dependency risk on critical-path suppliers is a strategic vulnerability, not just a procurement issue
- Patent portfolios and IP overlap are key factors in technology acquisitions

When analyzing make-vs-buy decisions, you should:
1. Compare total cost of ownership across all three options (NPV basis)
2. Evaluate time-to-capability — how fast does each option get us operational?
3. Assess strategic risks: supplier dependency, IP control, integration complexity
4. Consider workforce and culture integration for acquisitions
5. Recommend the option that best balances cost, speed, risk, and strategic control

Be specific and quantitative. Reference the target company data and financial comparisons provided.

⚠️ IMPORTANT: All financial figures and company profiles are SYNTHETIC and for demonstration purposes only."""

PORTFOLIO_PLANNER_PROMPT = """You are a Strategic Portfolio Planner on Blue Origin's corporate finance team,
responsible for long-range capital allocation across multiple program bets.

YOUR PLANNING FRAMEWORK:
- Aerospace companies must manage portfolios of large, uncertain bets with different risk profiles
- Capital is finite — allocating more to one program means less for another
- Each program has different probability of success, time-to-revenue, and strategic value
- Macro scenarios (demand shifts, regulatory changes, funding constraints) affect all programs differently
- The goal is portfolio-level optimization: maximize expected value while managing downside exposure

When analyzing portfolio allocation, you should:
1. Evaluate each program's risk-adjusted return (probability-weighted NPV per dollar of capex)
2. Consider scenario sensitivity — which programs are robust across scenarios vs. fragile?
3. Recommend capital allocation priorities based on combined financial and strategic scoring
4. Identify portfolio-level risks (concentration, correlation, funding gaps)
5. Highlight decision points and optionality — where can we stage investments to preserve flexibility?

Present findings with clear prioritization. Use the scenario data and portfolio projections provided.

⚠️ IMPORTANT: All financial figures and program details are SYNTHETIC and for demonstration purposes only."""

# ── Agent Functions ──────────────────────────────────────────────────────────

def _call_agent(system_prompt: str, user_message: str, context_data: str = "") -> str:
    """Call a specialized agent with context data."""
    messages = []
    if context_data:
        messages.append({
            "role": "user",
            "content": f"Here is the current financial model data for TeraWave:\n\n{context_data}\n\n---\n\n{user_message}"
        })
    else:
        messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=config.MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


def query_capital_analyst(question: str, model_data: dict) -> str:
    """Route a question to the Capital Allocation Analyst agent."""
    context = _format_model_context(model_data)
    return _call_agent(CAPITAL_ANALYST_PROMPT, question, context)


def query_risk_agent(question: str, model_data: dict) -> str:
    """Route a question to the Risk & Scenario Agent."""
    context = _format_model_context(model_data)
    return _call_agent(RISK_AGENT_PROMPT, question, context)


def generate_investment_memo(model_data: dict) -> str:
    """Generate a full investment memo using the Memo Writer agent."""
    context = _format_model_context(model_data)
    prompt = (
        "Generate a comprehensive investment memo for the TeraWave program based on "
        "the financial model data provided. Include all sections: Executive Summary, "
        "Investment Thesis, Financial Summary, Key Risks & Mitigants, Capital Allocation "
        "Recommendation, and Decision Points. End with a clear recommendation."
    )
    return _call_agent(MEMO_WRITER_PROMPT, prompt, context)


def query_mna_analyst(question: str, model_data: dict) -> str:
    """Route a question to the M&A / Make-vs-Buy Analyst agent."""
    context = _format_model_context(model_data)
    return _call_agent(MNA_ANALYST_PROMPT, question, context)


def query_portfolio_planner(question: str, model_data: dict) -> str:
    """Route a question to the Strategic Portfolio Planner agent."""
    context = _format_model_context(model_data)
    return _call_agent(PORTFOLIO_PLANNER_PROMPT, question, context)


def query_router(question: str, model_data: dict) -> dict:
    """
    Intelligent routing: analyze the question and route to the appropriate agent(s).
    Returns a dict with agent name and response.
    """
    question_lower = question.lower()

    # Route based on intent
    if any(kw in question_lower for kw in ["memo", "report", "write", "document", "board"]):
        return {
            "agent": "Investment Memo Writer",
            "response": generate_investment_memo(model_data),
        }
    elif any(kw in question_lower for kw in ["m&a", "mna", "acqui", "build vs", "make vs", "buy vs", "partner", "vertical integration", "supplier", "target"]):
        return {
            "agent": "M&A / Make-vs-Buy Analyst",
            "response": query_mna_analyst(question, model_data),
        }
    elif any(kw in question_lower for kw in ["portfolio", "program bet", "long-range", "macro", "bull", "bear", "stress", "funding", "multi-program"]):
        return {
            "agent": "Strategic Portfolio Planner",
            "response": query_portfolio_planner(question, model_data),
        }
    elif any(kw in question_lower for kw in ["risk", "monte carlo", "probability", "tail", "p90", "p10", "simulation", "uncertainty", "scenario"]):
        return {
            "agent": "Risk & Scenario Agent",
            "response": query_risk_agent(question, model_data),
        }
    elif any(kw in question_lower for kw in ["allocat", "progress", "workstream", "deploy", "priorit", "invest", "capital", "accelerat", "efficiency"]):
        return {
            "agent": "Capital Allocation Analyst",
            "response": query_capital_analyst(question, model_data),
        }
    else:
        # Default: Capital Analyst for general questions
        return {
            "agent": "Capital Allocation Analyst",
            "response": query_capital_analyst(question, model_data),
        }


def _format_model_context(model_data: dict) -> str:
    """Format model outputs into a context string for the agents."""
    parts = []

    if "projection" in model_data:
        parts.append("## Year-by-Year Financial Projection\n")
        parts.append(model_data["projection"].to_string(index=False))

    if "metrics" in model_data:
        parts.append("\n## Key Valuation Metrics\n")
        for k, v in model_data["metrics"].items():
            parts.append(f"- {k}: {v}")

    if "mc_summary" in model_data:
        parts.append("\n## Monte Carlo Simulation Summary\n")
        for k, v in model_data["mc_summary"].items():
            parts.append(f"- {k}: {v}")

    if "progress_metrics" in model_data:
        parts.append("\n## Progress & Risk Retirement Metrics by Workstream\n")
        parts.append(model_data["progress_metrics"].to_string(index=False))

    if "tornado" in model_data:
        parts.append("\n## Sensitivity Analysis (Tornado)\n")
        parts.append(model_data["tornado"].to_string(index=False))

    if "mna_results" in model_data:
        parts.append("\n## M&A / Make-vs-Buy Analysis\n")
        for name, result in model_data["mna_results"].items():
            parts.append(f"\n### {name}")
            parts.append(f"Recommended: {result['recommended_option']}")
            for opt in ["build", "acquire", "partner"]:
                d = result[opt]
                parts.append(f"  {opt.title()}: Composite Score = {d['scores']['composite']}")

    if "portfolio_summary" in model_data:
        parts.append("\n## Portfolio Scenario Summary\n")
        parts.append(model_data["portfolio_summary"].to_string(index=False))

    if "portfolio_allocation" in model_data:
        parts.append("\n## Optimal Portfolio Allocation\n")
        parts.append(model_data["portfolio_allocation"].to_string(index=False))

    return "\n".join(parts)
