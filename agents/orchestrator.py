"""
Agent Orchestrator — Routes queries to specialized agents and synthesizes results.

Three specialized agents:
1. Capital Allocation Analyst — Evaluates workstream investments through the
   "progress per dollar" and "risk retired per dollar" lens.
2. Risk & Scenario Agent — Interprets Monte Carlo results, identifies tail risks,
   and recommends hedging strategies.
3. Investment Memo Writer — Produces structured investment memos and board-ready
   narratives from quantitative outputs.

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


def query_router(question: str, model_data: dict) -> dict:
    """
    Intelligent routing: analyze the question and route to the appropriate agent(s).
    Returns a dict with agent name and response.
    """
    question_lower = question.lower()

    # Route based on intent
    if any(kw in question_lower for kw in ["memo", "report", "write", "document", "board", "summary"]):
        return {
            "agent": "Investment Memo Writer",
            "response": generate_investment_memo(model_data),
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

    return "\n".join(parts)
