"""
Agentic CapEx Workflow Engine — True tool-use agent loop.

This is the core of what makes the system genuinely agentic:
1. Claude receives a CapEx request + tool definitions
2. Claude PLANS which tools to call (not hardcoded)
3. Claude calls tools, interprets results, decides what to do next
4. Claude chains multiple tool calls based on intermediate findings
5. Claude produces a final recommendation with full reasoning trace

The agent loop handles the tool_use API pattern:
  User message → Claude response with tool_use → execute tool → feed result back → repeat

⚠️ ALL DATA IS SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic

import config
from agents.tools import TOOL_DEFINITIONS, execute_tool

client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

MAX_TOOL_ROUNDS = 10  # Safety limit on agent loops


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    step_number: int
    step_type: str        # "thinking", "tool_call", "tool_result", "final_answer"
    title: str            # Human-readable title for the step
    content: str          # The actual content (reasoning text, tool input, tool output)
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    duration_ms: Optional[int] = None


@dataclass
class AgentResult:
    """Complete result of an agentic workflow run."""
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_tool_calls: int = 0
    total_duration_ms: int = 0
    model: str = config.MODEL
    error: Optional[str] = None


# ── System Prompt for the Agentic CapEx Analyst ─────────────────────────────

AGENTIC_CAPEX_PROMPT = """You are an autonomous Capital Expenditure Analyst for Blue Origin's TeraWave satellite constellation program.

YOUR MISSION: Independently analyze CapEx requests by gathering data, running analyses, and producing actionable recommendations. You have tools to check budgets, search documents, find comparable requests, run financial impact analysis, check approval routing, and review variance status.

ANALYTICAL FRAMEWORK (critical — this is Blue Origin's capital philosophy):
- The goal is NOT cost suppression. It is to ACCELERATE capital deployment efficiently.
- Higher risk tolerance is acceptable when it buys more progress or retires more technical/market risk.
- For every dollar: "Does this investment accelerate progress or retire risk?"
- Empowerment-oriented: recommend giving teams authority when progress-per-dollar is high.

HOW TO WORK:
1. PLAN: Think about what information you need to evaluate this request
2. GATHER: Use your tools to collect budget status, policy requirements, comparable requests, financial impact, and variance data
3. ANALYZE: Synthesize findings — don't just summarize tool outputs, draw conclusions
4. RECOMMEND: Give a clear GO / CONDITIONAL GO / DEFER / NO-GO with specific rationale

IMPORTANT RULES:
- Call tools proactively — don't wait to be asked. You decide what data you need.
- Chain tool calls logically: e.g., check budget first, then if tight, check variance to see if there's underspend elsewhere that could offset.
- If a tool result raises a concern, investigate further with additional tool calls.
- Always check at least: budget status, financial impact, and approval routing.
- Search documents when the request involves vendor contracts, policy questions, or technical risk.
- Find comparable requests to establish precedent.
- Be specific and quantitative in your final recommendation.

⚠️ All financial figures are SYNTHETIC and for demonstration purposes only.
"""

AGENTIC_GENERAL_PROMPT = """You are an autonomous analyst for Blue Origin's TeraWave satellite constellation program.

You have tools to investigate budget status, search program documents, analyze financial impact, check approval routing, find comparable past requests, and review variance data. Use these tools proactively to answer questions with specific, data-grounded answers.

ANALYTICAL FRAMEWORK:
- Capital deployment acceleration, not cost suppression
- "Progress per dollar" and "risk retired per dollar" are the key metrics
- Higher risk tolerance when it buys more progress or retires technical risk
- Empowerment-oriented decision making

Use your tools to gather real data before answering. Don't speculate — investigate.

⚠️ All financial figures are SYNTHETIC and for demonstration purposes only.
"""


def run_agentic_workflow(
    query: str,
    system_prompt: str = AGENTIC_CAPEX_PROMPT,
    on_step: Optional[callable] = None,
) -> AgentResult:
    """
    Run the full agentic workflow loop.

    This is the core agent loop:
    1. Send user message + tools to Claude
    2. If Claude responds with tool_use blocks, execute each tool
    3. Feed tool results back as tool_result messages
    4. Repeat until Claude gives a final text response (no more tool calls)

    Args:
        query: The user's question or CapEx request description
        system_prompt: System prompt to use (defaults to CapEx analyst)
        on_step: Optional callback called with each AgentStep for real-time UI updates

    Returns:
        AgentResult with full reasoning chain and final answer
    """
    result = AgentResult(query=query)
    start_time = time.time()
    step_num = 0

    # Build the conversation
    messages = [{"role": "user", "content": query}]

    for round_num in range(MAX_TOOL_ROUNDS):
        # Call Claude with tools
        step_num += 1
        call_start = time.time()

        try:
            response = client.messages.create(
                model=config.MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except Exception as e:
            result.error = f"API error: {str(e)}"
            break

        call_duration = int((time.time() - call_start) * 1000)

        # Process response content blocks
        has_tool_use = False
        text_parts = []
        tool_outputs = {}  # Cache: tool_use_id → output string

        for block in response.content:
            if block.type == "text":
                # Claude is thinking / providing intermediate reasoning
                text_parts.append(block.text)
                thinking_step = AgentStep(
                    step_number=step_num,
                    step_type="thinking",
                    title=f"Agent Reasoning (Round {round_num + 1})",
                    content=block.text,
                    duration_ms=call_duration,
                )
                result.steps.append(thinking_step)
                if on_step:
                    on_step(thinking_step)

            elif block.type == "tool_use":
                has_tool_use = True
                step_num += 1

                # Record the tool call
                tool_call_step = AgentStep(
                    step_number=step_num,
                    step_type="tool_call",
                    title=f"Calling: {block.name}",
                    content=json.dumps(block.input, indent=2),
                    tool_name=block.name,
                    tool_input=block.input,
                )
                result.steps.append(tool_call_step)
                result.total_tool_calls += 1
                if on_step:
                    on_step(tool_call_step)

                # Execute the tool (once, cached for API response)
                step_num += 1
                tool_start = time.time()
                tool_output = execute_tool(block.name, block.input)
                tool_duration = int((time.time() - tool_start) * 1000)
                tool_outputs[block.id] = tool_output

                # Record the tool result
                tool_result_step = AgentStep(
                    step_number=step_num,
                    step_type="tool_result",
                    title=f"Result: {block.name}",
                    content=tool_output,
                    tool_name=block.name,
                    duration_ms=tool_duration,
                )
                result.steps.append(tool_result_step)
                if on_step:
                    on_step(tool_result_step)

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": response.content})

        if has_tool_use:
            # Build tool results message using cached outputs
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_outputs[block.id],
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool calls — Claude is done, this is the final answer
            final_text = "\n".join(text_parts)
            result.final_answer = final_text

            final_step = AgentStep(
                step_number=step_num,
                step_type="final_answer",
                title="Final Recommendation",
                content=final_text,
            )
            # Replace the last thinking step with a final_answer step
            if result.steps and result.steps[-1].step_type == "thinking":
                result.steps[-1] = final_step
            else:
                result.steps.append(final_step)

            if on_step:
                on_step(final_step)
            break

        # Check stop reason
        if response.stop_reason == "end_turn":
            final_text = "\n".join(text_parts)
            result.final_answer = final_text
            break

    result.total_duration_ms = int((time.time() - start_time) * 1000)
    return result


def run_capex_analysis(
    title: str,
    description: str,
    amount_m: float,
    budget_pool: str,
    priority_tag: str,
    urgency: str,
    justification: str,
    completion_months: int = 12,
    requestor: str = "Analyst",
    on_step: Optional[callable] = None,
) -> AgentResult:
    """
    Run a full agentic CapEx analysis.

    Constructs a detailed prompt and lets the agent autonomously gather
    data and produce a recommendation.
    """
    query = f"""Analyze the following Capital Expenditure Request:

**Request Title:** {title}
**Amount:** ${amount_m}M
**Budget Pool:** {budget_pool}
**Priority:** {priority_tag}
**Urgency:** {urgency}
**Expected Completion:** {completion_months} months
**Requestor:** {requestor}

**Description:** {description}

**Justification:** {justification}

Please perform a comprehensive analysis:
1. Check the budget pool status — is there room for this request?
2. Run a financial impact analysis (NPV, ROI, payback)
3. Determine the approval routing and SLA
4. Find comparable past requests for precedent
5. Search relevant documents for any policy requirements or vendor context
6. Check variance status for the relevant workstream
7. Synthesize all findings into a clear recommendation

End with a definitive recommendation: APPROVE / APPROVE WITH CONDITIONS / DEFER / REJECT
Include specific conditions or risk mitigants if applicable."""

    return run_agentic_workflow(
        query=query,
        system_prompt=AGENTIC_CAPEX_PROMPT,
        on_step=on_step,
    )


def run_general_query(
    query: str,
    on_step: Optional[callable] = None,
) -> AgentResult:
    """
    Run a general analytical query through the agentic engine.

    The agent will autonomously decide which tools to use to answer
    the question.
    """
    return run_agentic_workflow(
        query=query,
        system_prompt=AGENTIC_GENERAL_PROMPT,
        on_step=on_step,
    )
