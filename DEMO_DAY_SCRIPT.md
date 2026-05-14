# Demo Day Script: TeraWave CapVal

Target length: 5-7 minutes.

## Opening

"TeraWave CapVal is an agentic CapEx analyst. The business problem is that
finance teams often spend days assembling the evidence behind a capital
recommendation: budget status, NPV, approval policy, comparable requests,
variance context, and source documents. This prototype compresses that into one
auditable workflow."

## Demo Scenario

Use the CapEx Workflow tab and click **Load Demo Day Scenario**.

Scenario:

- Request: Critical-Path Launch Acceleration Package
- Amount: $45M
- Budget pool: TeraWave - Launch Services
- Priority: Critical Path
- Urgency: Expedited
- Expected completion: 9 months

Business framing:

"This is the kind of request that should not be evaluated only through cost
control. If it pulls operational capability forward or retires schedule risk,
it may be worth approving with controls."

## Live Walkthrough

1. Submit the demo request.
2. Show the agent's live tool trace.
3. Point out that the agent chooses tools rather than following a fixed script.
4. Highlight the recommendation: GO, CONDITIONAL GO, DEFER, or NO-GO.
5. Open the Evidence Pack & Concise Investment Memo expander.
6. Show that the memo cites the same budget, financial, routing, precedent,
   variance, and document evidence surfaced by the workflow.
7. Download the memo to show the tangible deliverable.

## What To Emphasize

- The prototype is useful to a business audience because it ends with a clear
  recommendation and memo, not just charts.
- The technical depth is in the tool-use loop, structured tools, DCF model,
  Monte Carlo analysis, RAG, variance checks, and audit trail.
- The evaluation is scenario-based: each test checks whether the agent called
  the right tools, used evidence correctly, kept numbers consistent, and made a
  sensible recommendation.

## Deployment Trade-Offs

Close with:

"This is decision support, not automated approval. In production I would keep a
human in the loop, add permissions and audit logging, defend against prompt
injection in retrieved documents, and connect the tools to real ERP, planning,
document, procurement, and workflow systems."

## Backup If The API Is Unavailable

The app has a rule-based fallback that still demonstrates the request intake,
financial metrics, approval routing, audit trail, evidence pack, and memo. If
the API is unavailable, explain that the live agentic path uses the same visible
tool schema and that the fallback keeps the demo reliable.
