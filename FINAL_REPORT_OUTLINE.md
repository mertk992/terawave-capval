# Final Report Outline: TeraWave CapVal

Maximum length: 10 pages.

## 1. Business Problem And Value

Finance analysts evaluating major aerospace CapEx requests spend days pulling
evidence from budgets, planning models, policies, contracts, historical spend,
approval workflows, and variance reports. The business problem is not just
valuation math; it is evidence assembly and decision communication.

TeraWave CapVal addresses that bottleneck by turning a capital request into an
evidence-backed recommendation and concise investment memo.

Rubric fit: Problem Choice & Business Value.

## 2. System Overview

The prototype is a Streamlit application with one core end-to-end workflow:

```
CapEx request -> agent tool loop -> evidence pack -> recommendation -> memo
```

The agent is not a fixed workflow or chatbot. It receives structured tools,
chooses which tools to call, interprets results, and continues until it can
produce a recommendation.

Rubric fit: Technical Depth and Clarity.

## 3. Data And Implementation

The prototype uses synthetic but enterprise-shaped data:

- Budget pools modeled after ERP availability checks
- Historical CapEx requests modeled after procurement/workflow records
- Approval thresholds modeled after policy workflows
- A document corpus with policies, contracts, memos, and technical reports
- Variance data modeled after FP&A monthly close reporting
- A DCF model with NPV, IRR, payback, CapEx schedule, and Monte Carlo outputs

The implementation uses Python, Streamlit, Pandas, NumPy, Plotly, SciPy, and
Claude tool use.

Rubric fit: Technical Depth.

## 4. Technical Depth

Key technical elements to explain:

- Structured Claude tool definitions for budget, RAG, comparables, financial impact, approval routing, and variance checks
- Multi-turn tool loop in which tool results are appended back into the conversation
- DCF / NPV / IRR / payback model for capital valuation
- Correlated Monte Carlo simulation for probabilistic outcome analysis
- TF-IDF retrieval over synthetic source documents for grounded document intelligence
- Transparent audit trail in the UI for each reasoning step, tool call, and tool result
- Evidence pack extraction that converts raw tool outputs into a memo-ready structure

Rubric fit: Technical Depth.

## 5. Evaluation And Evidence

Use the app's evaluation evidence panel to report two concrete tests:

1. Repeatability: run the same critical-path CapEx request 5 times and report
   recommendation consistency plus NPV / ROI range.
2. Ground-truth scenarios: run one clearly approve case, one clearly reject
   case, and one ambiguous / conditional case, then compare expected vs.
   observed recommendation.

| Scenario | Expected Behavior | Evidence To Report |
| --- | --- | --- |
| Repeatability check | Same recommendation across 5 runs | Recommendation consistency %, NPV range, ROI range |
| Clearly approve | Approve | Budget fit, high NPV / ROI, strong risk-retirement rationale |
| Clearly reject | Reject | Budget breach, low strategic fit, approval burden |
| Ambiguous / conditional | Approve with Conditions | Budget fit but weaker strategic fit, milestone controls |

Score each run on tool-selection accuracy, recommendation correctness, evidence
grounding, numerical consistency, business usability, and estimated cycle-time
compression.

The memo should be used as an evaluation artifact. It should explicitly cite
which tools were called and the specific evidence behind the recommendation,
such as remaining budget headroom, NPV impact, approval tier, comparable
requests, source documents, and variance context.

Rubric fit: Evaluation & Evidence.

## 6. Deployment Trade-Offs And Limitations

The prototype is decision support, not autonomous approval. A production system
would need:

- Human approval before funds are committed
- Permissions and audit logging across ERP, planning, document, procurement, and workflow systems
- Privacy controls for financial and vendor data
- Prompt-injection defenses for retrieved documents
- Monitoring for hallucinations and numerical inconsistencies
- Model/version governance and fallback behavior
- Clear disclaimers because this project uses synthetic data

Rubric fit: Problem Choice & Business Value and Clarity.

## 7. Conclusion

TeraWave CapVal demonstrates a practical use of agentic AI in corporate finance:
not replacing finance judgment, but compressing the repetitive evidence assembly
and memo-writing work that slows capital allocation decisions.
