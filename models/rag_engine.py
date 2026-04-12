"""
Document Intelligence — RAG Engine

Lightweight retrieval-augmented generation over synthetic financial documents.
Uses TF-IDF vectorization for retrieval (no external vector DB dependency)
and Claude for grounded Q&A.

⚠️ ALL DOCUMENTS ARE SYNTHETIC FOR DEMONSTRATION PURPOSES ONLY.
"""

from __future__ import annotations

import numpy as np
import re
from dataclasses import dataclass, field
from typing import Optional

import config


# ── Synthetic Document Corpus ────────────────────────────────────────────────

@dataclass
class Document:
    id: str
    title: str
    doc_type: str  # "contract", "memo", "report", "policy", "proposal"
    source: str
    date: str
    content: str
    metadata: dict = field(default_factory=dict)


DOCUMENT_CORPUS = [
    Document(
        id="DOC-001",
        title="TeraWave Ground Segment — Managed Services Agreement (DataLink Corp)",
        doc_type="contract",
        source="Procurement",
        date="2025-09-15",
        content="""MANAGED SERVICES AGREEMENT

Between: Blue Origin Satellite Systems LLC ("Client")
And: DataLink Corp ("Provider")
Effective Date: September 15, 2025
Term: 7 years with two 3-year renewal options

SCOPE OF SERVICES:
Provider shall deliver managed ground station operations for the TeraWave constellation at 12 gateway sites across North America, Europe, and Asia-Pacific. Services include:
- 24/7 Network Operations Center (NOC) staffing and monitoring
- RF equipment maintenance and calibration (Q/V-band systems)
- Uplink/downlink management with 99.95% availability SLA
- Site security and facility management
- Spare parts inventory management (minimum 15% critical spares on-site)

PRICING:
- Base fee: $18.5M per year for first 12 sites
- Additional sites: $1.2M per site per year
- Annual escalation: CPI + 1.5%, capped at 5%
- Performance bonus: 2% of base fee if availability exceeds 99.99%
- Early termination fee: 18 months of base fee if terminated before Year 3

KEY PERFORMANCE INDICATORS:
- Gateway availability: ≥99.95% per site per month
- Mean time to repair: ≤4 hours for critical systems
- Incident response: ≤15 minutes for P1 incidents
- Planned maintenance windows: ≤8 hours per month per site

INTELLECTUAL PROPERTY:
All monitoring data, telemetry, and operational logs are the exclusive property of Client.
Provider shall not use Client data for any purpose other than service delivery.
Provider's proprietary NOC software is licensed to Client for the term of agreement.

LIABILITY AND INSURANCE:
Provider maintains $50M professional liability coverage.
Consequential damages capped at 2x annual base fee.
Force majeure includes: natural disasters, government action, satellite anomalies.

CHANGE MANAGEMENT:
Changes to service scope require 60 days written notice.
Cost impact assessment due within 15 business days of change request.
Client may add up to 6 additional sites per year without contract amendment.""",
        metadata={"vendor": "DataLink Corp", "value_m": 129.5, "term_years": 7},
    ),
    Document(
        id="DOC-002",
        title="TeraWave MEO Satellite Bus — Vendor Selection Memo",
        doc_type="memo",
        source="Engineering / Procurement",
        date="2025-11-20",
        content="""INTERNAL MEMO — CONFIDENTIAL

To: TeraWave Program Office
From: Satellite Engineering, Procurement
Date: November 20, 2025
Re: MEO Satellite Bus Vendor Down-Select

EXECUTIVE SUMMARY:
After a 6-month competitive evaluation, we recommend Northstar Space Systems as the primary vendor for the TeraWave MEO satellite bus. This memo summarizes the evaluation process, scoring, and rationale.

EVALUATION CRITERIA (weighted):
1. Technical capability (30%) — Bus performance, power budget, thermal management
2. Manufacturing scalability (25%) — Ability to deliver 128 units on schedule
3. Cost (20%) — Unit price, NRE, and total program cost
4. Schedule risk (15%) — Production timeline, supply chain resilience
5. Heritage/reliability (10%) — Flight history, anomaly rate

CANDIDATES EVALUATED:
A) Northstar Space Systems — Score: 87/100
   - Strongest thermal management design for MEO orbit environment
   - Proven production line capable of 3 units/month sustained rate
   - Unit price: $22.8M (includes integration and test)
   - NRE: $180M for MEO-specific adaptations
   - 45 satellites in orbit with 99.2% reliability over 5 years
   - Risk: Single facility — disaster recovery plan required

B) AstroForge Industries — Score: 79/100
   - Innovative modular design enabling faster integration
   - Less mature production capability (max 1.5 units/month currently)
   - Unit price: $19.5M (lower but with more schedule risk)
   - NRE: $240M (more development needed)
   - 12 satellites in orbit with 98.8% reliability
   - Risk: Scaling production to required rate is unproven

C) Constellation Dynamics — Score: 72/100
   - Strong heritage in GEO buses, less MEO experience
   - Unit price: $26.2M
   - NRE: $150M
   - Risk: MEO thermal and radiation environment requires significant redesign

RECOMMENDATION:
Select Northstar Space Systems as primary vendor. Key decision factors:
1. Production scalability is critical path — 128 units in <3 years requires proven capacity
2. Thermal management is the #1 technical risk for MEO — Northstar's design is most mature
3. Higher unit price ($22.8M vs $19.5M) is justified by lower schedule risk
4. Recommend maintaining AstroForge as qualified alternate vendor (invest $15M in qualification)

BUDGET IMPACT:
- Primary contract value: $3.1B (128 units + NRE + spares)
- Alternate vendor qualification: $15M
- Total: $3.115B vs. budgeted $3.2B — $85M favorable to budget

NEXT STEPS:
1. Board approval for vendor selection (December 2025)
2. Contract negotiation (January–February 2026)
3. NRE kickoff (March 2026)
4. First article delivery (Q2 2027)""",
        metadata={"recommended_vendor": "Northstar Space Systems", "contract_value_m": 3100},
    ),
    Document(
        id="DOC-003",
        title="TeraWave Capital Allocation Policy",
        doc_type="policy",
        source="Corporate Finance",
        date="2025-07-01",
        content="""TERAWAVE PROGRAM — CAPITAL ALLOCATION POLICY

Version: 2.0
Effective: July 1, 2025
Owner: VP Corporate Finance

PURPOSE:
This policy establishes the framework for capital allocation decisions within the TeraWave program. It reflects Blue Origin's commitment to accelerating progress through efficient capital deployment rather than traditional cost minimization.

GUIDING PRINCIPLES:
1. SPEED OVER SAVINGS — When evaluating trade-offs, prioritize options that accelerate program timeline, even if they cost 10-20% more than alternatives.
2. RISK RETIREMENT — Allocate capital proactively to retire technical and market risks early. A dollar spent retiring risk in Year 1 is worth more than a dollar saved in Year 3.
3. EMPOWERMENT — Program managers are authorized to make capital decisions within their tier without escalation, provided they document the progress-acceleration rationale.
4. PROGRESS MEASUREMENT — All CapEx requests must quantify expected progress impact using the Program Progress Framework (PPF) scoring methodology.

APPROVAL THRESHOLDS:
- Up to $5M: Program Manager approval (3-day SLA)
- $5M–$25M: VP Finance approval (5-day SLA)
- $25M–$100M: CFO approval (7-day SLA)
- Over $100M: CEO/Board approval (14-day SLA)
Emergency requests may follow expedited process (50% of standard SLA).

BUDGET FLEXIBILITY:
- Program managers may reallocate up to 15% of their annual budget between workstreams without VP approval, provided the reallocation accelerates the critical path.
- Unspent quarterly budget does NOT revert — it carries forward within the fiscal year.
- Supplemental funding requests require demonstration of progress-per-dollar above program average.

PROGRESS FRAMEWORK (PPF):
Each capital request is scored on two dimensions:
- Progress Contribution (0–1.0): How much does this advance the program toward operational capability?
- Risk Retirement Value (0–1.0): How much technical/market/regulatory risk does this eliminate?

Combined PPF Score = 0.5 × Progress + 0.5 × Risk Retirement
Requests with PPF Score ≥ 0.6 are fast-tracked.
Requests with PPF Score < 0.3 require additional justification.

VENDOR SELECTION:
- Sole-source contracts up to $10M allowed if vendor is on approved supplier list
- Competitive bid required for >$10M unless strategic justification documented
- Speed-to-contract weighted at 20% in vendor evaluation scoring

REPORTING:
- Monthly: CapEx dashboard showing spend vs. plan, PPF scores, approval pipeline
- Quarterly: Capital efficiency review — actual progress per dollar vs. planned
- Annual: Program-level capital allocation retrospective""",
        metadata={"version": "2.0", "owner": "VP Corporate Finance"},
    ),
    Document(
        id="DOC-004",
        title="TeraWave Launch Services — New Glenn Manifest Planning",
        doc_type="report",
        source="Mission Planning",
        date="2026-01-15",
        content="""TERAWAVE LAUNCH MANIFEST — PLANNING DOCUMENT

Prepared by: Mission Planning Team
Date: January 15, 2026
Classification: Internal Use Only

OVERVIEW:
This document outlines the planned launch manifest for deploying the TeraWave constellation using New Glenn launch vehicles. The manifest covers the initial deployment phase (2027–2030) targeting operational capability with 1,500+ satellites.

VEHICLE CAPABILITY:
- New Glenn payload to LEO (28.5° inclination): 45,000 kg
- TeraWave LEO satellite mass (with dispenser): ~350 kg each
- Maximum satellites per launch: ~60 (mass-limited)
- Estimated launches required for full LEO constellation: 88 launches
- MEO satellites launched separately: 16 launches (8 sats per launch)
- Total launches: 104

MANIFEST SUMMARY:
Phase 1 — Initial Coverage (2027-2028):
  Launches 1-6:   Q4 2027 – Q1 2028 (6 launches, 360 LEO satellites)
  Target: Demonstrate basic service capability to anchor customers
  Launch cadence: 1 per month
  Cost per launch: $32M (internal pricing, marginal cost basis)

Phase 2 — Regional Coverage (2028-2029):
  Launches 7-30:  Q2 2028 – Q4 2029 (24 launches, 1,440 LEO satellites)
  Target: North America, Europe, and key Asia-Pacific coverage
  Launch cadence: Ramp to 2 per month by end of Phase 2
  Cost per launch: $28M (production rate benefit)

Phase 3 — Global Coverage (2029-2031):
  Launches 31-88: 2029–2031 (58 launches, 3,480 LEO satellites)
  MEO Launches 1-16: Interleaved (128 MEO satellites)
  Launch cadence: 2-3 per month sustained
  Cost per launch: $25M (high-rate production)

COST SUMMARY:
- Phase 1: $192M (6 launches)
- Phase 2: $672M (24 launches)
- Phase 3: $1,450M (58 LEO launches) + $640M (16 MEO launches)
- Contingency (10%): $295M
- Total launch services budget: $3,249M

KEY RISKS:
1. Launch cadence — Achieving 2+ launches/month requires pad turnaround <20 days
2. Vehicle reliability — Any launch failure delays manifest by 3-6 months minimum
3. Satellite production matching — Factory must deliver 120+ satellites/month at peak
4. Weather and range scheduling at Cape Canaveral

INSURANCE:
- First 10 launches: Full replacement value coverage ($650M per launch)
- Subsequent launches: Self-insured with reserve fund ($50M per launch set-aside)
- Total insurance/reserve budget: $4.7B""",
        metadata={"total_launches": 104, "total_cost_m": 3249},
    ),
    Document(
        id="DOC-005",
        title="TeraWave Enterprise Customer Pipeline — Q4 2025 Review",
        doc_type="report",
        source="Business Development",
        date="2025-12-20",
        content="""TERAWAVE ENTERPRISE PIPELINE REVIEW — Q4 2025

Prepared by: Business Development
Confidentiality: Restricted

PIPELINE SUMMARY:
Total pipeline value: $4.2B in annual recurring revenue (ARR) across 847 qualified opportunities.
Weighted pipeline (probability-adjusted): $1.8B ARR
Target TAM: Enterprise/government secure connectivity market estimated at $28B by 2030.

SEGMENT BREAKDOWN:
1. Cloud & Data Center Providers (38% of pipeline)
   - 12 hyperscaler and Tier-1 colocation prospects
   - Use case: Intercontinental backbone diversity, disaster recovery links
   - Average deal size: $45M ARR
   - Win probability: 35-55%
   - Key requirement: Minimum 100 Gbps per link, <30ms latency

2. Defense & Intelligence (28% of pipeline)
   - 8 government agency opportunities (US, allied nations)
   - Use case: Resilient SATCOM, JADC2 connectivity, BLOS communications
   - Average deal size: $80M ARR
   - Win probability: 25-40% (long procurement cycles)
   - Key requirement: ITAR compliance, anti-jam capability, encryption

3. Financial Services (18% of pipeline)
   - 15 major banks and exchanges
   - Use case: Ultra-low-latency market data, trading venue connectivity
   - Average deal size: $12M ARR
   - Win probability: 40-60%
   - Key requirement: Sub-millisecond jitter, five-nines reliability

4. Energy & Resources (10% of pipeline)
   - 22 oil/gas, mining, and utility companies
   - Use case: Remote operations connectivity, SCADA backhaul
   - Average deal size: $5M ARR
   - Win probability: 50-70%
   - Key requirement: Coverage in remote/offshore locations

5. Maritime & Aviation (6% of pipeline)
   - 18 shipping lines and airlines
   - Use case: Fleet connectivity, passenger WiFi backbone
   - Average deal size: $8M ARR
   - Win probability: 30-50%

ANCHOR CUSTOMERS (signed LOIs):
- GlobalCloud Inc.: 200 Gbps backbone across 8 intercontinental routes — $52M ARR
- US DOD (Program of Record): Phase 1 evaluation contract — $15M initial, $120M ARR at scale
- Pinnacle Financial Group: 14 exchange-to-data-center links — $18M ARR

COMPETITIVE POSITIONING:
vs. Starlink Business: TeraWave offers 10x capacity per site, dedicated (not shared) bandwidth
vs. Telesat Lightspeed: TeraWave has broader coverage and vertical integration advantage
vs. SES mPOWER: TeraWave offers lower latency (LEO) with MEO backbone capacity
vs. Subsea fiber: TeraWave provides instant deployment, no cable-cut risk, competitive at >5,000km

REVENUE FORECAST (conservative):
- 2028: $200M ARR (anchor customers + early adopters)
- 2029: $800M ARR (regional coverage enables broader adoption)
- 2030: $2.0B ARR (approaching global coverage)
- 2032: $5.0B+ ARR (full constellation, enterprise scale)""",
        metadata={"pipeline_value_b": 4.2, "weighted_pipeline_b": 1.8, "anchor_customers": 3},
    ),
    Document(
        id="DOC-006",
        title="TeraWave Optical Inter-Satellite Link — Technical Risk Assessment",
        doc_type="report",
        source="Systems Engineering",
        date="2026-02-10",
        content="""TECHNICAL RISK ASSESSMENT: OPTICAL INTER-SATELLITE LINKS (OISL)

Document: TW-ENG-2026-042
Date: February 10, 2026
Classification: Internal

OVERVIEW:
The OISL subsystem is the highest-risk, highest-reward technology element in the TeraWave architecture. This assessment identifies key risks, quantifies their potential impact, and recommends risk retirement investments.

TECHNOLOGY DESCRIPTION:
- Free-space optical communication between satellites using 1550nm laser terminals
- Data rate: 100 Gbps per link (LEO-LEO), 400 Gbps per link (LEO-MEO)
- Pointing accuracy requirement: <1 microradian
- Acquisition time: <5 seconds for LEO-LEO, <15 seconds for LEO-MEO
- Each LEO satellite has 4 OISL terminals; each MEO satellite has 8

RISK REGISTER:

RISK-001: Pointing and Tracking in LEO Environment
- Probability: Medium (40%)
- Impact: High — failure to maintain lock causes link drops
- Root cause: Vibration from reaction wheels, thermal distortion of optical bench
- Mitigation: Vibration isolation system (funded, $45M), prototype testing Q2 2026
- Residual risk after mitigation: Low (15%)
- Cost to retire: $45M (already budgeted)

RISK-002: Optical Terminal Manufacturing Yield
- Probability: Medium-High (55%)
- Impact: High — low yield drives cost overrun and schedule delay
- Root cause: Precision alignment of optical components at scale
- Current yield: 72% (prototype line)
- Required yield for cost target: >90%
- Mitigation: Automated alignment station investment ($28M), hire 15 optical engineers
- Cost to retire: $35M
- Expected yield after investment: 88-93%

RISK-003: LEO-MEO Link Budget Margin
- Probability: Medium (35%)
- Impact: Critical — insufficient margin means MEO backbone underperforms
- Root cause: Atmospheric scintillation, space weather effects on optical path
- Mitigation: Adaptive optics upgrade ($18M), increased transmit power (+15%)
- Cost to retire: $25M
- Residual risk: Low-Medium (20%)

RISK-004: On-Orbit Degradation of Optical Surfaces
- Probability: Low-Medium (25%)
- Impact: Medium — gradual performance reduction over satellite lifetime
- Root cause: Atomic oxygen erosion, micrometeorite damage, contamination
- Mitigation: Protective coatings (included in design), redundant terminals
- Cost to retire: $8M (coating validation testing)
- Expected lifetime: >7 years before significant degradation

RISK-005: Regulatory — Laser Safety Compliance
- Probability: Low (15%)
- Impact: Medium — could delay deployment in certain orbital slots
- Root cause: Evolving ITU/national regulations on space-based lasers
- Mitigation: Proactive engagement with ITU, FAA, and international bodies
- Cost to retire: $3M (regulatory affairs + compliance testing)

AGGREGATE RISK SUMMARY:
- Total risk retirement investment needed: $116M
- Expected risk reduction: From aggregate 65% confidence to 90% confidence in OISL performance
- Return on risk investment: Each 1% of confidence improvement valued at ~$200M in program NPV
- Recommendation: FUND ALL RISK RETIREMENT ITEMS — total $116M is <1% of program value and moves OISL from "amber" to "green" on program risk register

DECISION REQUIRED BY: March 15, 2026""",
        metadata={"total_risk_investment_m": 116, "risks_identified": 5},
    ),
]


# ── Retrieval Engine (TF-IDF based — no external deps) ──────────────────────

class SimpleVectorStore:
    """
    Lightweight TF-IDF vector store for document retrieval.
    No external vector DB required — runs entirely in memory.
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.tfidf_matrix: np.ndarray = np.array([])
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Simple stop word removal
        stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                 'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
                 'both', 'either', 'neither', 'each', 'every', 'all', 'any', 'few',
                 'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                 'than', 'too', 'very', 'just', 'this', 'that', 'these', 'those', 'it',
                 'its', 'per', 'if', 'then', 'also', 'about', 'up', 'out', 'which'}
        return [t for t in tokens if t not in stops and len(t) > 2]

    def _build_index(self):
        # Build vocabulary
        doc_tokens = []
        for doc in self.documents:
            full_text = f"{doc.title} {doc.content}"
            tokens = self._tokenize(full_text)
            doc_tokens.append(tokens)

        # Build vocab from all tokens
        all_tokens = set()
        for tokens in doc_tokens:
            all_tokens.update(tokens)
        self.vocab = {token: i for i, token in enumerate(sorted(all_tokens))}

        n_docs = len(self.documents)
        n_terms = len(self.vocab)

        # Term frequency matrix
        tf = np.zeros((n_docs, n_terms))
        for i, tokens in enumerate(doc_tokens):
            for token in tokens:
                if token in self.vocab:
                    tf[i, self.vocab[token]] += 1
            # Normalize by doc length
            if len(tokens) > 0:
                tf[i] /= len(tokens)

        # IDF
        df = np.sum(tf > 0, axis=0)
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1

        # TF-IDF
        self.tfidf_matrix = tf * self.idf

        # Normalize rows
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_matrix /= norms

    def search(self, query: str, top_k: int = 3) -> list[tuple[Document, float]]:
        """Search for documents relevant to the query."""
        tokens = self._tokenize(query)

        # Build query vector
        query_vec = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                query_vec[self.vocab[token]] = 1
        query_vec *= self.idf

        # Normalize
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # Cosine similarity
        similarities = self.tfidf_matrix @ query_vec

        # Rank
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # minimum relevance threshold
                results.append((self.documents[idx], float(similarities[idx])))

        return results


# ── Global store instance ────────────────────────────────────────────────────

_store: Optional[SimpleVectorStore] = None


def get_store() -> SimpleVectorStore:
    global _store
    if _store is None:
        _store = SimpleVectorStore(DOCUMENT_CORPUS)
    return _store


def search_documents(query: str, top_k: int = 3) -> list[tuple[Document, float]]:
    """Search the document corpus."""
    return get_store().search(query, top_k)


def get_all_documents() -> list[Document]:
    """Return all documents in the corpus."""
    return DOCUMENT_CORPUS


def format_context_for_llm(results: list[tuple[Document, float]]) -> str:
    """Format search results as context for the LLM."""
    parts = []
    for doc, score in results:
        parts.append(f"--- DOCUMENT: {doc.title} (Relevance: {score:.2f}) ---")
        parts.append(f"Type: {doc.doc_type} | Source: {doc.source} | Date: {doc.date}")
        # Truncate very long documents for context window
        content = doc.content[:3000] if len(doc.content) > 3000 else doc.content
        parts.append(content)
        parts.append("")
    return "\n".join(parts)
