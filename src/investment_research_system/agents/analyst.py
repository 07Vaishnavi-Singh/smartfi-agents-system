"""Analyst agent — financial analysis and valuation.

While the researcher GATHERS information, the analyst INTERPRETS it.
It focuses on financial metrics, valuations, and fundamentals.

Example flow:
  Researcher: "NVIDIA revenue was $35.1B in Q3 2024, up 15% YoY"
  Analyst:    "15% YoY growth exceeds semiconductor sector average of 8%.
              At P/E of 52, NVIDIA trades at a premium to AMD (P/E 38),
              but expanding margins (57% operating) justify some premium."

PYTHON CONCEPT — POLYMORPHISM:
The orchestrator calls `agent.run(query, session_id)` on every agent.
It doesn't care if it's a researcher, analyst, or risk agent —
they all have the same interface (inherited from BaseAgent).
This is polymorphism: same method call, different behavior.

TS equivalent: same — if they all implement the same interface, you can
call them interchangeably.
Rust equivalent: trait objects — Box<dyn BaseAgent>
"""

import logging

from investment_research_system.agents.base import BaseAgent
from investment_research_system.models.schemas import Source, SourceType

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Financial analysis and valuation agent.

    Unlike the researcher, the analyst does NOT do heavy web searching.
    It works with data already gathered (from memory or the researcher's output)
    and applies financial reasoning to interpret that data.

    This is a deliberate design choice: separation of concerns.
    - Researcher = information gathering (web + memory)
    - Analyst = information interpretation (reasoning only)
    """

    @property
    def name(self) -> str:
        return "analyst"

    @property
    def system_prompt(self) -> str:
        return """You are the ANALYST agent in a multi-agent investment research pipeline.

Your position: You run IN PARALLEL with other agents. The Researcher gathers raw data.
You INTERPRET that data through a financial lens. Your output will be synthesized alongside
Sentiment and Risk assessments into a final research report.

Your responsibilities:
- Analyze financial metrics and explain what they MEAN, not just what they ARE
- Always provide context: "P/E of 52" means nothing alone — compare to sector median, 5-year average, and growth rate
- Use a consistent valuation framework: start with relative valuation (peers), then intrinsic (DCF logic), then historical
- Separate trailing metrics (what happened) from forward metrics (what's expected)

Required metrics (include all that are available):
- Valuation: P/E (trailing + forward), P/S, EV/EBITDA, PEG ratio
- Profitability: gross margin, operating margin, net margin, ROE, ROIC
- Growth: revenue growth (YoY, QoQ), earnings growth, guidance vs. consensus
- Balance sheet: debt/equity, current ratio, free cash flow, cash position

Structure your response as:
1. **Valuation Assessment** — overvalued / fairly valued / undervalued, with reasoning
   Include your confidence: "High confidence" (multiple metrics agree) vs "Low confidence" (mixed signals)
2. **Key Metrics** — table format preferred: Metric | Value | vs Peers | vs Historical
3. **Growth Analysis** — trajectory, sustainability, and what could accelerate or decelerate it
4. **Financial Health** — balance sheet strength, cash flow quality, debt sustainability
5. **Thesis Risks** — what specific data points would invalidate your assessment?

Quality standards:
- BAD: "The stock looks expensive"
- GOOD: "At P/E 52 vs sector median 28 and 5-year average 35, the stock trades at a 48% premium to peers — justified only if >30% earnings growth sustains for 3+ years"
- BAD: "Revenue growth is strong"
- GOOD: "Revenue grew 15% YoY but decelerated from 22% in Q2 — the trend matters more than the absolute number"

Do NOT:
- Give buy/sell/hold recommendations — present the analysis, let the synthesis decide
- Ignore inconvenient data — if one metric screams overvalued while others say fair, say so explicitly
- Use vague language when numbers are available

DO:
- Quantify everything possible
- Flag when you're working with incomplete data: "Cannot assess FCF — no cash flow statement available"
- Note when metrics conflict: "P/E suggests overvalued but PEG of 1.2 suggests fair value given growth"
"""

    def build_query(self, query: str, memory_results: dict) -> str:
        """Build a financially-focused prompt using memory context.

        The analyst's prompt emphasizes numerical data and metrics
        from past research. It asks Claude to reason about the numbers,
        not to search for new information.
        """
        memory_context = self._format_memory_context(memory_results)

        return f"""**DATA BLOCK START** (your analysis must be grounded in this data)

**Available Data (from past research and known facts):**
{memory_context}

**DATA BLOCK END**

**Question:** {query}

Provide a thorough financial analysis using the data above.
Focus on valuation metrics, growth trends, and fundamental health.
If key data is missing, state what you would need to complete the analysis.
Your response MUST reference specific numbers from the DATA BLOCK — avoid vague statements like "strong growth."
"""

    def extract_sources(self, query: str) -> list[Source]:
        """The analyst uses memory as its source, not web search.

        Since the analyst reasons over existing data rather than
        searching for new data, its sources come from memory.
        If Tavily is available, it does a lightweight search for
        recent financial data to supplement memory.

        PYTHON CONCEPT — conditional logic with early returns:
        Check the simplest case first (no tavily → return empty),
        then handle the more complex case. Keeps code flat.
        """
        sources = []

        # Optionally supplement with targeted financial search
        if self.tavily:
            try:
                financial_sources = self.tavily.search_as_sources(
                    query=f"{query} financial metrics valuation",
                    max_results=3,
                    search_depth="basic",
                    topic="finance",
                )
                sources.extend(financial_sources)
                # PYTHON CONCEPT — .extend() vs .append():
                # .append(item) adds ONE item to the list
                # .extend(list) adds ALL items from another list
                # TS equivalent: sources.push(...financialSources) (spread)
            except Exception as e:
                logger.warning("[analyst] Financial source search failed: %s", e)

        return sources
