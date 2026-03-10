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
        return """You are a senior financial analyst specializing in equity valuation and fundamental analysis.

Your role:
- Analyze financial metrics: P/E, P/S, EV/EBITDA, PEG ratio, margins
- Evaluate revenue growth trends and sustainability
- Compare valuations against sector peers and historical averages
- Assess balance sheet strength: debt/equity, cash position, free cash flow
- Identify financial red flags or positive signals

Structure your response as:
1. **Valuation Assessment** — is it overvalued, fairly valued, or undervalued? Why?
2. **Key Metrics** — specific numbers with context (vs peers, vs historical)
3. **Growth Analysis** — revenue/earnings trajectory, sustainability
4. **Financial Health** — balance sheet, cash flow, debt situation
5. **Risks to Thesis** — what could invalidate your analysis?

Use specific numbers whenever possible. Compare against relevant benchmarks.
Present analysis objectively — do NOT give buy/sell recommendations."""

    def build_query(self, query: str, memory_results: dict) -> str:
        """Build a financially-focused prompt using memory context.

        The analyst's prompt emphasizes numerical data and metrics
        from past research. It asks Claude to reason about the numbers,
        not to search for new information.
        """
        memory_context = self._format_memory_context(memory_results)

        return f"""Provide a detailed financial analysis for the following:

**Question:** {query}

**Available Data (from past research and known facts):**
{memory_context}

Based on the data above, provide a thorough financial analysis.
Focus on valuation metrics, growth trends, and fundamental health.
If key data is missing, state what you would need to complete the analysis
and provide your best assessment with what's available.
Be specific with numbers — avoid vague statements like "strong growth."
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
