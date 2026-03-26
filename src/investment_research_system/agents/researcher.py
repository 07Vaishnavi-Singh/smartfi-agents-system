"""Researcher agent — gathers and synthesizes information.

This is the primary information-gathering agent. It:
1. Searches memory for past research on the topic
2. Searches the web via Tavily for fresh data
3. Combines everything and asks Claude to synthesize

PYTHON CONCEPT — INHERITANCE:
ResearcherAgent inherits from BaseAgent. It gets run(), _estimate_confidence(),
and _format_memory_context() for free. It only implements the 4 abstract pieces:
    - name (property)
    - system_prompt (property)
    - build_query() (how to frame the prompt)
    - extract_sources() (where to get sources)

TS equivalent: class ResearcherAgent extends BaseAgent { ... }
Rust equivalent: impl BaseAgent for ResearcherAgent { ... }
"""

import logging

from investment_research_system.agents.base import BaseAgent
from investment_research_system.models.schemas import Source, SourceType

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Information gathering and synthesis agent.

    The researcher is the most "tool-heavy" agent — it actively searches
    the web and memory, then synthesizes findings. Other agents (analyst,
    risk, sentiment) mostly reason over data the researcher provides.
    """

    @property
    def name(self) -> str:
        return "researcher"

    @property
    def system_prompt(self) -> str:
        # PYTHON CONCEPT — triple-quoted strings for multi-line:
        # """...""" preserves newlines and indentation.
        # TS equivalent: template literals `...`
        # Rust equivalent: raw strings r#"..."#
        return """You are the RESEARCHER agent in a multi-agent investment research pipeline.

Your position: You run FIRST. Your output feeds into the Analyst, Sentiment, and Risk agents.
This means your job is to gather and present RAW FACTS — not to interpret or recommend.
Other agents handle interpretation. You handle evidence.

Your responsibilities:
- Gather and synthesize information from web search results AND memory context provided below
- Present facts with specific numbers, dates, and sources — never round or approximate when exact figures are available
- Distinguish confirmed data from estimates/projections/speculation using explicit labels:
  [CONFIRMED] — verified from official filings, earnings reports, press releases
  [ESTIMATED] — analyst estimates, consensus projections
  [UNVERIFIED] — single-source claims, social media, rumors
- Cite sources inline: "Revenue was $35.1B (Source: Q3 2024 10-Q filing)"
- Flag data staleness: if a data point is >3 months old, note "[as of DATE — may be outdated]"

Structure your response as:
1. **Key Findings** — 3-5 bullet points, each tagged [CONFIRMED/ESTIMATED/UNVERIFIED]
2. **Detailed Analysis** — deeper context, connect the dots between data points
3. **Data Points** — specific numbers, dates, metrics in a scannable format
4. **Information Gaps** — what you couldn't find or verify, and WHY it matters

Quality standards:
- BAD: "The company has strong revenue growth"
- GOOD: "Revenue grew 15% YoY to $35.1B in Q3 2024 [CONFIRMED], above the sector average of 8%"
- BAD: "Analysts are optimistic"
- GOOD: "12 of 15 analysts rate BUY with a median price target of $142 [ESTIMATED, as of Jan 2025]"

Do NOT:
- Interpret the data (that's the Analyst's job)
- Assess risk (that's the Risk Assessor's job)
- Gauge sentiment (that's the Sentiment agent's job)
- Give buy/sell opinions — you are an evidence-gathering tool

DO:
- Research any valid financial topic, including "should I invest in X" — gather the relevant evidence
- Present both bullish and bearish data points when they exist
- Be explicit about what you DON'T know — gaps are as valuable as findings"""

    def build_query(self, query: str, memory_results: dict) -> str:
        """Build a research prompt that includes memory context and web results.

        The researcher's prompt has 3 sections:
        1. The user's question
        2. What we already know (from memory)
        3. Instructions on what to focus on

        PYTHON CONCEPT — multi-line f-strings:
        You can use f-strings inside triple quotes for multi-line templates.
        Variables are embedded with {variable_name}.
        TS equivalent: template literals with ${variable}
        """
        memory_context = self._format_memory_context(memory_results)

        # Build web search context if Tavily is available
        web_context = "No web search results available."
        if self.tavily:
            try:
                web_results = self.tavily.search(query, max_results=5)
                if web_results["results"]:
                    web_parts = ["=== Recent Web Results ==="]
                    # PYTHON CONCEPT — enumerate():
                    # Gives you (index, item) pairs. Like .forEach((item, i) => ...) in TS
                    # but the index comes FIRST in Python.
                    for i, result in enumerate(web_results["results"], start=1):
                        web_parts.append(
                            f"{i}. [{result['title']}]({result['url']})\n"
                            f"   {result['content'][:200]}"
                        )
                    web_context = "\n".join(web_parts)

                    # Include Tavily's AI summary if available
                    if web_results.get("answer"):
                        web_context += f"\n\nWeb Summary: {web_results['answer']}"

            except Exception as e:
                logger.warning("[researcher] Tavily search failed: %s", e)
                web_context = "Web search unavailable — analyze based on existing knowledge."

        return f"""**DATA BLOCK START** (read carefully — your analysis must use this data)

**Existing Knowledge (from past research):**
{memory_context}

**Fresh Web Data:**
{web_context}

**DATA BLOCK END**

**Question:** {query}

Using the data above, provide a comprehensive research synthesis.
If existing knowledge conflicts with fresh data, highlight the discrepancy.
Your response MUST reference specific data points from the DATA BLOCK above."""

    def extract_sources(self, query: str) -> list[Source]:
        """Gather sources from Tavily web search.

        The researcher is the primary web searcher — it uses
        search_as_sources() to get Source objects directly.

        PYTHON CONCEPT — early return:
        If Tavily isn't available, return empty list immediately.
        Don't nest the rest of the logic inside an if block.
        This keeps code flat and readable.
        TS/Rust: same pattern, same benefit.
        """
        if not self.tavily:
            return []

        try:
            return self.tavily.search_as_sources(
                query=query,
                max_results=5,
                topic="finance",
            )
        except Exception as e:
            logger.warning("[researcher] Source extraction failed: %s", e)
            return []
