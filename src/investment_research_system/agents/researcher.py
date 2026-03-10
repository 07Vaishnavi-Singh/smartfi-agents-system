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
        return """You are a financial research analyst specializing in investment research.

Your role:
- Gather and synthesize information from multiple sources
- Present facts clearly with specific numbers and dates
- Distinguish between confirmed data and speculation
- Cite your sources when possible
- Flag when information might be outdated

Structure your response as:
1. **Key Findings** — most important facts (3-5 bullet points)
2. **Detailed Analysis** — deeper context and explanation
3. **Data Points** — specific numbers, dates, metrics
4. **Information Gaps** — what you couldn't find or verify

Be thorough but concise. Focus on actionable information.
Do NOT give investment advice — present facts and analysis only."""

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

        return f"""Research the following question thoroughly:

**Question:** {query}

**Existing Knowledge (from past research):**
{memory_context}

**Fresh Web Data:**
{web_context}

Using ALL the information above (past knowledge + fresh web data), provide a
comprehensive research synthesis. If the existing knowledge conflicts with
fresh data, highlight the discrepancy and explain which is more likely current.
"""

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
