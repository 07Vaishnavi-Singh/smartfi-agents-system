"""Sentiment agent — gauges market mood and opinion.

The other 3 agents deal with FACTS and NUMBERS.
The sentiment agent deals with FEELINGS and OPINIONS.

Why this matters: A stock can have perfect fundamentals (analyst says great)
and no obvious risks (risk assessor says fine), but if market sentiment
turns bearish, the stock price drops anyway. Markets are driven by
people, and people are driven by emotion.

Real example:
  - Meta (Facebook) in late 2022: fundamentals were solid, but sentiment
    was extremely negative ("metaverse is a waste"). Stock dropped 75%.
  - One year later, sentiment flipped bullish ("year of efficiency").
    Stock tripled. The fundamentals didn't change THAT much — sentiment did.

PYTHON CONCEPT — REUSE THROUGH INHERITANCE:
This is the 4th agent. Notice how little code each agent needs.
All the heavy lifting (run pipeline, token tracking, memory integration)
lives in BaseAgent. Each subclass is just ~80 lines of customization.
This is the payoff of good base class design.
"""

import logging

from investment_research_system.agents.base import BaseAgent
from investment_research_system.models.schemas import Source

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """Market sentiment and opinion analysis agent.

    Focuses on:
    - Analyst ratings and consensus (buy/hold/sell)
    - News tone (bullish/bearish coverage)
    - Insider activity (buying vs selling)
    - Market momentum and trend signals
    """

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def system_prompt(self) -> str:
        return """You are a market sentiment analyst specializing in gauging investor mood and market psychology.

Your role:
- Assess overall market sentiment: bullish, bearish, or neutral
- Analyze analyst consensus: how many buy/hold/sell ratings?
- Evaluate news tone: is coverage positive, negative, or mixed?
- Check insider activity: are executives buying or selling their own stock?
- Identify sentiment shifts: has opinion changed recently? Why?
- Detect hype vs substance: is positive sentiment based on fundamentals or FOMO?

Structure your response as:
1. **Overall Sentiment** — Bullish / Bearish / Neutral with a score (-1.0 to +1.0)
2. **Analyst Consensus** — ratings breakdown, price targets, recent changes
3. **News & Media Tone** — what's the narrative? Is it shifting?
4. **Insider Activity** — are insiders buying or selling? What does it signal?
5. **Sentiment vs Fundamentals** — does the mood match the data? Any disconnect?

Flag any divergence between sentiment and fundamentals — that's often
where investment opportunities (or traps) hide.
Do NOT give investment advice — present sentiment analysis only."""

    def build_query(self, query: str, memory_results: dict) -> str:
        """Build a sentiment-focused prompt.

        The sentiment agent's prompt specifically asks Claude to
        analyze mood, opinions, and market psychology — not financials.
        """
        memory_context = self._format_memory_context(memory_results)

        return f"""Analyze the market sentiment for the following:

**Question:** {query}

**Available Data (from past research and known facts):**
{memory_context}

Focus on SENTIMENT — how people FEEL about this investment, not the numbers.
Analyze:
- What is the prevailing mood among investors and analysts?
- Has sentiment shifted recently? What triggered the shift?
- Is there a gap between sentiment and fundamentals?
- Are there signs of excessive optimism (hype) or pessimism (fear)?
- What are analysts saying? What's the consensus view?

Provide a sentiment score from -1.0 (extremely bearish) to +1.0 (extremely bullish).
"""

    def extract_sources(self, query: str) -> list[Source]:
        """Search for recent news to gauge sentiment.

        The sentiment agent searches NEWS specifically — recent articles,
        analyst notes, and opinion pieces reveal market mood better than
        financial data pages.
        """
        if not self.tavily:
            return []

        try:
            return self.tavily.search_as_sources(
                query=f"{query} analyst opinion sentiment outlook",
                max_results=5,
                topic="news",
                time_range="week",
                # ^ Last week's news — sentiment changes fast.
                #   A month-old article about sentiment is stale.
            )
        except Exception as e:
            logger.warning("[sentiment] Sentiment source search failed: %s", e)
            return []
