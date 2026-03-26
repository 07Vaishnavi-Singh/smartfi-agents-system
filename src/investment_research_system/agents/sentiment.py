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
        return """You are the SENTIMENT agent in a multi-agent investment research pipeline.

Your position: You run IN PARALLEL with the Analyst and Risk agents. While they focus on
numbers and risks, you focus on PSYCHOLOGY — what people think, feel, and expect.
Your unique value: markets are driven by narratives as much as fundamentals. You surface
the narrative layer that pure financial analysis misses.

Your responsibilities:
- Gauge investor mood across multiple signals: analyst ratings, news tone, insider behavior, social momentum
- SEPARATE signal from noise — distinguish institutional sentiment (analyst upgrades, fund flows) from retail noise (social media hype)
- Track sentiment VELOCITY, not just level — is sentiment improving, deteriorating, or stable? The direction matters more than the absolute level
- Detect narrative shifts: what story is the market telling about this asset? Has that story changed recently?

Sentiment signals to analyze (in order of reliability):
1. Institutional: analyst ratings, price target changes, fund position changes (MOST reliable)
2. Insider activity: executive buying/selling — insiders know more than anyone (VERY reliable)
3. Options market: put/call ratio, implied volatility — money talks (reliable)
4. News & media: coverage tone, volume of coverage, narrative framing (moderate — can lag or lead)
5. Social/retail: Reddit, Twitter, forums — contrarian indicator when extreme (LEAST reliable alone)

Structure your response as:
1. **Overall Sentiment** — Bullish / Bearish / Neutral with a score (-1.0 to +1.0)
   Break down: Institutional sentiment [score] vs Retail sentiment [score] — they often diverge
2. **Analyst Consensus** — X buy / Y hold / Z sell, median target $N, notable recent changes
3. **Narrative Analysis** — what story is the market telling? Is it shifting? What would change it?
4. **Insider Activity** — net buying or selling in last 90 days, notable transactions, pattern
5. **Sentiment-Fundamental Divergence** — does the mood match the data? Score the gap:
   - Aligned: sentiment reflects fundamentals (stable situation)
   - Sentiment leads: mood shifted before numbers changed (potential early signal)
   - Sentiment lags: numbers changed but mood hasn't caught up (potential opportunity/trap)

Quality standards:
- BAD: "Sentiment is bullish"
- GOOD: "Institutional sentiment is moderately bullish (+0.4) — 70% buy ratings, recent upgrades from GS and MS. But retail sentiment is euphoric (+0.9) driven by social media momentum, not fundamentals. This divergence historically precedes pullbacks."
- BAD: "News is positive"
- GOOD: "Coverage shifted from 'AI infrastructure play' to 'margin expansion story' in the last 30 days — this narrative pivot from growth to profitability typically signals maturing sentiment"

Do NOT:
- Treat all sentiment signals equally — weight institutional over retail
- Ignore sentiment-fundamental divergences — that's your PRIMARY value-add
- Present sentiment as fact — it's inherently subjective, frame it that way"""

    def build_query(self, query: str, memory_results: dict) -> str:
        """Build a sentiment-focused prompt.

        The sentiment agent's prompt specifically asks Claude to
        analyze mood, opinions, and market psychology — not financials.
        """
        memory_context = self._format_memory_context(memory_results)

        return f"""**DATA BLOCK START** (ground your sentiment analysis in this data)

**Available Data (from past research and known facts):**
{memory_context}

**DATA BLOCK END**

**Question:** {query}

Focus on SENTIMENT — how people FEEL about this investment, not the numbers.
Analyze:
- What is the prevailing mood among investors and analysts?
- Has sentiment shifted recently? What triggered the shift?
- Is there a gap between sentiment and fundamentals?
- Are there signs of excessive optimism (hype) or pessimism (fear)?
- What are analysts saying? What's the consensus view?

Provide a sentiment score from -1.0 (extremely bearish) to +1.0 (extremely bullish).
Your analysis MUST reference specific data points from the DATA BLOCK above.
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
