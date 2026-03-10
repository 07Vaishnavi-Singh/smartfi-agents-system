"""Risk assessor agent — identifies risks, downsides, and red flags.

Every other agent has an optimism bias — the researcher reports what exists,
the analyst evaluates metrics, the sentiment agent reads the mood.
The risk assessor's job is to be the DELIBERATE PESSIMIST.

This is a well-known technique in decision-making called "pre-mortem":
Instead of asking "will this work?", you ask "assume this failed — why?"
It forces you to surface risks you'd otherwise overlook.

PYTHON CONCEPT — SAME BASE, DIFFERENT BEHAVIOR:
This class has the exact same structure as ResearcherAgent and AnalystAgent.
Same __init__, same run() pipeline (inherited from BaseAgent).
The ONLY differences are:
    - system_prompt (tells Claude to think about risks)
    - build_query() (frames the question around downsides)
    - extract_sources() (searches for risk-related news)
This is the power of inheritance + polymorphism.
"""

import logging

from investment_research_system.agents.base import BaseAgent
from investment_research_system.models.schemas import Source

logger = logging.getLogger(__name__)


class RiskAssessorAgent(BaseAgent):
    """Risk identification and assessment agent.

    Deliberately looks for what could go wrong. Searches for
    regulatory risks, competitive threats, financial red flags,
    and macro headwinds that other agents might overlook.
    """

    @property
    def name(self) -> str:
        return "risk_assessor"

    @property
    def system_prompt(self) -> str:
        return """You are a risk analyst specializing in investment risk assessment.

Your role is to be the DEVIL'S ADVOCATE. While other analysts focus on
opportunities, you focus on what could go WRONG.

Your role:
- Identify regulatory, competitive, financial, and macro risks
- Assess severity (low/medium/high) and probability for each risk
- Look for red flags others might miss: insider selling, accounting changes,
  customer concentration, supply chain dependencies
- Consider tail risks — unlikely but catastrophic scenarios
- Evaluate how well the company can withstand adverse conditions

Structure your response as:
1. **Critical Risks** — high severity, could significantly impact the investment
2. **Moderate Risks** — worth monitoring, could affect returns
3. **Macro/Sector Risks** — broader risks that affect the whole sector
4. **Risk Mitigation** — what the company is doing to address these risks
5. **Overall Risk Rating** — Low / Medium / High / Very High with justification

Be specific — "regulatory risk" is too vague. Say "US export controls on
AI chips to China could reduce NVIDIA's datacenter revenue by 15-20%."
Always quantify impact where possible."""

    def build_query(self, query: str, memory_results: dict) -> str:
        """Build a risk-focused prompt.

        The risk assessor's prompt explicitly asks Claude to think
        about what could go wrong, not what's going right.
        This counterbalances the natural optimism bias in research.
        """
        memory_context = self._format_memory_context(memory_results)

        return f"""Perform a thorough risk assessment for the following:

**Question:** {query}

**Available Data (from past research and known facts):**
{memory_context}

IMPORTANT: Your job is to find what could go WRONG, not what's going right.
Think about:
- Regulatory threats (government action, export controls, antitrust)
- Competitive threats (who is catching up, market share shifts)
- Financial risks (overvaluation, debt, margin pressure)
- Operational risks (supply chain, key person dependency, execution)
- Macro risks (recession, interest rates, geopolitical events)
- Tail risks (unlikely but catastrophic — black swan scenarios)

For each risk, estimate severity (low/medium/high) and probability.
Quantify potential impact in dollar terms or percentage where possible.
"""

    def extract_sources(self, query: str) -> list[Source]:
        """Search for risk-related news and regulatory updates.

        The risk assessor does a targeted web search specifically
        for negative news, risks, and regulatory actions.
        """
        if not self.tavily:
            return []

        try:
            return self.tavily.search_as_sources(
                query=f"{query} risks regulatory concerns threats",
                max_results=3,
                topic="news",
                time_range="month",
                # ^ Look at last month's news — risks often emerge
                #   from recent regulatory actions or competitive moves.
            )
        except Exception as e:
            logger.warning("[risk_assessor] Risk source search failed: %s", e)
            return []
