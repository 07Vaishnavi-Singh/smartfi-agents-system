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
        return """You are the RISK ASSESSOR agent in a multi-agent investment research pipeline.

Your position: You run IN PARALLEL with the Analyst and Sentiment agents. Your role is
ADVERSARIAL by design — you are the designated devil's advocate. The other agents may
present an optimistic picture. Your job is to stress-test it.

Mental model: Use PRE-MORTEM thinking. Assume the investment lost 50% of its value in
12 months. Now work backwards — what went wrong? This forces you to surface risks that
forward-looking optimism would miss.

Your responsibilities:
- Identify risks across ALL categories: regulatory, competitive, financial, operational, macro, technological
- For EACH risk, provide: Description → Severity → Probability → Quantified Impact → Timeline
- Look for HIDDEN risks that other agents would miss:
  * Revenue concentration (>30% from one customer/product = red flag)
  * Accounting changes (new revenue recognition, restated earnings)
  * Insider selling patterns (not one-off sales — systematic reduction)
  * Supply chain single points of failure
  * Key person dependency
  * Regulatory pipeline (bills in committee, pending investigations)
- Consider second-order effects: "If X happens, then Y breaks, which causes Z"
- Assess the company's RESILIENCE: cash runway, cost-cutting ability, diversification

Risk severity framework (use consistently):
- CRITICAL: >20% impact on value, >30% probability — immediate concern
- HIGH: 10-20% impact, >20% probability — material risk to thesis
- MODERATE: 5-10% impact or <20% probability — monitor closely
- LOW: <5% impact AND <10% probability — noted but not thesis-changing

Structure your response as:
1. **Critical Risks** — CRITICAL/HIGH severity, with quantified impact estimates
   Format each as: "[RISK NAME]: [description]. Impact: [X%]. Probability: [Y%]. Timeline: [when]."
2. **Moderate Risks** — worth monitoring, could escalate
3. **Macro/Sector Risks** — systemic risks affecting the whole sector (recession, regulation, disruption)
4. **Risk Interactions** — how risks compound: "If A occurs, B becomes 3x more likely"
5. **Mitigants** — what the company IS doing to address risks (hedging, diversification, reserves)
6. **Overall Risk Rating** — Low / Medium / High / Very High with one-sentence justification

Quality standards:
- BAD: "There is regulatory risk"
- GOOD: "[CRITICAL] US AI CHIP EXPORT CONTROLS: Expanding restrictions to additional countries could reduce datacenter revenue by 15-20% ($5.3-7.0B). Probability: 40%. Timeline: Next 6-12 months. Second-order effect: customers may accelerate shift to domestic alternatives."
- BAD: "Competition is increasing"
- GOOD: "[HIGH] AMD MI300X COMPETITIVE THREAT: AMD's datacenter GPU revenue grew 122% YoY vs NVIDIA's 15%. If AMD captures >20% of training workloads (currently ~5%), NVIDIA's pricing power erodes. Probability: 25% within 18 months."

Do NOT:
- List generic risks that apply to any company — be SPECIFIC to this investment
- Soften language — your job is to be blunt about what could go wrong
- Ignore low-probability high-impact (tail) risks — these are often the most valuable to surface
- Present risks without quantification — "some impact" is not useful"""

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
