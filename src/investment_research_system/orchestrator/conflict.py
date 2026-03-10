"""Conflict detection between agent responses.

When 4 agents analyze the same question, they WILL disagree sometimes.
That's not a bug — it's a feature. Disagreement surfaces nuance.

Example conflict:
  Analyst: "NVIDIA is fairly valued given 94% revenue growth"
  Risk:    "NVIDIA is overvalued — P/E of 52 is 2x sector average"

Both are technically correct — they're just weighing different factors.
The conflict detector finds these disagreements so the final report
can present both sides instead of pretending everyone agrees.

PYTHON CONCEPT — PURE FUNCTIONS:
Most functions in this file are "pure" — they take inputs, return outputs,
and don't modify anything. No self, no state, no side effects.
This makes them easy to test and reason about.
TS equivalent: standalone utility functions (not class methods)
Rust equivalent: free functions (fn detect_conflicts(...) -> Vec<Conflict>)
"""

import logging

from investment_research_system.models.schemas import AgentResponse

logger = logging.getLogger(__name__)

# Keywords that signal a particular stance on valuation/outlook
# Used to detect when agents disagree on the same topic
BULLISH_SIGNALS = [
    "undervalued", "fairly valued", "strong growth", "buy",
    "upside", "outperform", "bullish", "opportunity",
    "attractive valuation", "positive outlook",
]
BEARISH_SIGNALS = [
    "overvalued", "expensive", "downside", "sell",
    "underperform", "bearish", "risk", "threat",
    "stretched valuation", "negative outlook", "concern",
]


def detect_conflicts(responses: list[AgentResponse]) -> list[dict]:
    """Find disagreements between agent responses.

    Compares every pair of agents and checks if they have
    opposing stances on the same topic (valuation, outlook, etc.).

    Args:
        responses: List of AgentResponse from all agents that responded.

    Returns:
        List of conflict dicts, each with:
        - agents_involved: which agents disagree
        - topic: what they disagree about
        - disagreement: description of the conflict
        - severity: "low", "medium", "high"

    PYTHON CONCEPT — itertools.combinations:
    We don't use it here for clarity, but it generates all pairs
    from a list without repeats. We use a manual nested loop instead
    so it's easier to follow.
    """
    if len(responses) < 2:
        return []

    conflicts = []

    # Compare every pair of agents
    # PYTHON CONCEPT — nested loops with index slicing:
    # for i, a in enumerate(responses):
    #     for b in responses[i+1:]:
    # This avoids comparing A-B AND B-A (duplicates)
    # TS equivalent: for (let i = 0; i < arr.length; i++)
    #                  for (let j = i+1; j < arr.length; j++)
    for i, response_a in enumerate(responses):
        for response_b in responses[i + 1:]:
            pair_conflicts = _compare_pair(response_a, response_b)
            conflicts.extend(pair_conflicts)

    if conflicts:
        logger.info(
            "Detected %d conflict(s) between agents",
            len(conflicts),
        )

    return conflicts


def _compare_pair(a: AgentResponse, b: AgentResponse) -> list[dict]:
    """Compare two agent responses for disagreements.

    Checks for:
    1. Sentiment mismatch (one bullish, other bearish)
    2. Confidence divergence (one very confident, other not)

    PYTHON CONCEPT — leading underscore:
    _compare_pair is "private" — only used inside this module.
    detect_conflicts() is the public API.
    """
    conflicts = []

    # Check 1: Opposing sentiment (bullish vs bearish)
    a_content = a.content.lower()
    b_content = b.content.lower()

    a_bullish = sum(1 for signal in BULLISH_SIGNALS if signal in a_content)
    a_bearish = sum(1 for signal in BEARISH_SIGNALS if signal in a_content)
    b_bullish = sum(1 for signal in BULLISH_SIGNALS if signal in b_content)
    b_bearish = sum(1 for signal in BEARISH_SIGNALS if signal in b_content)
    # PYTHON CONCEPT — generator expression with sum():
    # sum(1 for x in list if condition) counts how many items match.
    # TS equivalent: list.filter(x => condition).length

    a_stance = _get_stance(a_bullish, a_bearish)
    b_stance = _get_stance(b_bullish, b_bearish)

    if a_stance and b_stance and a_stance != b_stance:
        severity = _assess_severity(a.confidence, b.confidence, a_stance, b_stance)
        conflicts.append({
            "agents_involved": [a.agent_name, b.agent_name],
            "topic": "outlook",
            "disagreement": (
                f"{a.agent_name} leans {a_stance} "
                f"(bullish signals: {a_bullish}, bearish: {a_bearish}) "
                f"while {b.agent_name} leans {b_stance} "
                f"(bullish signals: {b_bullish}, bearish: {b_bearish})"
            ),
            "severity": severity,
        })

    # Check 2: Large confidence gap on the same question
    confidence_gap = abs(a.confidence - b.confidence)
    if confidence_gap > 0.3:
        conflicts.append({
            "agents_involved": [a.agent_name, b.agent_name],
            "topic": "confidence",
            "disagreement": (
                f"{a.agent_name} confidence: {a.confidence:.2f} vs "
                f"{b.agent_name} confidence: {b.confidence:.2f} "
                f"(gap: {confidence_gap:.2f})"
            ),
            "severity": "medium" if confidence_gap > 0.4 else "low",
        })

    return conflicts


def _get_stance(bullish_count: int, bearish_count: int) -> str | None:
    """Determine if an agent leans bullish, bearish, or neutral.

    Returns None if signals are too weak to determine a stance.
    """
    total = bullish_count + bearish_count
    if total < 2:
        # Not enough signals to determine stance
        return None

    if bullish_count > bearish_count * 1.5:
        return "bullish"
    elif bearish_count > bullish_count * 1.5:
        return "bearish"

    return None  # mixed signals — no clear stance


def _assess_severity(
    conf_a: float,
    conf_b: float,
    stance_a: str,
    stance_b: str,
) -> str:
    """Assess how serious a conflict is.

    High severity = both agents are confident AND directly oppose each other.
    Low severity = one or both have low confidence (they're not sure themselves).
    """
    avg_confidence = (conf_a + conf_b) / 2

    if avg_confidence > 0.8:
        return "high"  # both are confident and disagree — serious
    elif avg_confidence > 0.6:
        return "medium"
    else:
        return "low"  # low confidence — disagreement is less meaningful


def format_conflicts_summary(conflicts: list[dict]) -> str:
    """Format conflicts into a readable string for the final report.

    Called by the report builder to include conflict information
    in the user-facing output.
    """
    if not conflicts:
        return "No significant disagreements between agents."

    parts = [f"Found {len(conflicts)} disagreement(s) between agents:\n"]

    for i, conflict in enumerate(conflicts, start=1):
        agents = " vs ".join(conflict["agents_involved"])
        parts.append(
            f"{i}. [{conflict['severity'].upper()}] {agents} — "
            f"{conflict['topic']}: {conflict['disagreement']}"
        )

    return "\n".join(parts)
