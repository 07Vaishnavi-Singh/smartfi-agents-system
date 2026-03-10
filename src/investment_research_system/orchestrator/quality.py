"""Quality checker for the final research report.

Before returning a report to the user, we check:
- How many agents actually responded?
- What's the average confidence?
- Are there unresolved conflicts?

Based on this, we assign a quality grade and add disclaimers
if the report is incomplete. This is the "honest reporting" layer —
we don't pretend a report is great when 2 agents failed.

PYTHON CONCEPT — DATACLASS:
We use a simple dataclass to hold the quality assessment result.
A dataclass is like a Pydantic model but lighter — no validation,
just a convenient container for related data.
TS equivalent: a plain interface/type
Rust equivalent: a simple struct
"""

import logging
from dataclasses import dataclass

from investment_research_system.models.schemas import AgentResponse

logger = logging.getLogger(__name__)

# How many agents we expect to respond
EXPECTED_AGENT_COUNT = 4


@dataclass
class QualityAssessment:
    """Result of the quality check.

    PYTHON CONCEPT — @dataclass:
    Automatically generates __init__, __repr__, __eq__ for you.
    You just define the fields. No boilerplate.

    TS equivalent: interface QualityAssessment { grade: string; ... }
    Rust equivalent: #[derive(Debug)] struct QualityAssessment { ... }
    """

    grade: str                    # "HIGH", "MEDIUM", "LOW", "MINIMAL", "FAILED"
    agent_count: int              # how many agents responded
    expected_count: int           # how many we expected (4)
    average_confidence: float     # mean confidence across responses
    conflict_count: int           # number of detected conflicts
    disclaimers: list[str]        # warnings to include in the report
    passed: bool                  # is this report good enough to return?


def assess_quality(
    responses: list[AgentResponse],
    conflicts: list[dict],
    failed_agents: list[str] | None = None,
) -> QualityAssessment:
    """Evaluate the quality of the research output.

    Args:
        responses: Agent responses we received.
        conflicts: Conflicts detected between agents.
        failed_agents: Names of agents that failed (for disclaimers).

    Returns:
        QualityAssessment with grade, disclaimers, and pass/fail.
    """
    failed_agents = failed_agents or []
    agent_count = len(responses)
    disclaimers = []

    # Calculate average confidence
    if responses:
        average_confidence = sum(r.confidence for r in responses) / len(responses)
    else:
        average_confidence = 0.0

    # Determine grade based on agent count
    if agent_count >= 4:
        grade = "HIGH"
    elif agent_count == 3:
        grade = "MEDIUM"
        disclaimers.append(
            f"1 agent unavailable ({', '.join(failed_agents) or 'unknown'}). "
            "Report may be missing some perspective."
        )
    elif agent_count == 2:
        grade = "LOW"
        disclaimers.append(
            f"Only {agent_count} of {EXPECTED_AGENT_COUNT} agents responded. "
            "This report has significant gaps."
        )
    elif agent_count == 1:
        grade = "MINIMAL"
        disclaimers.append(
            "Only 1 agent responded. This report should not be relied upon "
            "for decision-making."
        )
    else:
        grade = "FAILED"
        disclaimers.append("No agents were able to produce analysis.")

    # Downgrade if average confidence is low
    if average_confidence < 0.4 and grade in ("HIGH", "MEDIUM"):
        grade = "LOW"
        disclaimers.append(
            f"Average confidence is low ({average_confidence:.2f}). "
            "Analysis may lack sufficient data."
        )

    # Note conflicts
    high_severity_conflicts = [c for c in conflicts if c.get("severity") == "high"]
    if high_severity_conflicts:
        disclaimers.append(
            f"{len(high_severity_conflicts)} high-severity disagreement(s) "
            "between agents. Review the conflicting viewpoints carefully."
        )

    # Determine pass/fail
    passed = grade in ("HIGH", "MEDIUM", "LOW")
    # MINIMAL and FAILED don't pass — but we still return what we have

    assessment = QualityAssessment(
        grade=grade,
        agent_count=agent_count,
        expected_count=EXPECTED_AGENT_COUNT,
        average_confidence=round(average_confidence, 2),
        conflict_count=len(conflicts),
        disclaimers=disclaimers,
        passed=passed,
    )

    logger.info(
        "Quality assessment: grade=%s, agents=%d/%d, confidence=%.2f, conflicts=%d",
        grade, agent_count, EXPECTED_AGENT_COUNT, average_confidence, len(conflicts),
    )

    return assessment


def format_quality_summary(assessment: QualityAssessment) -> str:
    """Format the quality assessment into a readable string for the report."""
    parts = [
        f"Report Quality: {assessment.grade}",
        f"Agents responded: {assessment.agent_count}/{assessment.expected_count}",
        f"Average confidence: {assessment.average_confidence:.2f}",
        f"Conflicts detected: {assessment.conflict_count}",
    ]

    if assessment.disclaimers:
        parts.append("\nDisclaimers:")
        for disclaimer in assessment.disclaimers:
            parts.append(f"  - {disclaimer}")

    return "\n".join(parts)
