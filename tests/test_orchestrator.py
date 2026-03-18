"""End-to-end test for the LangGraph orchestrator.

Run: PYTHONPATH=src uv run pytest tests/test_orchestrator.py -v -s

This runs ALL 4 agents through the full pipeline:
query → agents (parallel) → conflict detection → quality check → report
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

skip_no_keys = pytest.mark.skipif(
    not GOOGLE_KEY or not TAVILY_KEY,
    reason="GOOGLE_API_KEY and TAVILY_API_KEY required for live test",
)


# --- Helpers ---
def make_mock_memory():
    """Create a mock MemoryManager with sample data."""
    from unittest.mock import AsyncMock, MagicMock

    mock = MagicMock()
    mock.parallel_search = AsyncMock(return_value={
        "long_term": [
            {"text": "NVIDIA revenue was $35.1B in Q3 2024, up 94% YoY. Operating margin was 57%.", "score": 0.90},
            {"text": "AMD revenue was $6.8B in Q3 2024. AI chip segment grew 122%.", "score": 0.80},
        ],
        "semantic": [
            {"memory": "NVIDIA P/E ratio is approximately 52"},
            {"memory": "NVIDIA top 5 customers account for 40% of revenue"},
            {"memory": "US export controls restrict AI chip sales to China"},
        ],
    })
    mock.update_agent_status = AsyncMock()
    mock.store_research = AsyncMock()
    mock.create_session = AsyncMock()
    mock.end_session = AsyncMock()
    return mock


def make_agents():
    """Create all 4 agents with Gemini + mock memory."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from investment_research_system.agents.analyst import AnalystAgent
    from investment_research_system.agents.researcher import ResearcherAgent
    from investment_research_system.agents.risk_assessor import RiskAssessorAgent
    from investment_research_system.agents.sentiment import SentimentAgent
    from investment_research_system.tools.tavily_search import TavilySearch

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_KEY,
        max_output_tokens=1024,
        temperature=0.7,
    )

    memory = make_mock_memory()
    tavily = TavilySearch(api_key=TAVILY_KEY)

    return [
        ResearcherAgent(llm=llm, memory=memory, tavily=tavily),
        AnalystAgent(llm=llm, memory=memory, tavily=tavily),
        RiskAssessorAgent(llm=llm, memory=memory, tavily=tavily),
        SentimentAgent(llm=llm, memory=memory, tavily=tavily),
    ]


# --- Unit tests for conflict + quality (no API calls) ---
def test_conflict_detection():
    """Detect conflicts between fake agent responses."""
    from investment_research_system.orchestrator.conflict import detect_conflicts
    from investment_research_system.models.schemas import AgentResponse

    responses = [
        AgentResponse(
            agent_name="analyst",
            content="NVIDIA is fairly valued with strong growth and bullish outlook. The upside potential is significant.",
            confidence=0.85,
        ),
        AgentResponse(
            agent_name="risk_assessor",
            content="NVIDIA is overvalued and faces serious downside risk. Bearish concerns about export controls are a major threat.",
            confidence=0.80,
        ),
    ]

    conflicts = detect_conflicts(responses)

    assert len(conflicts) > 0
    assert any(c["topic"] == "outlook" for c in conflicts)

    print(f"\n--- Conflicts ({len(conflicts)}) ---")
    for c in conflicts:
        print(f"  [{c['severity']}] {' vs '.join(c['agents_involved'])}: {c['disagreement']}")


def test_quality_assessment():
    """Quality scoring with different agent counts."""
    from investment_research_system.orchestrator.quality import assess_quality
    from investment_research_system.models.schemas import AgentResponse

    # Full responses — should be HIGH
    full = [AgentResponse(agent_name=f"agent_{i}", content="analysis", confidence=0.8) for i in range(4)]
    result = assess_quality(full, [])
    assert result.grade == "HIGH"
    assert result.passed is True
    print(f"\n4 agents → grade: {result.grade}")

    # 3 responses — should be MEDIUM
    partial = full[:3]
    failed = [{"agent": "sentiment", "error_type": "timeout", "error_message": "Timed out after 30s"}]
    result = assess_quality(partial, [], failed_agents=failed)
    assert result.grade == "MEDIUM"
    assert result.passed is True
    assert any("unavailable" in d or "sentiment" in d for d in result.disclaimers)
    print(f"3 agents → grade: {result.grade}, disclaimers: {result.disclaimers}")

    # 1 response — should be MINIMAL
    minimal = full[:1]
    result = assess_quality(minimal, [])
    assert result.grade == "MINIMAL"
    assert result.passed is False
    print(f"1 agent  → grade: {result.grade}")

    # 0 responses — should be FAILED
    result = assess_quality([], [])
    assert result.grade == "FAILED"
    assert result.passed is False
    print(f"0 agents → grade: {result.grade}")


def test_circuit_breaker():
    """Circuit breaker trips after max failures."""
    from investment_research_system.orchestrator.graph import CircuitBreaker

    cb = CircuitBreaker(max_failures=2, reset_timeout=1)

    # Starts closed
    assert cb.can_call("researcher") is True
    assert cb.get_state("researcher") == "closed"

    # 1 failure — still closed
    cb.record_failure("researcher")
    assert cb.can_call("researcher") is True

    # 2 failures — trips to open
    cb.record_failure("researcher")
    assert cb.can_call("researcher") is False
    assert cb.get_state("researcher") == "open"

    # Other agents unaffected
    assert cb.can_call("analyst") is True

    # Success resets
    cb.record_success("researcher")
    assert cb.can_call("researcher") is True
    assert cb.get_state("researcher") == "closed"

    print("\n--- Circuit Breaker: all states verified ---")


# --- Full end-to-end test (makes real API calls) ---
@skip_no_keys
def test_full_orchestrator_run():
    """Run the complete pipeline: query → 4 agents → conflicts → quality → report."""
    from investment_research_system.models.schemas import ResearchQuery, ResearchReport
    from investment_research_system.orchestrator.graph import ResearchOrchestrator

    agents = make_agents()
    orchestrator = ResearchOrchestrator(agents=agents, max_retries=1)

    query = ResearchQuery(
        query="Should I invest in NVIDIA?",
        focus_areas=["financials", "risk", "sentiment"],
        depth="standard",
    )

    report = asyncio.run(orchestrator.run(query))

    # Verify report structure
    assert isinstance(report, ResearchReport)
    assert report.id  # has a UUID
    assert report.query == query
    assert len(report.agent_responses) > 0
    assert report.total_tokens > 0
    assert report.processing_time_seconds > 0

    print(f"\n{'='*60}")
    print(f"RESEARCH REPORT")
    print(f"{'='*60}")
    print(f"Query: {report.query.query}")
    print(f"Agents responded: {len(report.agent_responses)}")
    print(f"Total tokens: {report.total_tokens}")
    print(f"Total cost: ${report.total_cost_usd:.4f}")
    print(f"Processing time: {report.processing_time_seconds:.1f}s")
    print(f"\n--- Agent Responses ---")
    for r in report.agent_responses:
        print(f"\n[{r.agent_name}] (confidence: {r.confidence:.2f})")
        print(f"{r.content[:300]}...")
    print(f"\n--- Summary ---")
    print(report.summary[:800])
