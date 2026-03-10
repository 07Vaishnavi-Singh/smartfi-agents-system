"""Smoke test for RiskAssessorAgent.

Run: PYTHONPATH=src uv run pytest tests/test_risk_assessor.py -v -s
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


@pytest.fixture
def risk_assessor():
    from unittest.mock import AsyncMock, MagicMock

    from langchain_google_genai import ChatGoogleGenerativeAI

    from investment_research_system.agents.risk_assessor import RiskAssessorAgent
    from investment_research_system.tools.tavily_search import TavilySearch

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_KEY,
        max_output_tokens=1024,
        temperature=0.7,
    )

    mock_memory = MagicMock()
    mock_memory.parallel_search = AsyncMock(return_value={
        "long_term": [
            {"text": "NVIDIA derives approximately 25% of datacenter revenue from China-based customers.", "score": 0.88},
            {"text": "US government expanded export controls on AI chips to China in October 2024.", "score": 0.85},
        ],
        "semantic": [
            {"memory": "NVIDIA top 5 customers account for 40% of revenue"},
            {"memory": "AMD launched MI300X as direct competitor to NVIDIA H100"},
        ],
    })
    mock_memory.update_agent_status = AsyncMock()
    mock_memory.store_research = AsyncMock()

    tavily = TavilySearch(api_key=TAVILY_KEY)

    return RiskAssessorAgent(llm=llm, memory=mock_memory, tavily=tavily)


@skip_no_keys
def test_risk_properties(risk_assessor):
    assert risk_assessor.name == "risk_assessor"
    assert "risk" in risk_assessor.system_prompt.lower()
    assert "devil" in risk_assessor.system_prompt.lower()
    print(f"\nName: {risk_assessor.name}")


@skip_no_keys
def test_risk_build_query(risk_assessor):
    """Prompt should frame around risks and downsides."""
    memory = {
        "long_term": [{"text": "NVIDIA 25% revenue from China", "score": 0.9}],
        "semantic": [{"memory": "AMD launched MI300X competitor"}],
    }
    prompt = risk_assessor.build_query("Should I invest in NVIDIA?", memory)

    assert "WRONG" in prompt  # explicitly asks for downsides
    assert "regulatory" in prompt.lower()
    assert "NVIDIA 25% revenue from China" in prompt

    print(f"\n--- Risk Prompt ({len(prompt)} chars) ---")
    print(prompt[:400])


@skip_no_keys
def test_risk_full_run(risk_assessor):
    """Full run — should produce risk-focused analysis."""
    from investment_research_system.models.schemas import AgentResponse

    response = asyncio.run(
        risk_assessor.run(
            "What are the risks of investing in NVIDIA?",
            session_id="test-session-003",
        )
    )

    assert isinstance(response, AgentResponse)
    assert response.agent_name == "risk_assessor"
    assert len(response.content) > 100
    # Should mention risks — check for risk-related keywords
    content_lower = response.content.lower()
    assert any(word in content_lower for word in ["risk", "threat", "concern", "regulatory"])

    print(f"\n--- Risk Assessor Full Run ---")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"\nContent preview:\n{response.content[:500]}")
