"""Smoke test for AnalystAgent.

Run: PYTHONPATH=src uv run pytest tests/test_analyst.py -v -s
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
def analyst():
    from unittest.mock import AsyncMock, MagicMock

    from langchain_google_genai import ChatGoogleGenerativeAI

    from investment_research_system.agents.analyst import AnalystAgent
    from investment_research_system.tools.tavily_search import TavilySearch

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_KEY,
        max_output_tokens=1024,
        temperature=0.7,
    )

    # Mock memory WITH some data — analyst should reason over it
    mock_memory = MagicMock()
    mock_memory.parallel_search = AsyncMock(return_value={
        "long_term": [
            {"text": "NVIDIA reported revenue of $35.1B in Q3 2024, up 94% YoY. Operating margin was 57%.", "score": 0.92},
            {"text": "AMD revenue was $6.8B in Q3 2024. AI chip segment grew 122% YoY.", "score": 0.78},
        ],
        "semantic": [
            {"memory": "NVIDIA P/E ratio is approximately 52"},
            {"memory": "AMD P/E ratio is approximately 38"},
            {"memory": "Semiconductor sector average P/E is 25"},
        ],
    })
    mock_memory.update_agent_status = AsyncMock()
    mock_memory.store_research = AsyncMock()

    tavily = TavilySearch(api_key=TAVILY_KEY)

    return AnalystAgent(llm=llm, memory=mock_memory, tavily=tavily)


@skip_no_keys
def test_analyst_properties(analyst):
    assert analyst.name == "analyst"
    assert "valuation" in analyst.system_prompt.lower()
    print(f"\nName: {analyst.name}")


@skip_no_keys
def test_analyst_build_query(analyst):
    """Analyst prompt should include memory data for financial reasoning."""
    memory = {
        "long_term": [{"text": "NVIDIA revenue $35B", "score": 0.9}],
        "semantic": [{"memory": "NVIDIA P/E is 52"}],
    }
    prompt = analyst.build_query("Is NVIDIA overvalued?", memory)

    assert "NVIDIA" in prompt
    assert "financial analysis" in prompt.lower()
    assert "NVIDIA revenue $35B" in prompt  # memory context included
    assert "NVIDIA P/E is 52" in prompt     # semantic facts included

    print(f"\n--- Built Prompt ({len(prompt)} chars) ---")
    print(prompt[:400])


@skip_no_keys
def test_analyst_full_run(analyst):
    """Full run with mock memory data — analyst interprets financial data."""
    from investment_research_system.models.schemas import AgentResponse

    response = asyncio.run(
        analyst.run("Is NVIDIA overvalued compared to AMD?", session_id="test-session-002")
    )

    assert isinstance(response, AgentResponse)
    assert response.agent_name == "analyst"
    assert len(response.content) > 100
    assert response.tokens_used > 0
    # Analyst should have higher confidence since it had memory data
    assert response.confidence >= 0.7

    print(f"\n--- Analyst Full Run ---")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Cost: ${response.cost_usd:.4f}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"\nContent preview:\n{response.content[:500]}")
