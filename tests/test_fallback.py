"""Integration tests for LLM fallback chain.

Tests that when a model hits its rate limit (429), the agent
automatically falls back to the next model in the chain.

Run: PYTHONPATH=src uv run pytest tests/test_fallback.py -v -s
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Create a mock MemoryManager that returns empty results."""
    mock = MagicMock()
    mock.parallel_search = AsyncMock(return_value={
        "long_term": [],
        "semantic": [],
    })
    mock.update_agent_status = AsyncMock()
    mock.store_research = AsyncMock()
    return mock


def make_gemini_llm(model: str):
    """Create a real Gemini LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_KEY,
        max_output_tokens=512,
        temperature=0.7,
    )


def make_rate_limit_llm():
    """Create a mock LLM that always raises a rate limit error."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=Exception("429 Resource has been exhausted (e.g. check quota).")
    )
    mock_llm.model = "mock-rate-limited"
    return mock_llm


# --- Unit Tests (no API calls) ---

def test_fallback_skips_rate_limited_model():
    """When primary model returns 429, agent falls back to second model."""
    from langchain_core.messages import AIMessage

    from investment_research_system.agents.researcher import ResearcherAgent

    # Primary: always rate limits
    primary = make_rate_limit_llm()

    # Fallback: returns a valid structured response
    fallback = AsyncMock()
    fallback.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"status": "completed", "analysis": "NVIDIA is a strong buy."}',
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    ))
    fallback.model = "mock-fallback"

    agent = ResearcherAgent(
        llm=primary,
        memory=make_mock_memory(),
        tavily=MagicMock(search=MagicMock(return_value=[])),
        fallback_llms=[fallback],
    )

    response = asyncio.run(agent.run("NVIDIA analysis", session_id="test-001"))

    assert response.agent_name == "researcher"
    assert "NVIDIA" in response.content
    # Verify primary was called and failed
    primary.ainvoke.assert_called_once()
    # Verify fallback was called and succeeded
    fallback.ainvoke.assert_called_once()


def test_fallback_all_models_exhausted():
    """When ALL models return 429, agent raises LLMRateLimitError."""
    from investment_research_system.agents.researcher import ResearcherAgent
    from investment_research_system.errors import LLMRateLimitError

    primary = make_rate_limit_llm()
    fallback1 = make_rate_limit_llm()
    fallback2 = make_rate_limit_llm()

    agent = ResearcherAgent(
        llm=primary,
        memory=make_mock_memory(),
        tavily=MagicMock(search=MagicMock(return_value=[])),
        fallback_llms=[fallback1, fallback2],
    )

    with pytest.raises(LLMRateLimitError, match="All 3 models exhausted"):
        asyncio.run(agent.run("NVIDIA analysis", session_id="test-002"))

    # All 3 models should have been tried
    primary.ainvoke.assert_called_once()
    fallback1.ainvoke.assert_called_once()
    fallback2.ainvoke.assert_called_once()


def test_non_rate_limit_error_does_not_fallback():
    """Non-429 errors propagate immediately without trying fallbacks."""
    from investment_research_system.agents.researcher import ResearcherAgent

    primary = AsyncMock()
    primary.ainvoke = AsyncMock(side_effect=ValueError("Invalid API key"))
    primary.model = "mock-bad-key"

    fallback = AsyncMock()
    fallback.model = "mock-fallback"

    agent = ResearcherAgent(
        llm=primary,
        memory=make_mock_memory(),
        tavily=MagicMock(search=MagicMock(return_value=[])),
        fallback_llms=[fallback],
    )

    with pytest.raises(ValueError, match="Invalid API key"):
        asyncio.run(agent.run("NVIDIA analysis", session_id="test-003"))

    # Fallback should NOT have been called
    fallback.ainvoke.assert_not_called()


def test_primary_succeeds_no_fallback_needed():
    """When primary model works fine, fallbacks are never touched."""
    from langchain_core.messages import AIMessage

    from investment_research_system.agents.researcher import ResearcherAgent

    primary = AsyncMock()
    primary.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"status": "completed", "analysis": "All good."}',
        usage_metadata={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
    ))
    primary.model = "mock-primary"

    fallback = AsyncMock()
    fallback.model = "mock-fallback"

    agent = ResearcherAgent(
        llm=primary,
        memory=make_mock_memory(),
        tavily=MagicMock(search=MagicMock(return_value=[])),
        fallback_llms=[fallback],
    )

    response = asyncio.run(agent.run("NVIDIA analysis", session_id="test-004"))

    assert "All good" in response.content
    fallback.ainvoke.assert_not_called()


# --- Live Integration Test (hits real Gemini API) ---

@skip_no_keys
def test_live_fallback_chain_with_real_models():
    """Live test: primary is a mock 429, fallback is real Gemini.

    This proves the fallback chain works end-to-end with a real LLM.
    Primary model simulates rate limit → falls back to real gemini-2.5-flash.
    """
    from investment_research_system.agents.researcher import ResearcherAgent
    from investment_research_system.models.schemas import AgentResponse
    from investment_research_system.tools.tavily_search import TavilySearch

    # Primary: always 429
    primary = make_rate_limit_llm()

    # Fallback: real Gemini model
    real_fallback = make_gemini_llm("gemini-2.5-flash")

    agent = ResearcherAgent(
        llm=primary,
        memory=make_mock_memory(),
        tavily=TavilySearch(api_key=TAVILY_KEY),
        fallback_llms=[real_fallback],
    )

    response = asyncio.run(agent.run(
        "What is Apple's current market cap?",
        session_id="test-live-fallback",
    ))

    assert isinstance(response, AgentResponse)
    assert response.agent_name == "researcher"
    assert len(response.content) > 20
    assert response.tokens_used > 0

    print(f"\n--- Live Fallback Test ---")
    print(f"Primary: mock (rate limited)")
    print(f"Fallback: gemini-2.5-flash (succeeded)")
    print(f"Tokens: {response.tokens_used}")
    print(f"Content preview: {response.content[:200]}")
