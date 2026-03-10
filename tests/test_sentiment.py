"""Smoke test for SentimentAgent.

Run: PYTHONPATH=src uv run pytest tests/test_sentiment.py -v -s
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
def sentiment():
    from unittest.mock import AsyncMock, MagicMock

    from langchain_google_genai import ChatGoogleGenerativeAI

    from investment_research_system.agents.sentiment import SentimentAgent
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
            {"text": "NVIDIA stock rose 240% in 2024, making it the most valuable company globally.", "score": 0.90},
            {"text": "Several hedge funds reduced NVIDIA positions in Q3 2024, citing valuation concerns.", "score": 0.82},
        ],
        "semantic": [
            {"memory": "Wall Street consensus on NVIDIA: 42 buy, 5 hold, 1 sell"},
            {"memory": "NVIDIA insider selling increased 15% in last quarter"},
            {"memory": "Retail investor sentiment on NVIDIA is strongly bullish"},
        ],
    })
    mock_memory.update_agent_status = AsyncMock()
    mock_memory.store_research = AsyncMock()

    tavily = TavilySearch(api_key=TAVILY_KEY)

    return SentimentAgent(llm=llm, memory=mock_memory, tavily=tavily)


@skip_no_keys
def test_sentiment_properties(sentiment):
    assert sentiment.name == "sentiment"
    assert "sentiment" in sentiment.system_prompt.lower()
    print(f"\nName: {sentiment.name}")


@skip_no_keys
def test_sentiment_build_query(sentiment):
    """Prompt should focus on mood and opinions, not numbers."""
    memory = {
        "long_term": [],
        "semantic": [{"memory": "Wall Street consensus: 42 buy, 5 hold, 1 sell"}],
    }
    prompt = sentiment.build_query("What's the sentiment on NVIDIA?", memory)

    assert "FEEL" in prompt  # sentiment-specific framing
    assert "sentiment score" in prompt.lower()
    assert "42 buy, 5 hold, 1 sell" in prompt

    print(f"\n--- Sentiment Prompt ({len(prompt)} chars) ---")
    print(prompt[:400])


@skip_no_keys
def test_sentiment_full_run(sentiment):
    """Full run — should produce sentiment-focused analysis."""
    from investment_research_system.models.schemas import AgentResponse

    response = asyncio.run(
        sentiment.run(
            "What is the market sentiment on NVIDIA stock?",
            session_id="test-session-004",
        )
    )

    assert isinstance(response, AgentResponse)
    assert response.agent_name == "sentiment"
    assert len(response.content) > 100
    # Should contain sentiment-related language
    content_lower = response.content.lower()
    assert any(word in content_lower for word in ["bullish", "bearish", "sentiment", "consensus"])

    print(f"\n--- Sentiment Full Run ---")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Sources: {len(response.sources)}")
    print(f"\nContent preview:\n{response.content[:500]}")
