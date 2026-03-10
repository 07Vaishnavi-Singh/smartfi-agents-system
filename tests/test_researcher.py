"""Smoke test for ResearcherAgent.

Run: PYTHONPATH=src uv run pytest tests/test_researcher.py -v -s

Uses Google Gemini (free tier) for LLM calls + Tavily for web search.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# --- Skip if API keys are missing ---
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

skip_no_keys = pytest.mark.skipif(
    not GOOGLE_KEY or not TAVILY_KEY,
    reason="GOOGLE_API_KEY and TAVILY_API_KEY required for live test",
)


# --- Fixtures ---
@pytest.fixture
def researcher():
    """Create a ResearcherAgent with Gemini + Tavily.

    Uses Google Gemini (free) instead of Claude for testing.
    Uses a mock MemoryManager (no Redis/Qdrant needed).
    """
    from unittest.mock import AsyncMock, MagicMock

    from langchain_google_genai import ChatGoogleGenerativeAI

    from investment_research_system.agents.researcher import ResearcherAgent
    from investment_research_system.tools.tavily_search import TavilySearch

    # Gemini — free tier, no billing needed
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_KEY,
        max_output_tokens=1024,
        temperature=0.7,
    )

    # Mock memory — returns empty results (no Redis/Qdrant needed)
    mock_memory = MagicMock()
    mock_memory.parallel_search = AsyncMock(return_value={
        "long_term": [],
        "semantic": [],
    })
    mock_memory.update_agent_status = AsyncMock()
    mock_memory.store_research = AsyncMock()

    # Real Tavily
    tavily = TavilySearch(api_key=TAVILY_KEY)

    return ResearcherAgent(llm=llm, memory=mock_memory, tavily=tavily)


# --- Tests ---
@skip_no_keys
def test_researcher_properties(researcher):
    """Check name and system_prompt are defined correctly."""
    assert researcher.name == "researcher"
    assert "financial research" in researcher.system_prompt.lower()
    print(f"\nName: {researcher.name}")
    print(f"System prompt length: {len(researcher.system_prompt)} chars")


@skip_no_keys
def test_researcher_extract_sources(researcher):
    """extract_sources() returns Source objects from Tavily."""
    from investment_research_system.models.schemas import Source

    sources = researcher.extract_sources("NVIDIA stock analysis")

    assert len(sources) > 0
    assert all(isinstance(s, Source) for s in sources)

    print(f"\n--- Sources ({len(sources)}) ---")
    for s in sources:
        print(f"  - {s.title} (score: {s.reliability_score:.2f})")


@skip_no_keys
def test_researcher_build_query(researcher):
    """build_query() produces a formatted prompt with web + memory context."""
    fake_memory = {"long_term": [], "semantic": []}
    prompt = researcher.build_query("Should I invest in NVIDIA?", fake_memory)

    assert "NVIDIA" in prompt
    assert "Question" in prompt
    assert "Web" in prompt or "web" in prompt

    print(f"\n--- Built Prompt ({len(prompt)} chars) ---")
    print(prompt[:500])
    print("...")


@skip_no_keys
def test_researcher_full_run(researcher):
    """Full end-to-end: run() calls Gemini and returns AgentResponse.

    1. Searches memory (mocked — returns empty)
    2. Searches Tavily (real API call)
    3. Builds prompt with web results
    4. Calls Gemini (real API call — free)
    5. Returns AgentResponse with content, cost, tokens
    """
    from investment_research_system.models.schemas import AgentResponse

    response = asyncio.run(
        researcher.run(
            "What is NVIDIA's current market position in AI chips?",
            session_id="test-session-001",
        )
    )

    # Verify response shape
    assert isinstance(response, AgentResponse)
    assert response.agent_name == "researcher"
    assert len(response.content) > 100
    assert response.tokens_used > 0
    assert response.latency_ms > 0
    assert 0.0 <= response.confidence <= 1.0

    print(f"\n--- Full Run Result ---")
    print(f"Agent: {response.agent_name}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Cost: ${response.cost_usd:.4f}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Sources: {len(response.sources)}")
    print(f"\nContent preview:\n{response.content[:500]}")
