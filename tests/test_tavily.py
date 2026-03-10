"""Quick smoke test for TavilySearch.

Run: PYTHONPATH=src uv run pytest tests/test_tavily.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from investment_research_system.tools.tavily_search import TavilySearch
from investment_research_system.models.schemas import Source, SourceType


@pytest.fixture
def tavily():
    """Create a TavilySearch instance using your .env API key."""
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        pytest.skip("TAVILY_API_KEY not set in .env — skipping live tests")

    return TavilySearch(api_key=api_key)


def test_basic_search(tavily):
    """search() returns results with expected fields."""
    result = tavily.search("NVIDIA stock price 2025", max_results=3)

    assert result["query"] == "NVIDIA stock price 2025"
    assert result["result_count"] > 0
    assert result["searched_at"]  # not empty

    # Check each result has the expected shape
    for item in result["results"]:
        assert "title" in item
        assert "url" in item
        assert "content" in item
        assert "score" in item

    print(f"\n--- Basic Search ---")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Results: {result['result_count']}")
    for r in result["results"]:
        print(f"  - {r['title']} ({r['url'][:60]})")


def test_search_as_sources(tavily):
    """search_as_sources() returns list[Source] matching our schema."""
    sources = tavily.search_as_sources("Tesla earnings report", max_results=3)

    assert len(sources) > 0
    assert all(isinstance(s, Source) for s in sources)
    assert all(s.source_type == SourceType.web for s in sources)
    assert all(0.0 <= s.reliability_score <= 1.0 for s in sources)

    print(f"\n--- Sources ---")
    for s in sources:
        print(f"  - {s.title} (score: {s.reliability_score:.2f})")


def test_search_news(tavily):
    """search_news() returns recent news articles."""
    result = tavily.search_news("AI chip market", max_results=3)

    assert result["result_count"] > 0

    print(f"\n--- News Search ---")
    print(f"Answer: {result['answer'][:200]}...")
    for r in result["results"]:
        print(f"  - {r['title']}")


def test_search_financials(tavily):
    """search_financials() uses advanced depth and financial domains."""
    result = tavily.search_financials("Apple P/E ratio 2025", max_results=3)

    assert result["result_count"] > 0

    print(f"\n--- Financial Search ---")
    print(f"Answer: {result['answer'][:200]}...")
    for r in result["results"]:
        print(f"  - {r['title']} ({r['url'][:60]})")


def test_ping(tavily):
    """ping() returns True when API key is valid."""
    assert tavily.ping() is True
    print("\n--- Ping: OK ---")
