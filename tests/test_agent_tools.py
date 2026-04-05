"""Tests for agent-level tools (plan-then-execute pattern).

Tests the 4 tool factories: create_plan, search_web, search_past_research, write_analysis.

Run: uv run pytest tests/test_agent_tools.py -v
"""

import os
import sys
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================================
# create_plan tool
# ============================================================================


@pytest.mark.asyncio
async def test_create_plan_tool_stores_plan():
    """create_plan should store the plan in context and return confirmation."""
    from investment_research_system.agents.agent_tools import create_plan_tool

    tool_fn = create_plan_tool()
    context = {"plan": None}

    result = await tool_fn(
        data_needed=[
            {"topic": "NVIDIA Q4 earnings", "source": "web", "priority": 1},
            {"topic": "NVIDIA valuation history", "source": "memory", "priority": 2},
        ],
        _context=context,
    )

    assert context["plan"] is not None
    assert len(context["plan"]) == 2
    assert context["plan"][0]["topic"] == "NVIDIA Q4 earnings"
    assert "2 items" in result


@pytest.mark.asyncio
async def test_create_plan_tool_rejects_empty():
    """create_plan should return error if no items provided."""
    from investment_research_system.agents.agent_tools import create_plan_tool

    tool_fn = create_plan_tool()
    context = {"plan": None}

    result = await tool_fn(data_needed=[], _context=context)

    assert context["plan"] is None
    assert "empty" in result.lower() or "no items" in result.lower()


# ============================================================================
# search_web tool
# ============================================================================


@pytest.mark.asyncio
async def test_search_web_tool_calls_tavily():
    """search_web should call tavily.search() and format results as text."""
    from investment_research_system.agents.agent_tools import create_search_web_tool

    tavily = MagicMock()
    tavily.search = MagicMock(return_value={
        "results": [
            {"title": "NVIDIA Q4 Earnings", "content": "Revenue hit $39B in Q4...", "url": "https://example.com", "published_date": "2025-01-28"},
            {"title": "NVIDIA Outlook", "content": "Analysts expect growth...", "url": "https://example2.com", "published_date": "2025-02-01"},
        ],
        "answer": "NVIDIA had strong Q4 results",
    })

    tool_fn = create_search_web_tool(tavily=tavily)
    result = await tool_fn("NVIDIA Q4 earnings", topic="finance")

    assert "[1]" in result
    assert "NVIDIA Q4 Earnings" in result
    assert "$39B" in result
    assert "[2]" in result
    tavily.search.assert_called_once()


@pytest.mark.asyncio
async def test_search_web_tool_handles_failure():
    """search_web should return error message when Tavily is down."""
    from investment_research_system.agents.agent_tools import create_search_web_tool

    tavily = MagicMock()
    tavily.search = MagicMock(side_effect=Exception("API rate limit"))

    tool_fn = create_search_web_tool(tavily=tavily)
    result = await tool_fn("test query")

    assert "failed" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_search_web_tool_no_tavily():
    """search_web with tavily=None should return unavailable message."""
    from investment_research_system.agents.agent_tools import create_search_web_tool

    tool_fn = create_search_web_tool(tavily=None)
    result = await tool_fn("test query")

    assert "unavailable" in result.lower() or "not available" in result.lower()


# ============================================================================
# search_past_research tool
# ============================================================================


@pytest.mark.asyncio
async def test_search_past_research_tool_calls_memory():
    """search_past_research should call memory.parallel_search() and format with labels."""
    from investment_research_system.agents.agent_tools import create_search_past_research_tool

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={
        "long_term": [{"text": "NVIDIA P/E was 58 in January", "score": 0.87}],
        "semantic": [{"memory": "NVIDIA competes with AMD in AI chips"}],
        "topics": ["nvidia"],
    })

    tool_fn = create_search_past_research_tool(memory=memory)
    result = await tool_fn("NVIDIA valuation")

    assert "[RESEARCH-1]" in result
    assert "0.87" in result
    assert "[FACT-1]" in result
    assert "AMD" in result
    memory.parallel_search.assert_called_once_with("NVIDIA valuation")


@pytest.mark.asyncio
async def test_search_past_research_empty_results():
    """search_past_research should return clear message when nothing found."""
    from investment_research_system.agents.agent_tools import create_search_past_research_tool

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={
        "long_term": [], "semantic": [], "topics": [],
    })

    tool_fn = create_search_past_research_tool(memory=memory)
    result = await tool_fn("something obscure")

    assert "no relevant" in result.lower() or "no past" in result.lower()


# ============================================================================
# write_analysis tool
# ============================================================================


@pytest.mark.asyncio
async def test_write_analysis_sets_content_and_exits():
    """write_analysis should store analysis in context."""
    from investment_research_system.agents.agent_tools import create_write_analysis_tool

    tool_fn = create_write_analysis_tool()
    context = {"analysis": None, "plan": [{"topic": "test", "source": "web", "priority": 1}]}

    result = await tool_fn(
        content="NVIDIA revenue $39B, P/E 52, China export risk moderate.",
        _context=context,
    )

    assert context["analysis"] == "NVIDIA revenue $39B, P/E 52, China export risk moderate."
    assert "analysis complete" in result.lower() or "written" in result.lower()
