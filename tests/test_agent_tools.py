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


# ============================================================================
# Full agent loop tests
# ============================================================================


def _make_concrete_agent(llm, memory, tavily=None, max_tool_turns=5):
    """Create a concrete agent subclass for testing."""
    from investment_research_system.agents.base import BaseAgent

    class TestAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "test_agent"

        @property
        def system_prompt(self) -> str:
            return "You are a test agent. Use tools to gather data, then call write_analysis."

    return TestAgent(llm=llm, memory=memory, tavily=tavily, max_tool_turns=max_tool_turns)


@pytest.mark.asyncio
async def test_agent_plan_then_execute_happy_path():
    """Full loop: plan → search → write_analysis → AgentResponse."""
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Plan
            return AIMessage(content="", tool_calls=[
                {"name": "create_plan", "args": {"data_needed": [{"topic": "NVIDIA earnings", "source": "web", "priority": 1}]}, "id": "c1"}
            ])
        elif call_count == 2:
            # Search
            return AIMessage(content="", tool_calls=[
                {"name": "search_past_research", "args": {"query": "NVIDIA earnings"}, "id": "c2"}
            ])
        elif call_count == 3:
            # Write analysis
            return AIMessage(content="", tool_calls=[
                {"name": "write_analysis", "args": {"content": "NVIDIA revenue is $39B with strong growth."}, "id": "c3"}
            ])
        else:
            return AIMessage(content="Done.")

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory)
    result = await agent.run("NVIDIA earnings", session_id="test-session")

    assert result.agent_name == "test_agent"
    assert "NVIDIA revenue is $39B" in result.content
    assert result.tokens_used >= 0


@pytest.mark.asyncio
async def test_agent_max_turns_forces_output():
    """When max turns hit without write_analysis, use last LLM message content."""
    from langchain_core.messages import AIMessage

    async def always_search(messages, **kwargs):
        return AIMessage(content="", tool_calls=[
            {"name": "search_past_research", "args": {"query": "test"}, "id": "c_loop"}
        ])

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await always_search(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory, max_tool_turns=2)
    result = await agent.run("test", session_id="test-session")

    # Should still return an AgentResponse (forced output)
    assert result is not None
    assert result.agent_name == "test_agent"


@pytest.mark.asyncio
async def test_agent_cost_accumulates_across_turns():
    """Token count should be the sum across all LLM calls in the loop."""
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            msg = AIMessage(content="", tool_calls=[
                {"name": "search_past_research", "args": {"query": "test"}, "id": f"c{call_count}"}
            ])
        else:
            msg = AIMessage(content="", tool_calls=[
                {"name": "write_analysis", "args": {"content": "Final analysis."}, "id": f"c{call_count}"}
            ])
        # Simulate 100 tokens per call
        msg.usage_metadata = {"input_tokens": 80, "output_tokens": 20}
        return msg

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory)
    result = await agent.run("test", session_id="test-session")

    # 3 LLM calls × 100 tokens each = 300 total
    assert result.tokens_used == 300


@pytest.mark.asyncio
async def test_agent_no_tavily_still_works():
    """Agent with tavily=None should work — search_web returns unavailable."""
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AIMessage(content="", tool_calls=[
                {"name": "search_web", "args": {"query": "NVIDIA"}, "id": "c1"}
            ])
        else:
            return AIMessage(content="", tool_calls=[
                {"name": "write_analysis", "args": {"content": "Analysis without web data."}, "id": "c2"}
            ])

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory, tavily=None)
    result = await agent.run("NVIDIA", session_id="test-session")

    assert "Analysis without web data" in result.content


@pytest.mark.asyncio
async def test_agent_skips_plan_still_works():
    """Agent that goes straight to search without create_plan should still work."""
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Skip plan, go straight to search
            return AIMessage(content="", tool_calls=[
                {"name": "search_web", "args": {"query": "NVIDIA"}, "id": "c1"}
            ])
        else:
            return AIMessage(content="", tool_calls=[
                {"name": "write_analysis", "args": {"content": "Analysis without plan."}, "id": "c2"}
            ])

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    tavily = MagicMock()
    tavily.search = MagicMock(return_value={"results": [{"title": "NVIDIA", "content": "Data", "url": "http://x", "published_date": ""}]})

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory, tavily=tavily)
    result = await agent.run("NVIDIA", session_id="test-session")

    assert "Analysis without plan" in result.content


# ============================================================================
# Real subclass agent tests (researcher, analyst, sentiment, risk)
# ============================================================================


def _make_mock_llm_for_agent(plan_items=None, analysis_text="Test analysis complete."):
    """Create a MockLLM that simulates plan → search → write_analysis flow."""
    from langchain_core.messages import AIMessage

    plan_items = plan_items or [{"topic": "test data", "source": "memory", "priority": 1}]
    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AIMessage(content="", tool_calls=[
                {"name": "create_plan", "args": {"data_needed": plan_items}, "id": "c1"}
            ])
        elif call_count == 2:
            return AIMessage(content="", tool_calls=[
                {"name": "search_past_research", "args": {"query": plan_items[0]["topic"]}, "id": "c2"}
            ])
        elif call_count == 3:
            return AIMessage(content="", tool_calls=[
                {"name": "write_analysis", "args": {"content": analysis_text}, "id": "c3"}
            ])
        else:
            return AIMessage(content="Done.")

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    return MockLLM()


def _make_standard_mock_memory():
    """Create a standard mock memory manager for agent tests."""
    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={
        "long_term": [{"text": "Past research data about NVIDIA.", "score": 0.85}],
        "semantic": [{"memory": "NVIDIA P/E is approximately 52"}],
        "topics": ["nvidia"],
    })
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()
    return memory


@pytest.mark.asyncio
async def test_researcher_agent_with_react_loop():
    """ResearcherAgent should work with the new plan-then-execute loop."""
    from investment_research_system.agents.researcher import ResearcherAgent

    llm = _make_mock_llm_for_agent(
        plan_items=[{"topic": "NVIDIA Q4 earnings", "source": "web", "priority": 1}],
        analysis_text="[CONFIRMED] NVIDIA Q4 revenue was $39B, up 15% YoY.",
    )
    memory = _make_standard_mock_memory()
    tavily = MagicMock()
    tavily.search = MagicMock(return_value={"results": [], "answer": None})
    tavily.search_as_sources = MagicMock(return_value=[])

    agent = ResearcherAgent(llm=llm, memory=memory, tavily=tavily)
    result = await agent.run("NVIDIA Q4 earnings", session_id="test-session")

    assert result.agent_name == "researcher"
    assert "NVIDIA Q4 revenue" in result.content
    assert result.confidence > 0
    memory.update_agent_status.assert_any_call("test-session", "researcher", "running")
    memory.update_agent_status.assert_any_call("test-session", "researcher", "done")
    memory.store_research.assert_called_once()


@pytest.mark.asyncio
async def test_analyst_agent_with_react_loop():
    """AnalystAgent should work with the new loop."""
    from investment_research_system.agents.analyst import AnalystAgent

    llm = _make_mock_llm_for_agent(
        plan_items=[{"topic": "NVIDIA valuation", "source": "memory", "priority": 1}],
        analysis_text="NVIDIA trades at P/E 52, fair value given 40% growth.",
    )
    memory = _make_standard_mock_memory()
    tavily = MagicMock()
    tavily.search_as_sources = MagicMock(return_value=[])

    agent = AnalystAgent(llm=llm, memory=memory, tavily=tavily)
    result = await agent.run("NVIDIA valuation analysis", session_id="test-session")

    assert result.agent_name == "analyst"
    assert "P/E 52" in result.content


@pytest.mark.asyncio
async def test_sentiment_agent_with_react_loop():
    """SentimentAgent should work with the new loop."""
    from investment_research_system.agents.sentiment import SentimentAgent

    llm = _make_mock_llm_for_agent(
        plan_items=[{"topic": "NVIDIA market sentiment", "source": "web", "priority": 1}],
        analysis_text="Bullish sentiment: 48 buy ratings, 3 hold, 1 sell.",
    )
    memory = _make_standard_mock_memory()
    tavily = MagicMock()
    tavily.search_as_sources = MagicMock(return_value=[])

    agent = SentimentAgent(llm=llm, memory=memory, tavily=tavily)
    result = await agent.run("NVIDIA sentiment", session_id="test-session")

    assert result.agent_name == "sentiment"
    assert "Bullish" in result.content


@pytest.mark.asyncio
async def test_risk_agent_with_react_loop():
    """RiskAssessorAgent should work with the new loop."""
    from investment_research_system.agents.risk_assessor import RiskAssessorAgent

    llm = _make_mock_llm_for_agent(
        plan_items=[{"topic": "NVIDIA China export risk", "source": "web", "priority": 1}],
        analysis_text="CRITICAL: China export ban could cut 15% of datacenter revenue.",
    )
    memory = _make_standard_mock_memory()
    tavily = MagicMock()
    tavily.search_as_sources = MagicMock(return_value=[])

    agent = RiskAssessorAgent(llm=llm, memory=memory, tavily=tavily)
    result = await agent.run("NVIDIA risks", session_id="test-session")

    assert result.agent_name == "risk_assessor"
    assert "China export ban" in result.content


# ============================================================================
# Edge case tests
# ============================================================================


@pytest.mark.asyncio
async def test_agent_memory_store_failure_doesnt_crash():
    """Agent should still return result even if memory.store_research fails."""
    from langchain_core.messages import AIMessage

    async def mock_ainvoke(messages, **kwargs):
        return AIMessage(content="", tool_calls=[
            {"name": "write_analysis", "args": {"content": "Analysis done."}, "id": "c1"}
        ])

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    memory = AsyncMock()
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock(side_effect=Exception("Redis down"))

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory)
    result = await agent.run("test", session_id="test-session")

    # Should still return despite store failure
    assert result.content == "Analysis done."


@pytest.mark.asyncio
async def test_agent_with_user_profile_context():
    """Agent should include user profile in the initial message."""
    from langchain_core.messages import AIMessage

    received_messages = []

    async def mock_ainvoke(messages, **kwargs):
        received_messages.extend(messages)
        return AIMessage(content="", tool_calls=[
            {"name": "write_analysis", "args": {"content": "Personalized analysis."}, "id": "c1"}
        ])

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    memory = AsyncMock()
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory)
    result = await agent.run(
        "NVIDIA investment",
        session_id="test-session",
        user_profile_context="[USER PROFILE]\nAge: 23-27\nRisk: moderate",
    )

    # Verify user profile was included in the human message
    human_msg = [m for m in received_messages if hasattr(m, 'content') and 'USER PROFILE' in str(m.content)]
    assert len(human_msg) > 0
    assert result.content == "Personalized analysis."


@pytest.mark.asyncio
async def test_agent_tool_call_failure_doesnt_crash():
    """If a tool throws an exception mid-loop, agent should handle gracefully."""
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Call a search that will fail internally (mocked memory raises)
            return AIMessage(content="", tool_calls=[
                {"name": "search_past_research", "args": {"query": "test"}, "id": "c1"}
            ])
        else:
            # After failure, write analysis anyway
            return AIMessage(content="", tool_calls=[
                {"name": "write_analysis", "args": {"content": "Analysis despite tool failure."}, "id": "c2"}
            ])

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(side_effect=Exception("Qdrant down"))
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory)
    result = await agent.run("test", session_id="test-session")

    assert "Analysis despite tool failure" in result.content


@pytest.mark.asyncio
async def test_agent_multiple_web_searches():
    """Agent should be able to call search_web multiple times in one loop."""
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AIMessage(content="", tool_calls=[
                {"name": "search_web", "args": {"query": "NVIDIA earnings"}, "id": "c1"}
            ])
        elif call_count == 2:
            return AIMessage(content="", tool_calls=[
                {"name": "search_web", "args": {"query": "NVIDIA China risk", "topic": "news"}, "id": "c2"}
            ])
        else:
            return AIMessage(content="", tool_calls=[
                {"name": "write_analysis", "args": {"content": "Comprehensive analysis with multiple searches."}, "id": "c3"}
            ])

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    tavily = MagicMock()
    tavily.search = MagicMock(return_value={
        "results": [{"title": "Result", "content": "Data", "url": "http://x", "published_date": ""}],
    })

    memory = AsyncMock()
    memory.update_agent_status = AsyncMock()
    memory.store_research = AsyncMock()

    agent = _make_concrete_agent(llm=MockLLM(), memory=memory, tavily=tavily)
    result = await agent.run("NVIDIA full analysis", session_id="test-session")

    assert "Comprehensive analysis" in result.content
    # Tavily should have been called twice
    assert tavily.search.call_count == 2
