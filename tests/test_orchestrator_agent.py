"""Tests for the orchestrator agent — ReAct loop with tool calling.

Tests the orchestrator tools (dispatch_agents, search_memory, build_report),
the StateGraph loop, stopping conditions, and edge cases.

Run: uv run pytest tests/test_orchestrator_agent.py -v
"""

import os
import sys
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from investment_research_system.models.schemas import AgentResponse, ResearchQuery


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_agent(name: str, content: str = "test output", confidence: float = 0.8):
    """Helper: create a mock agent that returns a predictable AgentResponse."""
    agent = AsyncMock()
    agent.name = name
    agent.run = AsyncMock(return_value=AgentResponse(
        agent_name=name,
        content=content,
        confidence=confidence,
        sources=[],
        tokens_used=100,
        cost_usd=0.001,
        latency_ms=500,
    ))
    return agent


def _make_mock_circuit_breaker(can_call: bool = True):
    """Helper: create a mock circuit breaker."""
    cb = MagicMock()
    cb.can_call = MagicMock(return_value=can_call)
    cb.record_success = MagicMock()
    cb.record_failure = MagicMock()
    return cb


# ============================================================================
# Task 1: Config
# ============================================================================


def test_config_has_orchestrator_model_field():
    """Settings should have orchestrator_model field defaulting to empty string."""
    from config import Settings
    s = Settings()
    assert hasattr(s, "orchestrator_model")
    assert s.orchestrator_model == ""


# ============================================================================
# Task 2: dispatch_agents tool
# ============================================================================


@pytest.mark.asyncio
async def test_dispatch_agents_tool_runs_agents_in_parallel():
    """dispatch_agents should run requested agents and return formatted text."""
    from investment_research_system.orchestrator.tools import create_dispatch_agents_tool

    agents = {
        "researcher": _make_mock_agent("researcher", "NVIDIA P/E is 52"),
        "sentiment": _make_mock_agent("sentiment", "Bullish consensus"),
    }
    cb = _make_mock_circuit_breaker()

    tool_fn = create_dispatch_agents_tool(agents=agents, circuit_breaker=cb, max_retries=1, retry_delay=0.0)

    context = {
        "query": "NVIDIA investment",
        "session_id": "test-session",
        "user_profile_context": "",
        "new_responses": [],
        "new_failures": [],
    }

    result = await tool_fn(["researcher", "sentiment"], _context=context)

    assert "researcher" in result
    assert "sentiment" in result
    assert "NVIDIA P/E is 52" in result
    assert len(context["new_responses"]) == 2
    assert len(context["new_failures"]) == 0


@pytest.mark.asyncio
async def test_dispatch_agents_tool_rejects_invalid_agent_name():
    """dispatch_agents should reject agent names not in the registry."""
    from investment_research_system.orchestrator.tools import create_dispatch_agents_tool

    agents = {"researcher": _make_mock_agent("researcher")}
    cb = _make_mock_circuit_breaker()

    tool_fn = create_dispatch_agents_tool(agents=agents, circuit_breaker=cb, max_retries=1, retry_delay=0.0)
    context = {"query": "test", "session_id": "s", "user_profile_context": "", "new_responses": [], "new_failures": []}

    result = await tool_fn(["researcher", "nonexistent"], _context=context)

    assert len(context["new_responses"]) == 1
    assert "nonexistent" in result


@pytest.mark.asyncio
async def test_dispatch_agents_tool_handles_agent_failure():
    """dispatch_agents should report failures gracefully."""
    from investment_research_system.orchestrator.tools import create_dispatch_agents_tool

    failing_agent = AsyncMock()
    failing_agent.name = "researcher"
    failing_agent.run = AsyncMock(side_effect=Exception("LLM timeout"))

    agents = {"researcher": failing_agent}
    cb = _make_mock_circuit_breaker()

    tool_fn = create_dispatch_agents_tool(agents=agents, circuit_breaker=cb, max_retries=0, retry_delay=0.0)
    context = {"query": "test", "session_id": "s", "user_profile_context": "", "new_responses": [], "new_failures": []}

    result = await tool_fn(["researcher"], _context=context)

    assert len(context["new_responses"]) == 0
    assert len(context["new_failures"]) == 1
    assert "FAILED" in result


# ============================================================================
# Task 3: search_memory and build_report tools
# ============================================================================


@pytest.mark.asyncio
async def test_search_memory_tool_formats_results():
    """search_memory should format Qdrant + Mem0 results as labeled text."""
    from investment_research_system.orchestrator.tools import create_search_memory_tool

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={
        "long_term": [{"text": "NVIDIA P/E was 58 in Jan", "score": 0.87}],
        "semantic": [{"memory": "NVIDIA competes with AMD"}],
        "topics": ["nvidia"],
    })

    tool_fn = create_search_memory_tool(memory_manager=memory)
    result = await tool_fn("NVIDIA investment")

    assert "[RESEARCH-1]" in result
    assert "0.87" in result
    assert "[FACT-1]" in result
    assert "AMD" in result


@pytest.mark.asyncio
async def test_search_memory_tool_returns_empty_message():
    """search_memory should return a clear message when nothing is found."""
    from investment_research_system.orchestrator.tools import create_search_memory_tool

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})

    tool_fn = create_search_memory_tool(memory_manager=memory)
    result = await tool_fn("something obscure")

    assert "No relevant past research found" in result


@pytest.mark.asyncio
async def test_search_memory_tool_handles_failure():
    """search_memory should handle memory backend failure gracefully."""
    from investment_research_system.orchestrator.tools import create_search_memory_tool

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(side_effect=Exception("Qdrant down"))

    tool_fn = create_search_memory_tool(memory_manager=memory)
    result = await tool_fn("test")

    assert "Memory search failed" in result


@pytest.mark.asyncio
async def test_build_report_tool_zero_responses():
    """build_report should handle zero agent responses gracefully."""
    from investment_research_system.orchestrator.tools import create_build_report_tool

    llm = AsyncMock()
    memory = AsyncMock()

    tool_fn = create_build_report_tool(orchestrator_llm=llm, memory_manager=memory)

    context = {
        "all_responses": [],
        "all_failures": [{"agent": "researcher", "error_type": "timeout", "error_message": "timed out"}],
        "new_responses": [],
        "new_failures": [],
        "research_query": ResearchQuery(query="test query"),
        "start_time": time.time(),
        "report": None,
    }

    result = await tool_fn("no data available", _context=context)

    assert context["report"] is not None
    assert "Unable to gather sufficient data" in context["report"].summary
    assert len(context["report"].failed_agents) == 1


@pytest.mark.asyncio
async def test_build_report_tool_with_responses():
    """build_report should synthesize agent responses into a report."""
    from investment_research_system.orchestrator.tools import create_build_report_tool

    mock_response = MagicMock()
    mock_response.content = "Synthesized report about NVIDIA..."
    mock_response.usage_metadata = {"input_tokens": 500, "output_tokens": 200}

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=mock_response)
    memory = AsyncMock()

    tool_fn = create_build_report_tool(orchestrator_llm=llm, memory_manager=memory)

    responses = [
        AgentResponse(agent_name="researcher", content="NVIDIA P/E is 52", confidence=0.8, sources=[], tokens_used=100, cost_usd=0.001, latency_ms=500),
        AgentResponse(agent_name="sentiment", content="Bullish consensus", confidence=0.75, sources=[], tokens_used=80, cost_usd=0.001, latency_ms=400),
    ]

    context = {
        "all_responses": responses,
        "all_failures": [],
        "new_responses": [],
        "new_failures": [],
        "research_query": ResearchQuery(query="NVIDIA investment"),
        "start_time": time.time(),
        "report": None,
    }

    result = await tool_fn("2 agents responded with high confidence", _context=context)

    assert result == "Report built successfully."
    assert context["report"] is not None
    assert "Synthesized report" in context["report"].summary
    assert len(context["report"].agent_responses) == 2


# ============================================================================
# Task 4: Full orchestrator loop
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_run_produces_report():
    """Full orchestrator.run() should produce a ResearchReport."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Orchestrator decides to dispatch agents
            return AIMessage(content="", tool_calls=[
                {"name": "dispatch_agents", "args": {"agent_names": ["researcher"]}, "id": "call_1"}
            ])
        elif call_count == 2:
            # Orchestrator decides to build report
            return AIMessage(content="", tool_calls=[
                {"name": "build_report", "args": {"reasoning": "1 agent responded"}, "id": "call_2"}
            ])
        elif call_count == 3:
            # Synthesis LLM call inside build_report tool
            return AIMessage(content="Synthesized NVIDIA research report", usage_metadata={"input_tokens": 100, "output_tokens": 50})
        else:
            # Orchestrator loop comes back after build_report — no more tool calls
            return AIMessage(content="Research complete.")

    class MockLLM:
        """Mock LLM that returns proper AIMessage objects for all calls."""
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    orchestrator_llm = MockLLM()

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.graph = None

    agents = [_make_mock_agent("researcher", "NVIDIA analysis", 0.85)]

    orchestrator = ResearchOrchestrator(
        agents=agents,
        memory_manager=memory,
        orchestrator_llm=orchestrator_llm,
    )

    query = ResearchQuery(query="Should I invest in NVIDIA?")
    report = await orchestrator.run(query)

    assert report is not None
    assert report.query == query
    assert report.summary != ""


@pytest.mark.asyncio
async def test_orchestrator_max_iterations_forces_report():
    """When max iterations exhausted, orchestrator should force a report."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator

    async def infinite_search(messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content="", tool_calls=[
            {"name": "search_memory", "args": {"query": "test"}, "id": "call_inf"}
        ])

    orchestrator_llm = MagicMock()
    orchestrator_llm.ainvoke = infinite_search
    orchestrator_llm.bind_tools = MagicMock(return_value=orchestrator_llm)

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.graph = None

    orchestrator = ResearchOrchestrator(
        agents=[],
        memory_manager=memory,
        orchestrator_llm=orchestrator_llm,
        max_iterations=2,
    )

    query = ResearchQuery(query="test")
    report = await orchestrator.run(query)

    assert report is not None
    assert "Unable to gather" in report.summary


# ============================================================================
# Task 6: Edge case tests
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_no_llm_configured():
    """Orchestrator with no LLM should still produce a forced report."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.graph = None

    orchestrator = ResearchOrchestrator(
        agents=[],
        memory_manager=memory,
        orchestrator_llm=None,
    )

    query = ResearchQuery(query="test")
    report = await orchestrator.run(query)

    assert report is not None


def test_circuit_breaker_trips_after_failures():
    """CircuitBreaker should trip to OPEN after max_failures."""
    from investment_research_system.orchestrator.graph import CircuitBreaker

    cb = CircuitBreaker(max_failures=2)
    assert cb.can_call("agent_a") is True

    cb.record_failure("agent_a")
    assert cb.can_call("agent_a") is True  # 1 failure, not tripped

    cb.record_failure("agent_a")
    assert cb.can_call("agent_a") is False  # 2 failures, tripped


def test_circuit_breaker_resets_on_success():
    """CircuitBreaker should reset to CLOSED on success."""
    from investment_research_system.orchestrator.graph import CircuitBreaker

    cb = CircuitBreaker(max_failures=2)
    cb.record_failure("agent_a")
    cb.record_success("agent_a")
    assert cb.can_call("agent_a") is True
    assert cb.get_state("agent_a") == "closed"
