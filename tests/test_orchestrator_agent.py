"""Tests for the plan-then-execute orchestrator.

Tests the orchestrator tools (dispatch_agents, search_memory, build_report),
plan validation, the StateGraph loop, replanning, and edge cases.

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
# Task 4: Agent filtering (code-level pre-check)
# ============================================================================


def test_filter_skips_researcher_with_enough_memory():
    """Researcher should be skipped when memory has ≥3 high-relevance items."""
    from investment_research_system.orchestrator.graph import filter_available_agents

    memory_context = (
        "[RESEARCH-1] (relevance: 0.92): NVIDIA Q4 revenue was $39.3B\n"
        "[RESEARCH-2] (relevance: 0.88): NVIDIA operating margin 57%\n"
        "[RESEARCH-3] (relevance: 0.90): NVIDIA data center revenue $35.1B\n"
        "[FACT-1]: NVIDIA P/E is 58"
    )
    skip = filter_available_agents(memory_context)
    assert skip["researcher"] is True
    assert skip["analyst"] is False  # never skipped
    assert skip["risk_assessor"] is False  # never skipped


def test_filter_keeps_researcher_with_few_memory_items():
    """Researcher should NOT be skipped with <3 high-relevance items."""
    from investment_research_system.orchestrator.graph import filter_available_agents

    memory_context = (
        "[RESEARCH-1] (relevance: 0.92): NVIDIA Q4 revenue was $39.3B\n"
        "[RESEARCH-2] (relevance: 0.60): Some low relevance result\n"
    )
    skip = filter_available_agents(memory_context)
    assert skip["researcher"] is False  # only 1 high-relevance item


def test_filter_skips_sentiment_with_sentiment_facts():
    """Sentiment should be skipped when memory has sentiment-related facts."""
    from investment_research_system.orchestrator.graph import filter_available_agents

    memory_context = (
        "[RESEARCH-1] (relevance: 0.70): Some research\n"
        "[FACT-1]: Market sentiment is bullish for NVIDIA\n"
    )
    skip = filter_available_agents(memory_context)
    assert skip["sentiment"] is True


def test_filter_keeps_sentiment_without_sentiment_facts():
    """Sentiment should NOT be skipped when no sentiment data in memory."""
    from investment_research_system.orchestrator.graph import filter_available_agents

    memory_context = (
        "[RESEARCH-1] (relevance: 0.92): Revenue data\n"
        "[FACT-1]: NVIDIA P/E is 58\n"
    )
    skip = filter_available_agents(memory_context)
    assert skip["sentiment"] is False


def test_filter_never_skips_analyst_or_risk():
    """Analyst and risk_assessor should NEVER be skipped."""
    from investment_research_system.orchestrator.graph import filter_available_agents

    # Even with tons of memory data
    memory_context = (
        "[RESEARCH-1] (relevance: 0.95): Data 1\n"
        "[RESEARCH-2] (relevance: 0.93): Data 2\n"
        "[RESEARCH-3] (relevance: 0.91): Data 3\n"
        "[FACT-1]: Market sentiment is very bullish\n"
        "[FACT-2]: Analyst rating is strong buy\n"
    )
    skip = filter_available_agents(memory_context)
    assert skip["analyst"] is False
    assert skip["risk_assessor"] is False


def test_get_available_agents_filters_correctly():
    """get_available_agents should return only non-skipped agents."""
    from investment_research_system.orchestrator.graph import get_available_agents

    # Enough to skip researcher + sentiment
    memory_context = (
        "[RESEARCH-1] (relevance: 0.95): Data 1\n"
        "[RESEARCH-2] (relevance: 0.93): Data 2\n"
        "[RESEARCH-3] (relevance: 0.91): Data 3\n"
        "[FACT-1]: Bearish sentiment dominates\n"
    )
    available = get_available_agents(memory_context)
    assert "researcher" not in available
    assert "sentiment" not in available
    assert "analyst" in available
    assert "risk_assessor" in available


def test_get_available_agents_no_memory():
    """With no memory data, all agents should be available."""
    from investment_research_system.orchestrator.graph import get_available_agents

    available = get_available_agents("No relevant past research found.")
    assert len(available) == 4


def test_default_plan_respects_available_agents():
    """default_plan should exclude filtered-out agents."""
    from investment_research_system.orchestrator.graph import default_plan

    plan = default_plan(["analyst", "risk_assessor"])
    # No gatherers step, only analyzers + report
    assert len(plan) == 2
    agent_step = plan[0]
    assert agent_step["action"] == "dispatch_agents"
    assert "researcher" not in agent_step["agent_names"]
    assert "sentiment" not in agent_step["agent_names"]


# ============================================================================
# Task 5: Plan validation
# ============================================================================


def test_validate_plan_valid():
    """Valid plan should pass through unchanged."""
    from investment_research_system.orchestrator.graph import validate_plan

    plan = [
        {"action": "dispatch_agents", "agent_names": ["researcher", "sentiment"]},
        {"action": "dispatch_agents", "agent_names": ["analyst"]},
        {"action": "build_report"},
    ]
    result = validate_plan(plan)
    assert len(result) == 3
    assert result[-1]["action"] == "build_report"


def test_validate_plan_adds_missing_build_report():
    """Plan without build_report should get one appended."""
    from investment_research_system.orchestrator.graph import validate_plan

    plan = [
        {"action": "dispatch_agents", "agent_names": ["researcher"]},
    ]
    result = validate_plan(plan)
    assert result[-1]["action"] == "build_report"
    assert len(result) == 2


def test_validate_plan_removes_invalid_actions():
    """Invalid actions should be stripped out."""
    from investment_research_system.orchestrator.graph import validate_plan

    plan = [
        {"action": "hack_the_mainframe"},
        {"action": "dispatch_agents", "agent_names": ["researcher"]},
        {"action": "build_report"},
    ]
    result = validate_plan(plan)
    assert len(result) == 2
    assert result[0]["action"] == "dispatch_agents"


def test_validate_plan_removes_invalid_agent_names():
    """Invalid agent names should be stripped from dispatch_agents steps."""
    from investment_research_system.orchestrator.graph import validate_plan

    plan = [
        {"action": "dispatch_agents", "agent_names": ["researcher", "fake_agent"]},
        {"action": "build_report"},
    ]
    result = validate_plan(plan)
    assert result[0]["agent_names"] == ["researcher"]


def test_validate_plan_skips_dispatch_with_no_valid_agents():
    """dispatch_agents with only invalid names should be removed entirely."""
    from investment_research_system.orchestrator.graph import validate_plan

    plan = [
        {"action": "dispatch_agents", "agent_names": ["fake1", "fake2"]},
        {"action": "build_report"},
    ]
    result = validate_plan(plan)
    assert len(result) == 1
    assert result[0]["action"] == "build_report"


def test_validate_plan_empty_input():
    """Empty plan should produce just build_report."""
    from investment_research_system.orchestrator.graph import validate_plan

    result = validate_plan([])
    assert len(result) == 1
    assert result[0]["action"] == "build_report"


def test_validate_plan_max_steps():
    """Plan exceeding MAX_PLAN_STEPS should be truncated."""
    from investment_research_system.orchestrator.graph import validate_plan, MAX_PLAN_STEPS

    plan = [{"action": "dispatch_agents", "agent_names": ["researcher"]}] * (MAX_PLAN_STEPS + 3)
    result = validate_plan(plan)
    # Truncated to MAX_PLAN_STEPS + build_report appended
    assert len(result) <= MAX_PLAN_STEPS + 1


# ============================================================================
# Task 5: Plan parsing from LLM output
# ============================================================================


def test_parse_plan_clean_json():
    """Clean JSON array should parse directly."""
    from investment_research_system.orchestrator.graph import parse_plan_from_llm

    text = '[{"action": "dispatch_agents", "agent_names": ["researcher"]}, {"action": "build_report"}]'
    result = parse_plan_from_llm(text)
    assert len(result) == 2


def test_parse_plan_markdown_code_block():
    """JSON wrapped in markdown code block should parse."""
    from investment_research_system.orchestrator.graph import parse_plan_from_llm

    text = '```json\n[{"action": "build_report"}]\n```'
    result = parse_plan_from_llm(text)
    assert len(result) == 1


def test_parse_plan_with_extra_text():
    """JSON array with surrounding text should still parse."""
    from investment_research_system.orchestrator.graph import parse_plan_from_llm

    text = 'Here is my plan:\n[{"action": "build_report"}]\nThis should work.'
    result = parse_plan_from_llm(text)
    assert len(result) == 1


def test_parse_plan_garbage_returns_empty():
    """Unparseable text should return empty list."""
    from investment_research_system.orchestrator.graph import parse_plan_from_llm

    result = parse_plan_from_llm("this is not json at all")
    assert result == []


# ============================================================================
# Task 6: Full orchestrator loop
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_run_produces_report():
    """Full orchestrator.run() should produce a ResearchReport via plan-then-execute."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # create_plan: LLM returns a JSON plan
            return AIMessage(content='[{"action": "dispatch_agents", "agent_names": ["researcher"]}, {"action": "build_report"}]')
        else:
            # build_report synthesis call
            return AIMessage(content="Synthesized NVIDIA research report", usage_metadata={"input_tokens": 100, "output_tokens": 50})

    class MockLLM:
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
    assert len(report.agent_responses) >= 1


@pytest.mark.asyncio
async def test_orchestrator_no_llm_uses_default_plan():
    """Orchestrator with no LLM should use default plan and still produce a report."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.graph = None

    agents = [_make_mock_agent("researcher"), _make_mock_agent("sentiment")]

    orchestrator = ResearchOrchestrator(
        agents=agents,
        memory_manager=memory,
        orchestrator_llm=None,
    )

    query = ResearchQuery(query="test query")
    report = await orchestrator.run(query)

    assert report is not None


@pytest.mark.asyncio
async def test_orchestrator_replan_on_agent_failure():
    """When an agent fails, orchestrator should replan and still produce a report."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator
    from langchain_core.messages import AIMessage

    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # create_plan: dispatch researcher (will fail) + sentiment
            return AIMessage(content='[{"action": "dispatch_agents", "agent_names": ["researcher"]}, {"action": "dispatch_agents", "agent_names": ["sentiment"]}, {"action": "build_report"}]')
        elif call_count == 2:
            # replan: skip researcher, just build report
            return AIMessage(content='[{"action": "build_report"}]')
        else:
            # synthesis
            return AIMessage(content="Report based on partial data", usage_metadata={"input_tokens": 50, "output_tokens": 30})

    class MockLLM:
        async def ainvoke(self, messages, **kwargs):
            return await mock_ainvoke(messages, **kwargs)
        def bind_tools(self, tools):
            return self

    # Researcher fails, sentiment succeeds
    failing_researcher = AsyncMock()
    failing_researcher.name = "researcher"
    failing_researcher.run = AsyncMock(side_effect=Exception("timeout"))

    agents = [failing_researcher, _make_mock_agent("sentiment", "Market is bullish")]

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.graph = None

    orchestrator = ResearchOrchestrator(
        agents=agents,
        memory_manager=memory,
        orchestrator_llm=MockLLM(),
    )

    query = ResearchQuery(query="NVIDIA analysis")
    report = await orchestrator.run(query)

    assert report is not None
    # Should have called replan
    assert call_count >= 2


# ============================================================================
# Task 7: Circuit breaker (unchanged)
# ============================================================================


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
