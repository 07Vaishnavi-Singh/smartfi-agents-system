# Orchestrator Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static LangGraph pipeline with an LLM-powered orchestrator agent that uses a custom tool-calling ReAct loop with 3 tools (dispatch_agents, search_memory, build_report).

**Architecture:** Custom LangGraph `StateGraph` with a message-based ReAct loop. The orchestrator LLM reasons over a growing message history, calls tools via `bind_tools()`, and a conditional edge decides whether to loop (execute more tools) or exit (build report). Tools are defined using LangChain's `@tool` decorator as async bound methods on the orchestrator class.

**Tech Stack:** LangGraph (StateGraph, add_messages), LangChain (ChatGoogleGenerativeAI/ChatAnthropic, @tool, bind_tools), asyncio, existing agent/memory/conflict/quality infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-04-orchestrator-agent-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/config.py` | Modify | Add `orchestrator_model` field |
| `src/investment_research_system/orchestrator/tools.py` | Create | 3 tool implementation functions |
| `src/investment_research_system/orchestrator/graph.py` | Rewrite | ReAct loop StateGraph, OrchestratorAgentState, ResearchOrchestrator |
| `src/investment_research_system/api/dependencies.py` | Modify | Pass memory_manager + orchestrator_llm to orchestrator |
| `tests/test_orchestrator_agent.py` | Create | Tests for tools, graph loop, stopping conditions |

---

## Chunk 1: Config + Tools

### Task 1: Add orchestrator_model config field

**Files:**
- Modify: `src/config.py:26`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator_agent.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def test_config_has_orchestrator_model_field():
    """Settings should have orchestrator_model field defaulting to empty string."""
    from config import Settings
    s = Settings()
    assert hasattr(s, "orchestrator_model")
    assert s.orchestrator_model == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/vaiz_07/Developer/ai-projects/multi-agent-orchestration && uv run pytest tests/test_orchestrator_agent.py::test_config_has_orchestrator_model_field -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Add the field to config.py**

In `src/config.py`, add after line 31 (`temperature: float = 0.7`):

```python
    # --- Orchestrator Agent ---
    # If empty, uses default_model. Override to use a more capable model
    # for the orchestrator's tool-calling loop (needs reliable bind_tools support).
    orchestrator_model: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestrator_agent.py::test_config_has_orchestrator_model_field -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_orchestrator_agent.py
git commit -m "feat: add orchestrator_model config field"
```

---

### Task 2: Implement dispatch_agents tool

**Files:**
- Create: `src/investment_research_system/orchestrator/tools.py`
- Test: `tests/test_orchestrator_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator_agent.py — add to the file

import pytest
from unittest.mock import AsyncMock, MagicMock
from investment_research_system.models.schemas import AgentResponse


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

    # Simulate tool context
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

    # Should still run the valid agent
    assert len(context["new_responses"]) == 1
    assert "nonexistent" in result  # should mention the invalid name


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator_agent.py -k "dispatch" -v`
Expected: FAIL with ModuleNotFoundError (tools.py doesn't exist)

- [ ] **Step 3: Implement dispatch_agents tool**

Create `src/investment_research_system/orchestrator/tools.py`:

```python
"""Orchestrator agent tools — the actions the orchestrator LLM can take.

Each tool is created via a factory function that captures dependencies
(agents, memory_manager, circuit_breaker) via closure. The orchestrator
class calls these factories in __init__ and binds the returned functions
as LangChain tools.

PYTHON CONCEPT — FACTORY + CLOSURE:
Instead of defining tools as class methods (which don't work with
LangChain's @tool decorator on bound methods), we use factory functions
that return async functions with captured dependencies.
TS equivalent: a function that returns another function with closed-over variables.
Rust equivalent: a closure that captures references from its environment.

WHY ASYNC TOOLS:
The orchestrator runs inside FastAPI's event loop. Tools must be async
to avoid blocking. LangGraph supports async tool execution natively.
"""

import asyncio
import logging

from investment_research_system.errors import (
    AgentError,
    BudgetExceededError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    MemoryUnavailableError,
)

logger = logging.getLogger(__name__)

# Cost constants — same as in base.py
COST_PER_INPUT_TOKEN = 3.0 / 1_000_000
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000

LLM_CALL_TIMEOUT_SECONDS = 30


async def _run_agent_with_retry(agent, query, session_id, user_profile_context, circuit_breaker, max_retries, retry_delay):
    """Run a single agent with retry + circuit breaker.

    Reuses the same retry/error logic from the old orchestrator.
    Returns (AgentResponse | None, failure_dict | None).
    """
    if not circuit_breaker.can_call(agent.name):
        logger.warning("[tools] Skipping %s — circuit breaker OPEN", agent.name)
        return None, {"agent": agent.name, "error_type": "circuit_breaker", "error_message": "Circuit breaker open"}

    last_error_type = "unknown"
    last_error_message = "Unknown error"

    for attempt in range(max_retries + 1):
        try:
            response = await agent.run(query, session_id, user_profile_context)
            circuit_breaker.record_success(agent.name)
            return response, None

        except BudgetExceededError:
            return None, {"agent": agent.name, "error_type": "budget_exceeded", "error_message": "Budget exceeded"}

        except LLMRefusalError as e:
            return None, {"agent": agent.name, "error_type": "refused", "error_message": str(e)}

        except LLMRateLimitError as e:
            last_error_type, last_error_message = "rate_limit", str(e)
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2 ** attempt))

        except LLMTimeoutError as e:
            last_error_type = "timeout"
            last_error_message = f"Timed out after {LLM_CALL_TIMEOUT_SECONDS}s"
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))

        except MemoryUnavailableError as e:
            last_error_type, last_error_message = "memory_unavailable", str(e)
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))

        except (AgentError, Exception) as e:
            last_error_type = "agent_error" if isinstance(e, AgentError) else "unexpected"
            last_error_message = str(e)
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))

    circuit_breaker.record_failure(agent.name)
    return None, {"agent": agent.name, "error_type": last_error_type, "error_message": last_error_message}


def create_dispatch_agents_tool(agents, circuit_breaker, max_retries=1, retry_delay=1.0):
    """Factory: create the dispatch_agents async function.

    Args:
        agents: dict of {name: BaseAgent} — the available agents.
        circuit_breaker: CircuitBreaker instance.
        max_retries: max retry attempts per agent.
        retry_delay: base delay between retries.

    Returns:
        An async function that takes (agent_names, _context) and returns str.
    """

    async def dispatch_agents(agent_names: list[str], _context: dict) -> str:
        """Run one or more research agents in parallel.

        Available agents: researcher, sentiment, analyst, risk_assessor.
        Returns each agent's analysis with confidence scores.
        """
        query = _context["query"]
        session_id = _context["session_id"]
        profile = _context.get("user_profile_context", "")

        # Validate agent names
        valid_agents = {}
        invalid_names = []
        for name in agent_names:
            if name in agents:
                valid_agents[name] = agents[name]
            else:
                invalid_names.append(name)

        if not valid_agents and invalid_names:
            return f"ERROR: No valid agents found. Invalid names: {invalid_names}. Available: {list(agents.keys())}"

        # Run valid agents in parallel
        results = await asyncio.gather(
            *[
                _run_agent_with_retry(agent, query, session_id, profile, circuit_breaker, max_retries, retry_delay)
                for agent in valid_agents.values()
            ],
        )

        # Accumulate results
        parts = []
        for response, failure in results:
            if response is not None:
                _context["new_responses"].append(response)
                # Truncate content for the LLM summary (full content is in state)
                preview = response.content[:300].replace("\n", " ")
                parts.append(f"{response.agent_name} (confidence: {response.confidence:.2f}): {preview}")
            elif failure is not None:
                _context["new_failures"].append(failure)
                parts.append(f"FAILED: {failure['agent']} ({failure['error_type']}: {failure['error_message']})")

        if invalid_names:
            parts.append(f"SKIPPED (invalid names): {invalid_names}")

        return " | ".join(parts)

    return dispatch_agents


def create_search_memory_tool(memory_manager):
    """Factory: create the search_memory async function.

    Args:
        memory_manager: MemoryManager instance with parallel_search().

    Returns:
        An async function that takes (query) and returns str.
    """

    async def search_memory(query: str) -> str:
        """Search past research in long-term memory.

        Use this before dispatching agents to check if relevant research already exists.
        """
        try:
            results = await memory_manager.parallel_search(query)
        except Exception as e:
            logger.warning("[tools] Memory search failed: %s", e)
            return "Memory search failed. Proceed by dispatching agents."

        long_term = results.get("long_term", [])
        semantic = results.get("semantic", [])

        if not long_term and not semantic:
            return "No relevant past research found."

        parts = []
        for i, item in enumerate(long_term[:5], 1):
            text = item.get("text", str(item))[:200]
            score = item.get("score", 0.0)
            parts.append(f"[RESEARCH-{i}] (relevance: {score:.2f}): {text}")

        for i, item in enumerate(semantic[:5], 1):
            text = item.get("memory", str(item))[:150]
            parts.append(f"[FACT-{i}]: {text}")

        return "\n".join(parts)

    return search_memory


def create_build_report_tool(orchestrator_llm, memory_manager):
    """Factory: create the build_report async function.

    Args:
        orchestrator_llm: LLM instance for synthesis.
        memory_manager: MemoryManager (for session cleanup).

    Returns:
        An async function that takes (reasoning, _context) and returns str.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    SYNTHESIS_PROMPT = """You are the SYNTHESIS editor in a multi-agent investment research pipeline.

You receive reports from specialized agents:
- RESEARCHER: raw facts, data points, sources (evidence layer)
- ANALYST: financial interpretation, valuation, metrics (numbers layer)
- SENTIMENT: market mood, narrative, insider activity (psychology layer)
- RISK ASSESSOR: risks, threats, devil's advocate view (adversarial layer)

Your job: Merge these into ONE coherent research summary that a decision-maker can act on.

Synthesis rules:
- NEVER add information that wasn't in the agent reports
- When agents AGREE: state the consensus concisely
- When agents DISAGREE: present the tension explicitly
- Weight reliability: Researcher's confirmed facts > Analyst's metrics > Sentiment signals
- Preserve uncertainty: if an agent flagged low confidence or data gaps, carry that through

Structure (aim for 400-600 words):
1. **Executive Summary** — 2-3 sentences
2. **Key Findings** — top 3-5 facts
3. **Financial Analysis** — valuation assessment
4. **Market Sentiment** — sentiment score, narrative
5. **Risk Assessment** — top 2-3 risks with severity
6. **Conclusion** — balanced synthesis

Do NOT give buy/sell recommendations — present evidence and let the reader decide."""

    async def build_report(reasoning: str, _context: dict) -> str:
        """Synthesize all collected agent responses into the final research report.

        Call this when you have sufficient data to answer the user's question.
        The reasoning parameter should explain your assessment of data quality and completeness.
        """
        from uuid import uuid4

        from investment_research_system.models.schemas import ResearchReport
        from investment_research_system.orchestrator.conflict import detect_conflicts, format_conflicts_summary
        from investment_research_system.orchestrator.quality import assess_quality, format_quality_summary

        # Merge all_responses + new_responses from this turn's dispatch_agents calls.
        # This handles the case where dispatch_agents and build_report are called
        # in the same LLM turn — new_responses wouldn't be in all_responses yet.
        agent_responses = _context.get("all_responses", []) + _context.get("new_responses", [])
        failed_agents = _context.get("all_failures", []) + _context.get("new_failures", [])
        research_query = _context["research_query"]
        start_time = _context["start_time"]

        logger.info("[tools] build_report called. reasoning: %s", reasoning[:200])

        # Run conflict detection and quality assessment
        conflicts = detect_conflicts(agent_responses)
        quality = assess_quality(
            responses=agent_responses,
            conflicts=conflicts,
            failed_agents=failed_agents,
        )

        conflicts_text = format_conflicts_summary(conflicts)
        quality_text = format_quality_summary(quality)

        total_tokens = sum(r.tokens_used for r in agent_responses)
        total_cost = sum(r.cost_usd for r in agent_responses)

        # Zero responses — skip synthesis, return failure report
        if not agent_responses:
            import time

            report = ResearchReport(
                id=str(uuid4()),
                query=research_query,
                summary="Unable to gather sufficient data to answer this query. All research agents failed or were not dispatched.",
                agent_responses=[],
                failed_agents=failed_agents,
                total_cost_usd=total_cost,
                total_tokens=total_tokens,
                processing_time_seconds=round(time.time() - start_time, 2),
            )
            _context["report"] = report
            return "Report built (no agent data available)."

        # Build agent output text for synthesis
        agent_outputs = []
        for response in agent_responses:
            agent_outputs.append(
                f"=== {response.agent_name.upper()} ===\n"
                f"Confidence: {response.confidence:.2f}\n"
                f"{response.content}"
            )

        human_prompt = (
            f"Original question: {research_query.query}\n\n"
            f"Orchestrator reasoning: {reasoning}\n\n"
            f"Agent reports:\n\n{''.join(o + chr(10) + chr(10) for o in agent_outputs)}"
            f"{conflicts_text}\n\n{quality_text}\n\n"
            f"Synthesize these into a single coherent research summary."
        )

        # Try LLM synthesis
        try:
            response = await orchestrator_llm.ainvoke([
                SystemMessage(content=SYNTHESIS_PROMPT),
                HumanMessage(content=human_prompt),
            ])
            summary = response.content

            # Track synthesis cost
            usage = response.usage_metadata or {}
            synthesis_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            synthesis_cost = synthesis_tokens * 0.5 / 1_000_000
            total_tokens += synthesis_tokens
            total_cost += synthesis_cost

        except Exception as e:
            logger.warning("[tools] LLM synthesis failed, using fallback: %s", e)
            summary = "\n\n".join(agent_outputs)
            summary += f"\n\n---\n{conflicts_text}\n\n{quality_text}"

        import time

        report = ResearchReport(
            id=str(uuid4()),
            query=research_query,
            summary=summary,
            agent_responses=agent_responses,
            failed_agents=failed_agents,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            processing_time_seconds=round(time.time() - start_time, 2),
        )
        _context["report"] = report
        return "Report built successfully."

    return build_report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator_agent.py -k "dispatch" -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/investment_research_system/orchestrator/tools.py tests/test_orchestrator_agent.py
git commit -m "feat: implement orchestrator tools (dispatch_agents, search_memory, build_report)"
```

---

### Task 3: Test search_memory and build_report tools

**Files:**
- Modify: `tests/test_orchestrator_agent.py`

- [ ] **Step 1: Write the tests**

```python
# Add to tests/test_orchestrator_agent.py

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
async def test_build_report_tool_zero_responses():
    """build_report should handle zero agent responses gracefully."""
    from investment_research_system.orchestrator.tools import create_build_report_tool
    from investment_research_system.models.schemas import ResearchQuery

    import time

    llm = AsyncMock()
    memory = AsyncMock()

    tool_fn = create_build_report_tool(orchestrator_llm=llm, memory_manager=memory)

    context = {
        "all_responses": [],
        "all_failures": [{"agent": "researcher", "error_type": "timeout", "error_message": "timed out"}],
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
    from investment_research_system.models.schemas import ResearchQuery
    from langchain_core.messages import AIMessage

    import time

    # Mock LLM that returns a synthesis
    llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "Synthesized report about NVIDIA..."
    mock_response.usage_metadata = {"input_tokens": 500, "output_tokens": 200}
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
        "research_query": ResearchQuery(query="NVIDIA investment"),
        "start_time": time.time(),
        "report": None,
    }

    result = await tool_fn("2 agents responded with high confidence", _context=context)

    assert result == "Report built successfully."
    assert context["report"] is not None
    assert "Synthesized report" in context["report"].summary
    assert len(context["report"].agent_responses) == 2
```

- [ ] **Step 2: Run all tool tests**

Run: `uv run pytest tests/test_orchestrator_agent.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_orchestrator_agent.py
git commit -m "test: add search_memory and build_report tool tests"
```

---

## Chunk 2: Graph Rewrite + Dependencies

### Task 4: Rewrite graph.py with ReAct loop

**Files:**
- Rewrite: `src/investment_research_system/orchestrator/graph.py`

- [ ] **Step 1: Write the failing test for the full orchestrator loop**

```python
# Add to tests/test_orchestrator_agent.py

@pytest.mark.asyncio
async def test_orchestrator_run_produces_report():
    """Full orchestrator.run() should produce a ResearchReport."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator, CircuitBreaker
    from investment_research_system.models.schemas import ResearchQuery
    from langchain_core.messages import AIMessage, ToolCall

    # Mock orchestrator LLM that simulates 2 turns:
    # Turn 1: call dispatch_agents(["researcher", "sentiment"])
    # Turn 2: call build_report
    call_count = 0

    async def mock_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: dispatch agents
            msg = AIMessage(content="", tool_calls=[
                ToolCall(name="dispatch_agents", args={"agent_names": ["researcher"]}, id="call_1")
            ])
            return msg
        else:
            # Second call: build report
            msg = AIMessage(content="", tool_calls=[
                ToolCall(name="build_report", args={"reasoning": "1 agent responded"}, id="call_2")
            ])
            return msg

    # Mock orchestrator LLM
    orchestrator_llm = MagicMock()
    orchestrator_llm.ainvoke = mock_ainvoke
    orchestrator_llm.bind_tools = MagicMock(return_value=orchestrator_llm)

    # Mock memory manager
    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.graph = None

    # Mock agents
    agents = [_make_mock_agent("researcher", "NVIDIA analysis", 0.85)]

    # Create orchestrator
    orchestrator = ResearchOrchestrator(
        agents=agents,
        memory_manager=memory,
        orchestrator_llm=orchestrator_llm,
    )

    query = ResearchQuery(query="Should I invest in NVIDIA?")
    report = await orchestrator.run(query)

    assert report is not None
    assert report.query == query
    assert report.summary != ""  # should have synthesized or forced a summary


@pytest.mark.asyncio
async def test_orchestrator_max_iterations_forces_report():
    """When max iterations exhausted, orchestrator should force a report."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator
    from investment_research_system.models.schemas import ResearchQuery
    from langchain_core.messages import AIMessage, ToolCall

    # LLM that always calls search_memory (never calls build_report)
    async def infinite_search(messages, **kwargs):
        return AIMessage(content="", tool_calls=[
            ToolCall(name="search_memory", args={"query": "test"}, id="call_inf")
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
        max_iterations=2,  # Force low iteration count
    )

    query = ResearchQuery(query="test")
    report = await orchestrator.run(query)

    # Should still produce a report (forced by max iterations with zero agents)
    assert report is not None
    assert "Unable to gather" in report.summary
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_orchestrator_agent.py -k "orchestrator_run or max_iterations" -v`
Expected: FAIL (graph.py still has old code)

- [ ] **Step 3: Rewrite graph.py**

Replace the entire contents of `src/investment_research_system/orchestrator/graph.py` with the new ReAct loop implementation:

```python
"""LangGraph orchestrator agent — LLM-powered ReAct loop.

ALTERNATIVE IMPLEMENTATION — LangGraph's built-in ReAct agent:

    from langgraph.prebuilt import create_react_agent

    orchestrator = create_react_agent(
        model=llm.bind_tools([dispatch_agents, search_memory, build_report]),
        tools=[dispatch_agents, search_memory, build_report],
        prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    )
    result = await orchestrator.ainvoke({"messages": [("user", query)]})

We built the custom loop because:
1. Python-level guardrails (max iterations, budget tracking)
2. Custom state accumulation (agent_responses across turns)
3. Forced report fallback when iterations exhausted
4. Full visibility for debugging and observability

ARCHITECTURE:
The orchestrator LLM reasons over a growing message history and calls
tools (dispatch_agents, search_memory, build_report) via bind_tools().
A conditional edge decides whether to loop or exit.

    START → fetch_user_profile → orchestrator_llm ⟲ execute_tools
                                        ↓ (done)
                                  build_final_report → END
"""

import asyncio
import logging
import operator
import time
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from investment_research_system.agents.base import BaseAgent
from investment_research_system.models.schemas import AgentResponse, ResearchQuery, ResearchReport
from investment_research_system.observability.tracing import get_run_config
from investment_research_system.orchestrator.tools import (
    create_build_report_tool,
    create_dispatch_agents_tool,
    create_search_memory_tool,
)

logger = logging.getLogger(__name__)


MAX_ITERATIONS_DEFAULT = 5

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator agent in a multi-agent investment research system.
You coordinate 4 specialist agents to answer investment research queries.

AVAILABLE AGENTS:
- researcher: Gathers raw facts, data points, financial metrics from the web
- sentiment: Analyzes market mood, analyst ratings, insider activity, news sentiment
- analyst: Interprets financial data, valuation, peer comparison (works best AFTER researcher)
- risk_assessor: Identifies risks, threats, downside scenarios (works best AFTER researcher)

STRATEGY:
1. FIRST call search_memory to check if past research already covers this query
2. If memory has sufficient recent data, you may skip some agents
3. If memory is stale or empty, dispatch gatherers first (researcher, sentiment)
4. After gatherers return, dispatch analyzers (analyst, risk_assessor) — they benefit from fresh data the gatherers stored in memory
5. After all agents return, evaluate: do you have enough to answer the question?
6. Call build_report when you have sufficient data

RULES:
- You can dispatch multiple agents in a single call (they run in parallel)
- Prefer dispatching gatherers before analyzers — analyzers use gatherer data
- If an agent fails, decide whether to retry it or proceed without it
- Do NOT dispatch the same agent twice unless the first attempt failed
- You have a limited number of turns — be efficient, don't over-research
- When in doubt, build the report with what you have rather than looping"""


# =============================================================================
# CIRCUIT BREAKER (unchanged from previous implementation)
# =============================================================================

class CircuitBreaker:
    """Circuit breaker pattern — prevents calling agents that keep failing."""

    def __init__(self, max_failures: int = 3, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self._failures: dict[str, int] = {}
        self._last_failure: dict[str, float] = {}
        self._state: dict[str, str] = {}

    def can_call(self, agent_name: str) -> bool:
        state = self._state.get(agent_name, "closed")
        if state == "closed":
            return True
        if state == "open":
            last_fail = self._last_failure.get(agent_name, 0)
            if time.time() - last_fail > self.reset_timeout:
                self._state[agent_name] = "half_open"
                return True
            return False
        return True

    def record_success(self, agent_name: str) -> None:
        self._failures[agent_name] = 0
        self._state[agent_name] = "closed"
        self._last_failure.pop(agent_name, None)

    def record_failure(self, agent_name: str) -> None:
        self._failures[agent_name] = self._failures.get(agent_name, 0) + 1
        self._last_failure[agent_name] = time.time()
        if self._failures[agent_name] >= self.max_failures:
            self._state[agent_name] = "open"

    def get_state(self, agent_name: str) -> str:
        return self._state.get(agent_name, "closed")


# =============================================================================
# STATE
# =============================================================================

class OrchestratorAgentState(TypedDict):
    """State for the orchestrator agent ReAct loop."""
    messages: Annotated[list[BaseMessage], add_messages]
    research_query: ResearchQuery
    session_id: str
    user_profile_context: str
    agent_responses: Annotated[list[AgentResponse], operator.add]
    failed_agents: Annotated[list[dict], operator.add]
    remaining_iterations: int
    total_cost_usd: float
    report: ResearchReport | None
    start_time: float


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class ResearchOrchestrator:
    """LLM-powered orchestrator agent using a custom ReAct loop.

    The orchestrator LLM decides at runtime which agents to call,
    evaluates results, and decides when to build the final report.
    Tools are bound via LangChain's bind_tools().
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        memory_manager,
        circuit_breaker: CircuitBreaker | None = None,
        orchestrator_llm=None,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        max_iterations: int = MAX_ITERATIONS_DEFAULT,
    ):
        self.agents = {agent.name: agent for agent in agents}
        self.memory_manager = memory_manager
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_iterations = max_iterations

        # Create tool functions via factories
        self._dispatch_fn = create_dispatch_agents_tool(
            agents=self.agents,
            circuit_breaker=self.circuit_breaker,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
        self._search_fn = create_search_memory_tool(memory_manager=self.memory_manager)
        self._report_fn = create_build_report_tool(
            orchestrator_llm=orchestrator_llm,
            memory_manager=self.memory_manager,
        )

        # Build LangChain @tool wrappers for bind_tools schema generation
        self._tools = self._make_langchain_tools()

        # Bind tools to the orchestrator LLM
        if orchestrator_llm is not None:
            try:
                self.orchestrator_llm = orchestrator_llm.bind_tools(self._tools)
                logger.info("[orchestrator] Tools bound to LLM successfully")
            except Exception as e:
                logger.warning("[orchestrator] bind_tools() failed: %s. Using unbound LLM.", e)
                self.orchestrator_llm = orchestrator_llm
        else:
            self.orchestrator_llm = None

        # Mutable context shared with tools during execution
        self._tool_context: dict = {}

        self.graph = self._build_graph()

    def _make_langchain_tools(self):
        """Create LangChain @tool wrappers for schema generation.

        These are thin wrappers that generate the JSON Schema for bind_tools().
        Actual execution goes through self._execute_tool_call() which routes
        to the factory-created functions with the _tool_context.
        """
        @tool
        async def dispatch_agents(agent_names: list[str]) -> str:
            """Run one or more research agents in parallel.
            Available agents: researcher, sentiment, analyst, risk_assessor.
            Returns each agent's analysis with confidence scores."""
            return await self._dispatch_fn(agent_names, _context=self._tool_context)

        @tool
        async def search_memory(query: str) -> str:
            """Search past research in long-term memory. Use this before
            dispatching agents to check if relevant research already exists."""
            return await self._search_fn(query)

        @tool
        async def build_report(reasoning: str) -> str:
            """Synthesize all collected agent responses into the final research report.
            Call this when you have sufficient data to answer the user's question.
            The reasoning parameter should explain your assessment of data quality."""
            return await self._report_fn(reasoning, _context=self._tool_context)

        return [dispatch_agents, search_memory, build_report]

    def _build_graph(self) -> StateGraph:
        """Build the ReAct loop StateGraph."""
        graph = StateGraph(OrchestratorAgentState)

        graph.add_node("fetch_user_profile", self._node_fetch_user_profile)
        graph.add_node("orchestrator_llm", self._node_orchestrator_llm)
        graph.add_node("execute_tools", self._node_execute_tools)
        graph.add_node("build_final_report", self._node_build_final_report)

        graph.add_edge(START, "fetch_user_profile")
        graph.add_edge("fetch_user_profile", "orchestrator_llm")
        graph.add_conditional_edges("orchestrator_llm", self._should_continue, {
            "execute_tools": "execute_tools",
            "build_final_report": "build_final_report",
        })
        graph.add_edge("execute_tools", "orchestrator_llm")
        graph.add_edge("build_final_report", END)

        return graph.compile()

    # =========================================================================
    # CONDITIONAL EDGE
    # =========================================================================

    def _should_continue(self, state: OrchestratorAgentState) -> str:
        """Decide whether to continue the loop or build the report."""
        if state["report"] is not None:
            return "build_final_report"

        messages = state["messages"]
        if not messages:
            return "build_final_report"

        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls and state["remaining_iterations"] > 0:
            return "execute_tools"

        return "build_final_report"

    # =========================================================================
    # NODE FUNCTIONS
    # =========================================================================

    async def _node_fetch_user_profile(self, state: OrchestratorAgentState) -> dict:
        """Fetch user profile from Neo4j once before the loop starts."""
        user_id = state["research_query"].user_id
        if not user_id:
            return {"user_profile_context": ""}

        if not self.memory_manager or not hasattr(self.memory_manager, "graph") or not self.memory_manager.graph:
            return {"user_profile_context": ""}

        try:
            profile = await asyncio.to_thread(
                self.memory_manager.graph.format_profile_for_agents, user_id
            )
            if profile:
                logger.info("[orchestrator] User profile loaded for %s", user_id)
            return {"user_profile_context": profile or ""}
        except Exception as e:
            logger.warning("[orchestrator] Neo4j profile fetch failed: %s", e)
            return {"user_profile_context": ""}

    async def _node_orchestrator_llm(self, state: OrchestratorAgentState) -> dict:
        """Call the orchestrator LLM with the current message history."""
        if self.orchestrator_llm is None:
            logger.error("[orchestrator] No LLM configured")
            return {}

        response = await self.orchestrator_llm.ainvoke(state["messages"])

        # Track orchestrator LLM cost
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
        output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
        orch_cost = (input_tokens + output_tokens) * 0.5 / 1_000_000

        return {
            "messages": [response],
            "total_cost_usd": state["total_cost_usd"] + orch_cost,
        }

    async def _node_execute_tools(self, state: OrchestratorAgentState) -> dict:
        """Execute tool calls from the LLM response."""
        messages = state["messages"]
        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"remaining_iterations": state["remaining_iterations"] - 1}

        # Budget enforcement — skip tool execution if budget exhausted
        max_budget = state["research_query"].max_budget_usd
        if state["total_cost_usd"] >= max_budget:
            logger.warning("[orchestrator] Budget exhausted ($%.4f >= $%.2f), forcing report", state["total_cost_usd"], max_budget)
            # Return a ToolMessage telling the LLM budget is exhausted
            tool_id = last_message.tool_calls[0]["id"]
            return {
                "messages": [ToolMessage(content="BUDGET EXHAUSTED. Call build_report immediately with whatever data you have.", tool_call_id=tool_id)],
                "remaining_iterations": 0,  # Force exit on next check
            }

        # Set up tool context for this iteration
        self._tool_context = {
            "query": state["research_query"].query,
            "session_id": state["session_id"],
            "user_profile_context": state.get("user_profile_context", ""),
            "research_query": state["research_query"],
            "start_time": state["start_time"],
            "new_responses": [],
            "new_failures": [],
            "all_responses": state["agent_responses"],
            "all_failures": state["failed_agents"],
            "report": None,
        }

        # Map tool names to functions
        tool_map = {t.name: t for t in self._tools}

        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in tool_map:
                try:
                    result = await tool_map[tool_name].ainvoke(tool_args)
                except Exception as e:
                    logger.error("[orchestrator] Tool %s failed: %s", tool_name, e)
                    result = f"Tool execution failed: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # Collect new responses/failures from tool context
        new_responses = self._tool_context.get("new_responses", [])
        new_failures = self._tool_context.get("new_failures", [])

        # Update cost with agent costs
        agent_cost = sum(r.cost_usd for r in new_responses)

        # Check if build_report was called
        report = self._tool_context.get("report")
        if report is not None:
            # Update the all_responses on the report to include everything
            all_responses = state["agent_responses"] + new_responses
            all_failures = state["failed_agents"] + new_failures
            report.agent_responses = all_responses
            report.failed_agents = all_failures

        return {
            "messages": tool_messages,
            "agent_responses": new_responses,
            "failed_agents": new_failures,
            "remaining_iterations": state["remaining_iterations"] - 1,
            "total_cost_usd": state["total_cost_usd"] + agent_cost,
            "report": report,
        }

    async def _node_build_final_report(self, state: OrchestratorAgentState) -> dict:
        """Final node — ensure a report exists."""
        if state["report"] is not None:
            logger.info("[orchestrator] Report already built by tool")
            return {}

        # Force-build a report with whatever data exists
        logger.info("[orchestrator] Forcing report build (loop exited without build_report)")

        self._tool_context = {
            "all_responses": state["agent_responses"],
            "all_failures": state["failed_agents"],
            "research_query": state["research_query"],
            "start_time": state["start_time"],
            "report": None,
        }

        await self._report_fn(
            reasoning="Forced report — orchestrator loop exited without calling build_report",
            _context=self._tool_context,
        )

        return {"report": self._tool_context.get("report")}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def run(self, query: ResearchQuery) -> ResearchReport:
        """Execute the orchestrator agent loop.

        This is what the API calls. Same interface as the old static pipeline.
        """
        session_id = str(uuid4())

        # Build initial messages
        user_message = f"Research query: {query.query}"
        if query.focus_areas:
            user_message += f"\nFocus areas: {', '.join(query.focus_areas)}"

        initial_state: OrchestratorAgentState = {
            "messages": [
                SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ],
            "research_query": query,
            "session_id": session_id,
            "user_profile_context": "",
            "agent_responses": [],
            "failed_agents": [],
            "remaining_iterations": self.max_iterations,
            "total_cost_usd": 0.0,
            "report": None,
            "start_time": time.time(),
        }

        logger.info("[orchestrator] Starting research: %s", query.query[:80])

        config = get_run_config(
            query=query.query,
            session_id=session_id,
            run_name=f"research: {query.query[:50]}",
        )

        final_state = await self.graph.ainvoke(initial_state, config=config)

        report = final_state["report"]
        if report is None:
            raise RuntimeError("Orchestrator completed but no report was generated")

        return report
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_orchestrator_agent.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/investment_research_system/orchestrator/graph.py
git commit -m "feat: rewrite orchestrator as LLM-powered ReAct agent with tool-calling loop"
```

---

### Task 5: Update dependencies.py

**Files:**
- Modify: `src/investment_research_system/api/dependencies.py:103-188`

- [ ] **Step 1: Update create_orchestrator to pass memory_manager and orchestrator_llm**

In `src/investment_research_system/api/dependencies.py`, replace line 188 (`return ResearchOrchestrator(agents=agents, max_retries=1)`) with:

```python
        # Create orchestrator LLM — may use a different model than agents
        orchestrator_model = settings.orchestrator_model or settings.default_model
        orchestrator_llm = create_llm(orchestrator_model, settings)
        logger.info("[dependencies] Orchestrator LLM created: %s", orchestrator_model)

        return ResearchOrchestrator(
            agents=agents,
            memory_manager=memory,
            orchestrator_llm=orchestrator_llm,
            max_retries=1,
        )
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `uv run pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 3: Commit**

```bash
git add src/investment_research_system/api/dependencies.py
git commit -m "feat: update dependencies to pass memory_manager and orchestrator_llm to orchestrator"
```

---

### Task 6: Add edge case tests

**Files:**
- Modify: `tests/test_orchestrator_agent.py`

- [ ] **Step 1: Write edge case tests**

```python
# Add to tests/test_orchestrator_agent.py

@pytest.mark.asyncio
async def test_orchestrator_no_llm_configured():
    """Orchestrator with no LLM should still produce a forced report."""
    from investment_research_system.orchestrator.graph import ResearchOrchestrator
    from investment_research_system.models.schemas import ResearchQuery

    memory = AsyncMock()
    memory.parallel_search = AsyncMock(return_value={"long_term": [], "semantic": [], "topics": []})
    memory.graph = None

    orchestrator = ResearchOrchestrator(
        agents=[],
        memory_manager=memory,
        orchestrator_llm=None,  # No LLM
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
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/test_orchestrator_agent.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS (including existing test_graph.py tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_orchestrator_agent.py
git commit -m "test: add edge case tests for orchestrator agent"
```
