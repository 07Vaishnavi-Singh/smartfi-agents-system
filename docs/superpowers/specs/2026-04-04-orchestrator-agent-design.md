# Orchestrator Agent Design — LLM-Powered ReAct Loop

## Summary

Replace the static LangGraph pipeline (hardcoded gatherers → analyzers → conflicts → quality → report) with an LLM-powered orchestrator agent that uses a custom tool-calling loop to decide at runtime which agents to call, in what order, and when it has enough data to build the report.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Minimal ReAct loop with 3 tools | Core pattern first; query classification, sub-query specialization, replanning added later |
| Implementation | LangChain `@tool` + custom `StateGraph` loop | `@tool` for clean schema generation, custom loop for full control and guardrails |
| Fallback | Replace static graph entirely | No feature flag, no dead code. The new orchestrator is the only entry point |
| `create_react_agent` | Documented as reference, not used | Custom loop gives budget tracking, forced report fallback, iteration limits |
| Max iterations | 5 orchestrator LLM turns | Happy path is 4 turns; 5th is buffer for one retry |
| Tools | `dispatch_agents`, `search_memory`, `build_report` | Minimal set of real tools with external effects; no fake wrappers |

## State Design

```python
class OrchestratorAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Growing conversation history
    research_query: ResearchQuery              # Original query object (needed by ResearchReport)
    session_id: str
    user_profile_context: str                  # Neo4j profile, fetched once before loop
    agent_responses: Annotated[list[AgentResponse], operator.add]  # Accumulated via reducer
    failed_agents: Annotated[list[dict], operator.add]             # Accumulated via reducer
    remaining_iterations: int                  # Counts down from 5
    total_cost_usd: float                      # Orchestrator + agent costs
    report: ResearchReport | None              # Set when build_report is called
```

**Reducer annotations:** `messages` uses LangGraph's `add_messages` reducer (auto-appends). `agent_responses` and `failed_agents` use `operator.add` (list concatenation) — tools return only NEW items, LangGraph merges them with the existing list. `remaining_iterations` and `total_cost_usd` are replaced (not accumulated) since they're updated atomically.

**`research_query` field:** Needed because `ResearchReport` schema requires a `ResearchQuery` object. Set once in the initial state from `orchestrator.run(query)`, read by the `build_report` tool.

Messages grow each turn: SystemMessage → HumanMessage → AIMessage (with tool calls) → ToolMessage (results) → repeat.

## Graph Topology

```
START → fetch_user_profile → orchestrator_llm ⟲ execute_tools
                                    ↓ (no tool calls OR iterations exhausted OR report built)
                              build_final_report → END
```

### Nodes

1. **`fetch_user_profile`** — Fetches Neo4j user profile once. Stores in `user_profile_context`. Gracefully returns empty string if Neo4j is down or no user_id. Unchanged from current implementation.

2. **`orchestrator_llm`** — Calls the LLM with full message history + bound tools (`dispatch_agents`, `search_memory`, `build_report`). Returns the AIMessage (which may contain tool calls or plain text).

3. **`execute_tools`** — Receives the LLM's tool calls. Executes each one sequentially (even if the LLM returned multiple tool calls in one turn — sequential execution avoids race conditions on shared `_tool_context`). Appends ToolMessage results to state. Decrements `remaining_iterations` once per node invocation (1 iteration = 1 LLM turn, not 1 per tool call). Routes back to `orchestrator_llm`.

4. **`build_final_report`** — If `build_report` tool was already called, the report exists in state — return it. If the loop exited without calling `build_report` (max iterations or no tool calls), force-build a report from whatever `agent_responses` exist using the existing synthesis logic.

### Conditional Edge

```python
def should_continue(state) -> str:
    # Report already built by tool → done
    if state["report"] is not None:
        return "build_final_report"

    last_message = state["messages"][-1]

    # LLM wants to call tools and we have iterations left → continue loop
    if last_message.tool_calls and state["remaining_iterations"] > 0:
        return "execute_tools"

    # Max iterations hit or LLM stopped calling tools → force report
    return "build_final_report"
```

## Tool Definitions

### Tool-to-State Access Pattern

Tools need access to mutable state (agent registry, memory manager, accumulated responses) but LangGraph's `@tool` functions are plain functions — they don't receive graph state.

**Solution: Class-based tools as bound methods on the orchestrator.**

Tools are defined as methods on `ResearchOrchestrator`. They capture `self` (which holds the agent registry, memory manager, circuit breaker) and receive a mutable `_tool_context` dict that's set before each loop iteration. This avoids closures and keeps the tools testable.

```python
class ResearchOrchestrator:
    def __init__(
        self,
        agents: list[BaseAgent],
        memory_manager: MemoryManager,
        circuit_breaker: CircuitBreaker | None = None,
        orchestrator_llm=None,
        max_retries: int = 1,
        retry_delay: float = 1.0,
    ):
        self.agents = {agent.name: agent for agent in agents}
        self.memory_manager = memory_manager  # Direct access for search_memory tool
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.orchestrator_llm = orchestrator_llm
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Mutable context shared with tools during execution
        self._tool_context: dict = {}
        self.tools = self._make_tools()
        self.graph = self._build_graph()

    def _make_tools(self):
        """Create LangChain tool wrappers bound to this orchestrator instance.

        ASYNC PATTERN: LangGraph supports async tools natively. The @tool
        functions are defined as async and use await directly — no asyncio.run()
        (which would crash inside FastAPI's already-running event loop).
        """
        @tool
        async def dispatch_agents(agents: list[str]) -> str:
            """Run one or more research agents in parallel..."""
            return await self._dispatch_agents_impl(agents)
        # ... same for search_memory, build_report
        return [dispatch_agents, search_memory, build_report]
```

**Why `memory_manager` is a direct parameter:** The current codebase accesses memory through `any_agent.memory` (reaching through agents). This is fragile — it couples the orchestrator to agent internals. The new design receives `MemoryManager` directly via dependency injection, same as agents do. `dependencies.py` already creates the `MemoryManager` instance; it just needs to pass it to the orchestrator too.

**Why async tools, not `asyncio.run()`:** The orchestrator runs inside FastAPI's event loop. `asyncio.run()` creates a NEW event loop and crashes if one is already running. LangGraph supports async `@tool` functions natively — the tool is `async def` and uses `await`, which runs on the existing event loop. This matches the pattern used throughout the codebase (`BaseAgent.run()` is async, `MemoryManager.parallel_search()` is async).

Before each `execute_tools` node invocation, the orchestrator copies relevant state into `_tool_context` (user_profile_context, session_id, research_query). After tool execution, the node reads accumulated results from `_tool_context` and returns them as state updates.

### Tool 1: `dispatch_agents`

```python
@tool
def dispatch_agents(agents: list[str]) -> str:
    """Run one or more research agents in parallel.
    Available agents: researcher, sentiment, analyst, risk_assessor.
    Returns each agent's analysis with confidence scores."""
```

- Validates agent names against `self.agents` registry
- Reuses `_run_agent_with_retry()` + `asyncio.gather()` for parallel execution
- Passes `user_profile_context` from `_tool_context` to each agent
- Accumulates results in `_tool_context["new_responses"]` and `_tool_context["new_failures"]`
- Returns formatted text summary: `"researcher (confidence: 0.82): NVIDIA P/E dropped to 52... | FAILED: analyst (timeout)"`
- Tools return strings (not objects) — LLMs reason better over text

### Tool 2: `search_memory`

```python
@tool
def search_memory(query: str) -> str:
    """Search past research in long-term memory. Use this before dispatching
    agents to check if relevant research already exists."""
```

- Calls `memory_manager.parallel_search(query)` (Qdrant + Mem0 parallel)
- Formats results as `[RESEARCH-N] (relevance: 0.87): ...` and `[FACT-N]: ...`
- Returns `"No relevant past research found."` if empty

### Tool 3: `build_report`

```python
@tool
def build_report(reasoning: str) -> str:
    """Synthesize all collected agent responses into the final research report.
    Call this when you have sufficient data to answer the user's question.
    The reasoning parameter should explain your assessment of data quality and completeness."""
```

- Reads accumulated `agent_responses` from `_tool_context`
- **Zero responses scenario:** If no agents responded, skips LLM synthesis and returns a report with summary: `"Unable to gather sufficient data to answer this query. All research agents failed or were not dispatched."` Quality grade: FAILED.
- Normal path: Runs LLM synthesis (a direct LLM call in the orchestrator, not via `_call_llm_with_fallback` which doesn't exist on BaseAgent — see pre-existing bug note below)
- Runs `detect_conflicts()` and `assess_quality()` on collected responses
- Constructs `ResearchReport`, stores in `_tool_context["report"]`
- Logs the `reasoning` parameter for debugging
- Returns `"Report built successfully."`

### Pre-existing bug: `_call_llm_with_fallback`

The current `_synthesize_summary()` calls `any_agent._call_llm_with_fallback()` which is never defined on `BaseAgent`. The new implementation will call the orchestrator's own LLM directly (`self.orchestrator_llm.ainvoke(messages)`) for synthesis, sidestepping this bug entirely.

## Orchestrator LLM Model

The orchestrator LLM is **separate from the agent LLMs**. It needs reliable tool calling (structured `tool_calls` on AIMessage, not just text mentioning tools). The agents use whatever `config.default_model` is set to (currently Gemini Flash Lite), but the orchestrator needs a model verified to work with LangChain's `bind_tools()`.

**Choice: Use the same model as agents (`config.default_model`) but with `bind_tools()` verified at startup.**

- `dependencies.py` creates the orchestrator LLM via `create_llm()` (same factory as agents)
- On init, `ResearchOrchestrator.__init__` calls `self.orchestrator_llm.bind_tools(self.tools)` and stores the bound version
- If `bind_tools()` fails (model doesn't support tool calling), log a warning and fall back to parsing tool calls from text output (regex-based, less reliable but functional)
- A new config field `orchestrator_model` (optional, defaults to `default_model`) allows overriding with a more capable model if needed

## Orchestrator LLM Cost Tracking

The orchestrator's own LLM calls (up to 5 turns) cost tokens separate from agent LLM calls. Both are tracked:

- **Agent costs:** Flow through `AgentResponse.cost_usd` as today. Accumulated in `state["total_cost_usd"]` by the `execute_tools` node after each `dispatch_agents` call.
- **Orchestrator costs:** After each `orchestrator_llm` node invocation, extract `response.usage_metadata` (input_tokens, output_tokens), calculate cost using the same `COST_PER_INPUT_TOKEN` / `COST_PER_OUTPUT_TOKEN` constants, add to `state["total_cost_usd"]`.
- **Budget check:** The `execute_tools` node checks `total_cost_usd >= max_budget_per_query_usd` before executing tools. If exceeded, it skips tool execution and forces route to `build_final_report`.

## Orchestrator System Prompt

```
You are the orchestrator agent in a multi-agent investment research system.
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
4. After gatherers return, dispatch analyzers (analyst, risk_assessor) — they
   benefit from fresh data the gatherers stored in memory
5. After all agents return, evaluate: do you have enough to answer the question?
6. Call build_report when you have sufficient data

RULES:
- You can dispatch multiple agents in a single call (they run in parallel)
- Prefer dispatching gatherers before analyzers — analyzers use gatherer data
- If an agent fails, decide whether to retry it or proceed without it
- Do NOT dispatch the same agent twice unless the first attempt failed
- You have a limited number of turns — be efficient, don't over-research
- When in doubt, build the report with what you have rather than looping
```

## Stopping Conditions

| Condition | Threshold | Behavior |
|-----------|-----------|----------|
| Max iterations | 5 orchestrator LLM turns | Force `build_report` with existing data |
| Budget | `max_budget_per_query_usd` from config ($0.50) | Stop dispatching agents, build report with what exists |
| Report built | `build_report` tool called | Exit loop, return report |
| No tool calls | LLM returns plain text | Exit loop, force report from existing data |

All enforced in Python (conditional edge + `execute_tools` node), not by the LLM.

## Integration Points

### Files rewritten
- **`orchestrator/graph.py`** — Static `StateGraph` replaced with ReAct loop. `ResearchOrchestrator` class stays, internals change completely. `CircuitBreaker` stays unchanged.
- **`api/dependencies.py`** — `create_orchestrator()` updated to create a separate orchestrator LLM (with `bind_tools()`), pass it to `ResearchOrchestrator`. Optional `orchestrator_model` config field read here.

### New files
- **`orchestrator/tools.py`** — The 3 tool definitions as standalone functions (called by bound methods on the orchestrator). Separate file for independent testability.

### Files with minor changes
- **`config.py`** — Add `orchestrator_model: str = ""` field. Empty string means "use `default_model`". Resolved at runtime in `dependencies.py`: `model = settings.orchestrator_model or settings.default_model`.

### Files unchanged
- `agents/base.py` and all 4 agent implementations — Same `run()` method
- `memory/manager.py` — Same `parallel_search()` and `store_research()` interface
- `orchestrator/conflict.py` — `detect_conflicts()` called from `build_report` tool
- `orchestrator/quality.py` — `assess_quality()` called from `build_report` tool
- `models/schemas.py` — All schemas stay the same
- `api/routes.py` — Still calls `orchestrator.run(query)`, same contract
- `errors.py` — Same error hierarchy

### Public API contract (unchanged)
```python
report: ResearchReport = await orchestrator.run(query: ResearchQuery)
```

## `create_react_agent` Reference

Documented as a comment block in `graph.py` — not used, but shows awareness of the built-in alternative:

```python
# ALTERNATIVE IMPLEMENTATION — LangGraph's built-in ReAct agent:
#
# from langgraph.prebuilt import create_react_agent
#
# orchestrator = create_react_agent(
#     model=llm.bind_tools([dispatch_agents, search_memory, build_report]),
#     tools=[dispatch_agents, search_memory, build_report],
#     prompt=ORCHESTRATOR_SYSTEM_PROMPT,
# )
# result = await orchestrator.ainvoke({"messages": [("user", query)]})
#
# We built the custom loop because:
# 1. Python-level guardrails (max iterations, budget tracking)
# 2. Custom state accumulation (agent_responses across turns)
# 3. Forced report fallback when iterations exhausted
# 4. Full visibility for debugging and observability
```

## Example Trace (Happy Path)

```
Turn 1: LLM → search_memory("NVIDIA investment")
        Result: "Found 2 results from 30 days ago, P/E was 58"

Turn 2: LLM → dispatch_agents(["researcher", "sentiment"])
        Result: "researcher (0.82): P/E=52, revenue $39B... | sentiment (0.75): bullish, 48 buy..."

Turn 3: LLM → dispatch_agents(["analyst", "risk_assessor"])
        Result: "analyst (0.78): fairly valued at P/E 52... | risk (0.80): China export ban risk..."

Turn 4: LLM → build_report(reasoning="4/4 agents responded, high confidence, one key tension on China")
        Result: "Report built successfully."
        → Loop exits, report returned
```

## Future Extensions (not in scope now)

- Query classifier (simple/moderate/complex routing)
- Sub-query specialization (agent-specific focused questions)
- Result evaluation + replanning loop
- Budget-aware orchestration (inject remaining budget into prompt)
- Email delivery tool (intent detection + side effect)

These slot in as additional tools or pre-loop nodes — no architecture change needed.
