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

PYTHON CONCEPT — Annotated reducers:
LangGraph uses Annotated[list[X], operator.add] to define how state
fields accumulate across nodes. When a node returns {"agent_responses": [new_item]},
LangGraph concatenates it with the existing list rather than replacing it.
TS equivalent: a Redux reducer that merges arrays.
"""

import asyncio
import logging
import operator
import time
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
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
    """Circuit breaker pattern — prevents calling agents that keep failing.

    Like a circuit breaker in your house:
    - Normal (CLOSED): electricity flows, agent gets called
    - Tripped (OPEN): too many failures, stop calling the agent
    - Testing (HALF_OPEN): timeout passed, try ONE call to test

    PYTHON CONCEPT — time.time():
    Returns seconds since epoch as a float. Used for tracking when
    failures happened and when to reset.
    TS equivalent: Date.now() / 1000
    Rust equivalent: std::time::Instant::now()
    """

    def __init__(self, max_failures: int = 3, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self._failures: dict[str, int] = {}
        self._last_failure: dict[str, float] = {}
        self._state: dict[str, str] = {}

    def can_call(self, agent_name: str) -> bool:
        """Check if an agent is safe to call."""
        state = self._state.get(agent_name, "closed")
        if state == "closed":
            return True
        if state == "open":
            last_fail = self._last_failure.get(agent_name, 0)
            if time.time() - last_fail > self.reset_timeout:
                self._state[agent_name] = "half_open"
                logger.info("[circuit_breaker] %s → HALF_OPEN (testing)", agent_name)
                return True
            return False
        return True  # half_open — allow test call

    def record_success(self, agent_name: str) -> None:
        """Agent call succeeded — reset the breaker."""
        self._failures[agent_name] = 0
        self._state[agent_name] = "closed"
        self._last_failure.pop(agent_name, None)

    def record_failure(self, agent_name: str) -> None:
        """Agent call failed — increment count, maybe trip."""
        self._failures[agent_name] = self._failures.get(agent_name, 0) + 1
        self._last_failure[agent_name] = time.time()
        if self._failures[agent_name] >= self.max_failures:
            self._state[agent_name] = "open"
            logger.warning(
                "[circuit_breaker] %s → OPEN (tripped after %d failures)",
                agent_name, self._failures[agent_name],
            )

    def get_state(self, agent_name: str) -> str:
        """Get the current circuit state for an agent."""
        return self._state.get(agent_name, "closed")


# =============================================================================
# STATE
# =============================================================================


class OrchestratorAgentState(TypedDict):
    """State for the orchestrator agent ReAct loop.

    LANGGRAPH CONCEPT — Annotated reducers:
    messages uses add_messages (auto-appends new messages).
    agent_responses and failed_agents use operator.add (list concat).
    remaining_iterations and total_cost_usd are plain fields (replaced atomically).
    """

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

    The flow:
    START → fetch_user_profile → orchestrator_llm ⟲ execute_tools
                                        ↓ (done)
                                  build_final_report → END
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

        # Create tool functions via factories (closure pattern)
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
        Actual execution goes through the factory-created functions with _tool_context.

        PYTHON CONCEPT — nested closures:
        Each @tool function captures `self` from the enclosing method.
        When LangGraph calls tool.ainvoke(), the wrapper runs,
        reads self._tool_context, and delegates to the factory function.
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
        """Build the ReAct loop StateGraph.

        LANGGRAPH CONCEPT — conditional edges:
        add_conditional_edges() takes a function that returns a string
        matching one of the edge targets. This is how the loop works:
        orchestrator_llm → should_continue() → "execute_tools" or "build_final_report"
        """
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
        """Decide whether to continue the loop or build the report.

        Routes to:
        - "build_final_report" if report already built, no tool calls, or iterations exhausted
        - "execute_tools" if LLM wants to call tools and iterations remain
        """
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
        """Fetch user profile from Neo4j once before the loop starts.

        WHY A SEPARATE NODE:
        If 4 agents each fetch the profile, that's 4x redundant Neo4j reads.
        By fetching once and storing in state, all agents share the same context.

        GRACEFUL DEGRADATION:
        If Neo4j is down or no user_id → returns empty string.
        Agents run normally, just without [USER PROFILE] section.
        """
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
        """Call the orchestrator LLM with the current message history.

        The LLM sees all previous messages (system prompt, user query,
        past tool calls and results) and decides what to do next:
        - Call a tool (dispatch_agents, search_memory, build_report)
        - Return plain text (loop exits)
        """
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
        """Execute tool calls from the LLM response.

        Runs each tool call sequentially (even if the LLM returned multiple
        in one turn) to avoid race conditions on shared _tool_context.
        Decrements remaining_iterations once per node invocation.
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"remaining_iterations": state["remaining_iterations"] - 1}

        # Budget enforcement — skip tool execution if budget exhausted
        max_budget = state["research_query"].max_budget_usd
        if state["total_cost_usd"] >= max_budget:
            logger.warning(
                "[orchestrator] Budget exhausted ($%.4f >= $%.2f), forcing report",
                state["total_cost_usd"], max_budget,
            )
            tool_id = last_message.tool_calls[0]["id"]
            return {
                "messages": [ToolMessage(
                    content="BUDGET EXHAUSTED. Call build_report immediately with whatever data you have.",
                    tool_call_id=tool_id,
                )],
                "remaining_iterations": 0,
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
            "all_responses": list(state["agent_responses"]),
            "all_failures": list(state["failed_agents"]),
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

        return {
            "messages": tool_messages,
            "agent_responses": new_responses,
            "failed_agents": new_failures,
            "remaining_iterations": state["remaining_iterations"] - 1,
            "total_cost_usd": state["total_cost_usd"] + agent_cost,
            "report": report,
        }

    async def _node_build_final_report(self, state: OrchestratorAgentState) -> dict:
        """Final node — ensure a report exists.

        If build_report tool was already called, the report is in state.
        Otherwise, force-build a report with whatever data exists.
        """
        if state["report"] is not None:
            logger.info("[orchestrator] Report already built by tool")
            return {}

        # Force-build a report with whatever data exists
        logger.info("[orchestrator] Forcing report build (loop exited without build_report)")

        self._tool_context = {
            "all_responses": list(state["agent_responses"]),
            "all_failures": list(state["failed_agents"]),
            "new_responses": [],
            "new_failures": [],
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

        This is what the API/UI calls. One method, one input, one output.
        Same interface as the old static pipeline — no changes needed
        in routes.py or dependencies.py (beyond constructor args).

        LANGGRAPH CONCEPT — ainvoke():
        Runs the compiled graph asynchronously. You pass the initial state,
        and it flows through all nodes, returning the final state.
        """
        session_id = str(uuid4())

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
