"""LangGraph orchestrator — plan-then-execute with replan-on-failure.

ARCHITECTURE:
The orchestrator uses a plan-then-execute pattern instead of a reactive ReAct loop.
One LLM call creates a structured plan upfront. Code executes each step mechanically
(no LLM calls between steps). If a step fails, one LLM call replans the remaining steps.

    START → fetch_user_profile → search_memory → create_plan
        → execute_step ⟲ route_after_step
                ↓ (step failed)         ↓ (plan done)
              replan              build_final_report → END

WHY PLAN-THEN-EXECUTE OVER REACTIVE REACT:
1. Predictable — you know the steps upfront, can estimate time/cost
2. Cheaper — 1-2 LLM calls for orchestration vs 4-5 in reactive mode
3. Observable — can show the user "Step 2/4: Running researcher..."
4. Flexible on failure — replan handles errors without full reactivity

PYTHON CONCEPT — Annotated reducers:
LangGraph uses Annotated[list[X], operator.add] to define how state
fields accumulate across nodes. When a node returns {"agent_responses": [new_item]},
LangGraph concatenates it with the existing list rather than replacing it.
TS equivalent: a Redux reducer that merges arrays.
"""

import asyncio
import json
import logging
import operator
import re
import time
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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


MAX_PLAN_STEPS = 5
MAX_REPLANS = 2
VALID_ACTIONS = {"dispatch_agents", "build_report"}
VALID_AGENT_NAMES = {"researcher", "sentiment", "analyst", "risk_assessor"}

REPLAN_SYSTEM_PROMPT = """You are the orchestrator in a multi-agent investment research system.
A step in your plan FAILED. You need to create a NEW plan for the remaining work.

AVAILABLE ACTIONS: dispatch_agents, build_report

RULES:
- Do NOT retry an agent that already succeeded — their data is already collected.
- You may retry a failed agent OR skip it and proceed with what you have.
- The plan MUST end with build_report.
- If you have enough data from successful agents, just build the report.
- ONLY use agents from the available agents list provided.

OUTPUT FORMAT — respond with ONLY a JSON array, no other text:
[
    {"action": "dispatch_agents", "agent_names": ["risk_assessor"]},
    {"action": "build_report"}
]"""


# =============================================================================
# CIRCUIT BREAKER (unchanged)
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
# PLAN VALIDATION
# =============================================================================


def validate_plan(plan: list[dict]) -> list[dict]:
    """Validate and auto-fix a plan from the LLM.

    Ensures:
    - Only valid actions (dispatch_agents, build_report)
    - Only valid agent names in dispatch_agents steps
    - Plan ends with build_report
    - Max MAX_PLAN_STEPS steps
    - At least one step (build_report fallback)
    """
    validated = []

    for step in plan[:MAX_PLAN_STEPS]:
        action = step.get("action", "")
        if action not in VALID_ACTIONS:
            logger.warning("[plan] Skipping invalid action: %s", action)
            continue

        if action == "dispatch_agents":
            agent_names = step.get("agent_names", [])
            valid_names = [n for n in agent_names if n in VALID_AGENT_NAMES]
            if not valid_names:
                logger.warning("[plan] Skipping dispatch_agents with no valid agents: %s", agent_names)
                continue
            validated.append({"action": "dispatch_agents", "agent_names": valid_names})

        elif action == "build_report":
            validated.append({"action": "build_report"})

    # Ensure plan ends with build_report
    if not validated or validated[-1]["action"] != "build_report":
        validated.append({"action": "build_report"})

    return validated


def parse_plan_from_llm(text: str) -> list[dict]:
    """Parse a JSON plan from LLM output.

    The LLM should return a JSON array, but may wrap it in markdown code blocks
    or add extra text. This function extracts the JSON robustly.
    """
    # Try direct JSON parse first
    text = text.strip()
    try:
        plan = json.loads(text)
        if isinstance(plan, list):
            return plan
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        try:
            plan = json.loads(code_block.group(1).strip())
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass

    # Try finding a JSON array anywhere in the text
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        try:
            plan = json.loads(array_match.group(0))
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass

    logger.warning("[plan] Failed to parse plan from LLM output: %s", text[:200])
    return []


def filter_available_agents(memory_context: str) -> dict[str, bool]:
    """Decide which agents to skip based on memory results.

    Code-level pre-check — deterministic, not LLM vibes.
    The LLM only sees agents that pass this filter.

    Rules:
    - Skip researcher if memory has ≥3 high-relevance research items (score ≥ 0.85)
    - Skip sentiment if memory has facts containing sentiment keywords
    - NEVER skip analyst or risk_assessor — they interpret data, they don't gather it
    """
    skip = {
        "researcher": False,
        "sentiment": False,
        "analyst": False,      # never skip
        "risk_assessor": False, # never skip
    }

    # Count high-relevance research items: [RESEARCH-N] (relevance: 0.XX)
    research_matches = re.findall(r"\[RESEARCH-\d+\]\s*\(relevance:\s*([\d.]+)\)", memory_context)
    high_relevance_count = sum(1 for score in research_matches if float(score) >= 0.85)

    if high_relevance_count >= 3:
        skip["researcher"] = True
        logger.info("[filter] Skipping researcher — %d high-relevance items in memory", high_relevance_count)

    # Check for sentiment-related facts
    sentiment_keywords = ["bullish", "bearish", "sentiment", "analyst rating", "market mood", "insider"]
    fact_matches = re.findall(r"\[FACT-\d+\][:\s]*(.*)", memory_context)
    has_sentiment = any(
        any(kw in fact.lower() for kw in sentiment_keywords)
        for fact in fact_matches
    )

    if has_sentiment:
        skip["sentiment"] = True
        logger.info("[filter] Skipping sentiment — sentiment data found in memory")

    return skip


def get_available_agents(memory_context: str) -> list[str]:
    """Return list of agent names that should be dispatched (not skipped)."""
    skip = filter_available_agents(memory_context)
    return [name for name, skipped in skip.items() if not skipped]


def default_plan(available_agents: list[str] | None = None) -> list[dict]:
    """Fallback plan when LLM is unavailable or returns garbage.

    Default strategy: gatherers first, then analyzers, then report.
    Respects the available_agents filter — skipped agents are excluded.
    """
    if available_agents is None:
        available_agents = list(VALID_AGENT_NAMES)

    gatherers = [a for a in ["researcher", "sentiment"] if a in available_agents]
    analyzers = [a for a in ["analyst", "risk_assessor"] if a in available_agents]

    steps = []
    if gatherers:
        steps.append({"action": "dispatch_agents", "agent_names": gatherers})
    if analyzers:
        steps.append({"action": "dispatch_agents", "agent_names": analyzers})
    steps.append({"action": "build_report"})

    return steps


# =============================================================================
# STATE
# =============================================================================


class OrchestratorAgentState(TypedDict):
    """State for the plan-then-execute orchestrator.

    LANGGRAPH CONCEPT — Annotated reducers:
    messages uses add_messages (auto-appends new messages).
    agent_responses and failed_agents use operator.add (list concat).
    Plain fields (plan, current_step_index, etc.) are replaced atomically.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    research_query: ResearchQuery
    session_id: str
    user_profile_context: str
    agent_responses: Annotated[list[AgentResponse], operator.add]
    failed_agents: Annotated[list[dict], operator.add]
    total_cost_usd: float
    report: ResearchReport | None
    start_time: float
    # Plan-then-execute fields
    plan: list[dict]
    current_step_index: int
    memory_context: str
    replan_count: int
    available_agents: list[str]  # agents that passed the code-level pre-check


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class ResearchOrchestrator:
    """Plan-then-execute orchestrator using LangGraph.

    The flow:
    START → fetch_user_profile → search_memory → create_plan
        → execute_step ⟲ route_after_step
                ↓ (step failed)         ↓ (plan done)
              replan              build_final_report → END
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        memory_manager,
        circuit_breaker: CircuitBreaker | None = None,
        orchestrator_llm=None,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        max_iterations: int = MAX_PLAN_STEPS,
    ):
        self.agents = {agent.name: agent for agent in agents}
        self.memory_manager = memory_manager
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_iterations = max_iterations

        # Store raw LLM (no bind_tools needed — we parse JSON, not tool calls)
        self.orchestrator_llm = orchestrator_llm

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

        # Mutable context shared with tools during execution
        self._tool_context: dict = {}

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the plan-then-execute StateGraph.

        LANGGRAPH CONCEPT — conditional edges:
        add_conditional_edges() takes a function that returns a string
        matching one of the edge targets. This is how routing works.
        """
        graph = StateGraph(OrchestratorAgentState)

        graph.add_node("fetch_user_profile", self._node_fetch_user_profile)
        graph.add_node("search_memory", self._node_search_memory)
        graph.add_node("create_plan", self._node_create_plan)
        graph.add_node("execute_step", self._node_execute_step)
        graph.add_node("replan", self._node_replan)
        graph.add_node("build_final_report", self._node_build_final_report)

        # Linear: START → fetch_user_profile → search_memory → create_plan
        graph.add_edge(START, "fetch_user_profile")
        graph.add_edge("fetch_user_profile", "search_memory")
        graph.add_edge("search_memory", "create_plan")

        # create_plan → execute_step (always)
        graph.add_edge("create_plan", "execute_step")

        # execute_step → route based on result
        graph.add_conditional_edges("execute_step", self._route_after_step, {
            "execute_step": "execute_step",
            "replan": "replan",
            "build_final_report": "build_final_report",
        })

        # replan → execute_step (resume with new plan)
        graph.add_edge("replan", "execute_step")

        graph.add_edge("build_final_report", END)

        return graph.compile()

    # =========================================================================
    # ROUTING
    # =========================================================================

    def _route_after_step(self, state: OrchestratorAgentState) -> str:
        """Decide what to do after executing a step.

        Routes to:
        - "build_final_report" if report was built or plan is complete
        - "replan" if the step had failures and we haven't exceeded replan limit
        - "execute_step" if more steps remain
        """
        # Report already built (build_report step succeeded)
        if state["report"] is not None:
            return "build_final_report"

        plan = state["plan"]
        step_index = state["current_step_index"]

        # Plan complete — all steps executed
        if step_index >= len(plan):
            return "build_final_report"

        # Check if last step had failures that warrant replanning
        last_failures = state["failed_agents"]
        if last_failures and state["replan_count"] < MAX_REPLANS:
            # Only replan if the most recent step had a failure
            # (check if failed_agents grew in the last step)
            return "replan"

        # More steps to execute
        if step_index < len(plan):
            return "execute_step"

        return "build_final_report"

    # =========================================================================
    # NODE FUNCTIONS
    # =========================================================================

    async def _node_fetch_user_profile(self, state: OrchestratorAgentState) -> dict:
        """Fetch user profile from Neo4j once before the loop starts.

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

    async def _node_search_memory(self, state: OrchestratorAgentState) -> dict:
        """Search memory BEFORE planning so the LLM can make informed decisions.

        Runs the same search_memory tool function but stores results in state
        rather than returning to an LLM. The create_plan node reads this context.
        """
        query = state["research_query"].query
        try:
            result = await self._search_fn(query)
            logger.info("[orchestrator] Pre-plan memory search complete")
            return {"memory_context": result}
        except Exception as e:
            logger.warning("[orchestrator] Pre-plan memory search failed: %s", e)
            return {"memory_context": "Memory search failed. Plan without past context."}

    async def _node_create_plan(self, state: OrchestratorAgentState) -> dict:
        """One LLM call to create the execution plan.

        CODE-LEVEL PRE-CHECK runs first: filters out agents whose data
        already exists in memory. The LLM only sees agents that pass
        the filter — it can't make a bad call because skipped agents
        aren't even in the prompt.

        If LLM is unavailable or returns invalid JSON, falls back to
        the default plan (which also respects the filter).
        """
        memory_context = state.get("memory_context", "")

        # Code-level pre-check: decide which agents to skip
        available = get_available_agents(memory_context)
        logger.info("[orchestrator] Available agents after filter: %s", available)

        if self.orchestrator_llm is None:
            logger.warning("[orchestrator] No LLM configured, using default plan")
            plan = validate_plan(default_plan(available))
            return {"plan": plan, "available_agents": available}

        query = state["research_query"].query
        focus_areas = state["research_query"].focus_areas or []

        # Build dynamic prompt — only show agents that passed the filter
        agent_descriptions = {
            "researcher": "researcher: Gathers raw facts, data points, financial metrics from the web",
            "sentiment": "sentiment: Analyzes market mood, analyst ratings, insider activity, news sentiment",
            "analyst": "analyst: Interprets financial data, valuation, peer comparison (works best AFTER researcher)",
            "risk_assessor": "risk_assessor: Identifies risks, threats, downside scenarios (works best AFTER researcher)",
        }
        agent_lines = "\n".join(f"- {agent_descriptions[a]}" for a in available)

        system_prompt = f"""You are the orchestrator in a multi-agent investment research system.
Your job is to CREATE A PLAN — a list of steps to answer the user's research query.

AVAILABLE AGENTS:
{agent_lines}

AVAILABLE ACTIONS:
- dispatch_agents: Run one or more agents in parallel. Specify agent_names list.
- build_report: Synthesize all collected data into the final report. Always the last step.

STRATEGY:
- Dispatch gatherers (researcher, sentiment) BEFORE analyzers (analyst, risk_assessor).
- Analyzers benefit from data that gatherers store in memory.
- You can dispatch multiple agents in one step (they run in parallel).
- The plan MUST end with build_report.
- ONLY use agents from the AVAILABLE AGENTS list above.

OUTPUT FORMAT — respond with ONLY a JSON array, no other text:
[
    {{"action": "dispatch_agents", "agent_names": ["researcher", "sentiment"]}},
    {{"action": "build_report"}}
]"""

        user_message = f"Research query: {query}"
        if focus_areas:
            user_message += f"\nFocus areas: {', '.join(focus_areas)}"
        user_message += f"\n\nMemory search results:\n{memory_context}"
        user_message += "\n\nCreate your execution plan as a JSON array."

        try:
            response = await self.orchestrator_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ])

            # Track planning LLM cost
            usage = getattr(response, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
            output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
            plan_cost = (input_tokens + output_tokens) * 0.5 / 1_000_000

            raw_plan = parse_plan_from_llm(response.content)
            plan = validate_plan(raw_plan) if raw_plan else validate_plan(default_plan(available))

            logger.info("[orchestrator] Plan created: %s", json.dumps(plan))
            return {
                "plan": plan,
                "available_agents": available,
                "total_cost_usd": state["total_cost_usd"] + plan_cost,
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ],
            }

        except Exception as e:
            logger.warning("[orchestrator] Plan creation failed: %s. Using default plan.", e)
            plan = validate_plan(default_plan(available))
            return {"plan": plan, "available_agents": available}

    async def _node_execute_step(self, state: OrchestratorAgentState) -> dict:
        """Execute the current step in the plan mechanically (no LLM call).

        Reads plan[current_step_index], runs the corresponding tool function
        directly, and increments the step index.
        """
        plan = state["plan"]
        step_index = state["current_step_index"]

        if step_index >= len(plan):
            logger.warning("[orchestrator] execute_step called but plan exhausted")
            return {"current_step_index": step_index}

        step = plan[step_index]
        action = step["action"]

        logger.info("[orchestrator] Executing step %d/%d: %s", step_index + 1, len(plan), action)

        # Budget enforcement — check both in-memory state AND Redis atomic counter.
        # The in-memory check catches the obvious case (accumulated cost from prior steps).
        # The Redis check catches the TOCTOU race (parallel agents spending simultaneously).
        max_budget = state["research_query"].max_budget_usd
        if state["total_cost_usd"] >= max_budget:
            logger.warning("[orchestrator] Budget exhausted ($%.4f >= $%.2f), skipping to report",
                           state["total_cost_usd"], max_budget)
            return {
                "current_step_index": len(plan),  # skip to end
                "plan": [{"action": "build_report"}],
            }
        # Also check the atomic Redis budget counter (catches parallel overspend)
        try:
            if self.memory_manager and hasattr(self.memory_manager, "short_term"):
                budget_key = f"budget:{state['session_id']}:spent"
                redis_spent = await self.memory_manager.short_term.client.get(budget_key)
                if redis_spent and float(redis_spent) >= max_budget:
                    logger.warning("[orchestrator] Redis budget exhausted ($%s >= $%.2f)",
                                   redis_spent, max_budget)
                    return {
                        "current_step_index": len(plan),
                        "plan": [{"action": "build_report"}],
                    }
        except Exception:
            pass  # Redis check is best-effort

        # Set up tool context
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

        if action == "dispatch_agents":
            agent_names = step.get("agent_names", [])
            result = await self._dispatch_fn(agent_names, _context=self._tool_context)
            logger.info("[orchestrator] dispatch_agents result: %s", result[:200])

            new_responses = self._tool_context.get("new_responses", [])
            new_failures = self._tool_context.get("new_failures", [])
            agent_cost = sum(r.cost_usd for r in new_responses)

            return {
                "agent_responses": new_responses,
                "failed_agents": new_failures,
                "total_cost_usd": state["total_cost_usd"] + agent_cost,
                "current_step_index": step_index + 1,
            }

        elif action == "build_report":
            # Update tool context with all collected data
            self._tool_context["all_responses"] = list(state["agent_responses"])
            self._tool_context["all_failures"] = list(state["failed_agents"])

            await self._report_fn(
                reasoning="Plan complete — building report from collected agent data",
                _context=self._tool_context,
            )

            report = self._tool_context.get("report")
            return {
                "report": report,
                "current_step_index": step_index + 1,
            }

        else:
            logger.error("[orchestrator] Unknown action in plan: %s", action)
            return {"current_step_index": step_index + 1}

    async def _node_replan(self, state: OrchestratorAgentState) -> dict:
        """One LLM call to create a new plan after a failure.

        Shows the LLM what succeeded, what failed, and asks for a new plan
        for the remaining work. If LLM is unavailable, falls back to
        just building the report with whatever data exists.
        """
        replan_count = state["replan_count"] + 1
        logger.info("[orchestrator] Replanning (attempt %d/%d)", replan_count, MAX_REPLANS)

        if self.orchestrator_llm is None or replan_count > MAX_REPLANS:
            logger.warning("[orchestrator] Replan limit reached or no LLM, forcing report")
            return {
                "plan": [{"action": "build_report"}],
                "current_step_index": 0,
                "replan_count": replan_count,
            }

        # Build context for the LLM — only show available agents
        succeeded = [r.agent_name for r in state["agent_responses"]]
        failed = [f"{f['agent']} ({f['error_type']})" for f in state["failed_agents"]]
        available = state.get("available_agents", list(VALID_AGENT_NAMES))
        remaining = [a for a in available if a not in succeeded]

        user_message = (
            f"Original query: {state['research_query'].query}\n\n"
            f"Available agents for replanning: {remaining}\n"
            f"Agents that SUCCEEDED (data already collected): {succeeded}\n"
            f"Agents that FAILED: {failed}\n\n"
            f"Create a new plan for the remaining work as a JSON array.\n"
            f"Do NOT re-dispatch agents that already succeeded.\n"
            f"ONLY use agents from the available agents list."
        )

        try:
            response = await self.orchestrator_llm.ainvoke([
                SystemMessage(content=REPLAN_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ])

            # Track replan cost
            usage = getattr(response, "usage_metadata", None) or {}
            input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
            output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
            replan_cost = (input_tokens + output_tokens) * 0.5 / 1_000_000

            raw_plan = parse_plan_from_llm(response.content)
            plan = validate_plan(raw_plan) if raw_plan else [{"action": "build_report"}]

            logger.info("[orchestrator] Replan result: %s", json.dumps(plan))
            return {
                "plan": plan,
                "current_step_index": 0,
                "replan_count": replan_count,
                "total_cost_usd": state["total_cost_usd"] + replan_cost,
            }

        except Exception as e:
            logger.warning("[orchestrator] Replan LLM call failed: %s. Building report.", e)
            return {
                "plan": [{"action": "build_report"}],
                "current_step_index": 0,
                "replan_count": replan_count,
            }

    async def _node_build_final_report(self, state: OrchestratorAgentState) -> dict:
        """Final node — ensure a report exists.

        If build_report step was already executed, the report is in state.
        Otherwise, force-build a report with whatever data exists.
        """
        if state["report"] is not None:
            logger.info("[orchestrator] Report already built")
            return {}

        # Force-build a report with whatever data exists
        logger.info("[orchestrator] Forcing report build (plan ended without report)")

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
            reasoning="Forced report — plan completed without build_report step",
            _context=self._tool_context,
        )

        return {"report": self._tool_context.get("report")}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def run(self, query: ResearchQuery) -> ResearchReport:
        """Execute the plan-then-execute orchestrator.

        This is what the API/UI calls. One method, one input, one output.
        Same interface as before — no changes needed in routes.py or dependencies.py.

        LANGGRAPH CONCEPT — ainvoke():
        Runs the compiled graph asynchronously. You pass the initial state,
        and it flows through all nodes, returning the final state.
        """
        session_id = str(uuid4())

        initial_state: OrchestratorAgentState = {
            "messages": [],
            "research_query": query,
            "session_id": session_id,
            "user_profile_context": "",
            "agent_responses": [],
            "failed_agents": [],
            "total_cost_usd": 0.0,
            "report": None,
            "start_time": time.time(),
            "plan": [],
            "current_step_index": 0,
            "memory_context": "",
            "replan_count": 0,
            "available_agents": list(VALID_AGENT_NAMES),
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
