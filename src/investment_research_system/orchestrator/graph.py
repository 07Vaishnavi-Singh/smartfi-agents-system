"""LangGraph orchestrator — the brain that runs all agents.

This uses a HYBRID approach:
  Phase 1 (parallel): Gatherers  — researcher + sentiment (fetch fresh data)
  Phase 2 (parallel): Analyzers  — analyst + risk (analyze the fresh data)

Why hybrid?
- Phase 1 agents GATHER information (web search, news)
- They store results in memory via MemoryManager.store_research()
- Phase 2 agents then pull that fresh data from memory
- Analyst gets the researcher's fresh findings, not just stale memory

Both phases are parallel WITHIN themselves. Only the boundary
between phases is sequential.

    ┌──────────────────────────────────┐
    │  Phase 1: GATHER (parallel)      │
    │  ┌────────────┐ ┌────────────┐  │
    │  │ Researcher │ │ Sentiment  │  │
    │  └────────────┘ └────────────┘  │
    └──────────────┬───────────────────┘
                   │ results stored in memory
                   ▼
    ┌──────────────────────────────────┐
    │  Phase 2: ANALYZE (parallel)     │
    │  ┌────────────┐ ┌────────────┐  │
    │  │  Analyst   │ │    Risk    │  │
    │  └────────────┘ └────────────┘  │
    └──────────────┬───────────────────┘
                   ▼
         Conflicts → Quality → Report

PYTHON CONCEPT — TypedDict:
LangGraph uses TypedDict to define the "state" — a dict with fixed keys
and known types. Each node receives the state, adds to it, returns it.

TS equivalent: interface State { query: string; responses: AgentResponse[]; ... }
Rust equivalent: struct State { query: String, responses: Vec<AgentResponse>, ... }

LANGGRAPH CONCEPT — StateGraph:
A StateGraph is a directed graph where:
- Nodes = functions that transform state
- Edges = connections between nodes (which runs after which)
- State = the data that flows through the graph
You define the graph, compile it, then invoke it with input.
"""

import asyncio
import logging
import time
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from investment_research_system.agents.base import LLM_CALL_TIMEOUT_SECONDS, BaseAgent
from investment_research_system.errors import (
    AgentError,
    BudgetExceededError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    MemoryUnavailableError,
)
from investment_research_system.models.schemas import AgentResponse, ResearchQuery, ResearchReport
from investment_research_system.observability.tracing import get_run_config
from investment_research_system.orchestrator.conflict import detect_conflicts, format_conflicts_summary
from investment_research_system.orchestrator.quality import QualityAssessment, assess_quality, format_quality_summary

logger = logging.getLogger(__name__)


# Agents are split into two phases based on their role
# Gatherers fetch new data, analyzers interpret it
GATHERER_AGENTS = {"researcher", "sentiment"}
ANALYZER_AGENTS = {"analyst", "risk_assessor"}


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """Circuit breaker pattern — prevents calling agents that keep failing.

    Like a circuit breaker in your house:
    - Normal (CLOSED): electricity flows, agent gets called
    - Tripped (OPEN): too many failures, stop calling the agent
    - Testing (HALF_OPEN): timeout passed, try ONE call to test

    PYTHON CONCEPT — time.time():
    Returns seconds since epoch (Jan 1, 1970) as a float.
    Used for tracking when failures happened and when to reset.
    TS equivalent: Date.now() / 1000
    Rust equivalent: std::time::Instant::now()
    """

    def __init__(self, max_failures: int = 3, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        # Track state per agent
        self._failures: dict[str, int] = {}          # agent_name → failure count
        self._last_failure: dict[str, float] = {}    # agent_name → timestamp
        self._state: dict[str, str] = {}             # agent_name → "closed"/"open"/"half_open"

    def can_call(self, agent_name: str) -> bool:
        """Check if an agent is safe to call.

        Returns True if the circuit is closed (healthy) or half-open (testing).
        Returns False if the circuit is open (agent is failing too much).
        """
        state = self._state.get(agent_name, "closed")

        if state == "closed":
            return True

        if state == "open":
            # Check if enough time has passed to try again
            last_fail = self._last_failure.get(agent_name, 0)
            if time.time() - last_fail > self.reset_timeout:
                self._state[agent_name] = "half_open"
                logger.info("[circuit_breaker] %s → HALF_OPEN (testing)", agent_name)
                return True  # allow one test call
            return False  # still too soon

        # half_open — allow the test call
        return True

    def record_success(self, agent_name: str) -> None:
        """Agent call succeeded — reset the breaker."""
        self._failures[agent_name] = 0
        self._state[agent_name] = "closed"
        if agent_name in self._last_failure:
            del self._last_failure[agent_name]

    def record_failure(self, agent_name: str) -> None:
        """Agent call failed — increment failure count, maybe trip the breaker."""
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
# LANGGRAPH STATE
# =============================================================================

class OrchestratorState(TypedDict):
    """The shared state that flows between nodes in the graph.

    LANGGRAPH CONCEPT — State:
    Every node receives this dict, can read any field, and returns
    updates to it. LangGraph merges the updates into the state
    automatically.

    Think of it as a shared document that each workstation fills in.
    """

    query: str                                    # user's original question
    research_query: ResearchQuery                 # parsed query with settings
    session_id: str                               # unique session ID
    agent_responses: list[AgentResponse]          # collected responses
    failed_agents: list[dict]                      # agents that couldn't respond (with error details)
    conflicts: list[dict]                         # disagreements between agents
    quality: QualityAssessment | None             # quality grade
    report: ResearchReport | None                 # final output
    start_time: float                             # for tracking total latency


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class ResearchOrchestrator:
    """LangGraph-powered research orchestrator using hybrid parallel strategy.

    HYBRID APPROACH:
    Phase 1 (parallel): Gatherers (researcher + sentiment) fetch fresh data
    Phase 2 (parallel): Analyzers (analyst + risk) interpret that data

    Both phases run their agents in parallel. The phases themselves
    are sequential so analyzers can use the gatherers' fresh results.

    The flow:
    START → run_gatherers → run_analyzers → detect_conflicts
          → quality_check → build_report → END
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        circuit_breaker: CircuitBreaker | None = None,
        max_retries: int = 1,
        retry_delay: float = 1.0,
    ):
        self.agents = {agent.name: agent for agent in agents}
        # PYTHON CONCEPT — dict comprehension:
        # Creates {"researcher": ResearcherAgent, "analyst": AnalystAgent, ...}
        # from a list of agents. Lookup by name is O(1).
        # TS equivalent: Object.fromEntries(agents.map(a => [a.name, a]))

        # Split agents into gatherers and analyzers
        self.gatherers = {
            name: agent for name, agent in self.agents.items()
            if name in GATHERER_AGENTS
        }
        self.analyzers = {
            name: agent for name, agent in self.agents.items()
            if name in ANALYZER_AGENTS
        }

        # Any agent not in either group goes into analyzers (safe default)
        for name, agent in self.agents.items():
            if name not in GATHERER_AGENTS and name not in ANALYZER_AGENTS:
                self.analyzers[name] = agent

        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine with hybrid phases.

        The flow:
        START → run_gatherers → run_analyzers → detect_conflicts
              → quality_check → build_report → END

        Phase 1 and 2 are separate nodes. Within each node,
        agents run in parallel via asyncio.gather().
        """
        graph = StateGraph(OrchestratorState)

        # Add nodes
        graph.add_node("run_gatherers", self._node_run_gatherers)
        graph.add_node("run_analyzers", self._node_run_analyzers)
        graph.add_node("detect_conflicts", self._node_detect_conflicts)
        graph.add_node("quality_check", self._node_quality_check)
        graph.add_node("build_report", self._node_build_report)

        # Add edges — the hybrid flow
        graph.add_edge(START, "run_gatherers")
        graph.add_edge("run_gatherers", "run_analyzers")
        # ^ This is the key sequential boundary.
        #   Gatherers MUST finish before analyzers start.
        #   Gatherers store results in memory → analyzers read from memory.
        graph.add_edge("run_analyzers", "detect_conflicts")
        graph.add_edge("detect_conflicts", "quality_check")
        graph.add_edge("quality_check", "build_report")
        graph.add_edge("build_report", END)

        return graph.compile()

    # =========================================================================
    # SHARED HELPER
    # =========================================================================

    async def _run_agent_with_retry(
        self, agent: BaseAgent, query: str, session_id: str,
    ) -> tuple[AgentResponse | None, dict | None]:
        """Run a single agent with retry + circuit breaker + typed error handling.

        Uses the error hierarchy to make intelligent recovery decisions:
        - Timeout/rate limit → retry with backoff
        - Refusal/budget exceeded → don't retry (won't help)
        - Memory unavailable → retry (transient infra issue)
        - Unknown error → retry (might be transient)

        Returns a tuple of (response, failure_info).
        On success: (AgentResponse, None)
        On failure: (None, {"agent": str, "error_type": str, "error_message": str})
        """
        if not self.circuit_breaker.can_call(agent.name):
            logger.warning("[orchestrator] Skipping %s — circuit breaker OPEN", agent.name)
            return None, {
                "agent": agent.name,
                "error_type": "circuit_breaker",
                "error_message": "Skipped — too many recent failures (circuit breaker open)",
            }

        # Track the last error so we can report it if all retries fail
        last_error_type = "unknown"
        last_error_message = "Unknown error"

        for attempt in range(self.max_retries + 1):
            try:
                response = await agent.run(query, session_id)
                self.circuit_breaker.record_success(agent.name)
                return response, None

            except BudgetExceededError:
                # Hard stop — retrying or switching providers won't help
                logger.info("[orchestrator] %s hit budget limit, stopping", agent.name)
                return None, {
                    "agent": agent.name,
                    "error_type": "budget_exceeded",
                    "error_message": "Stopped — query exceeded its allocated cost budget",
                }

            except LLMRefusalError as e:
                # Model won't answer this — retrying is pointless
                logger.warning("[orchestrator] %s refused query: %s", agent.name, e)
                return None, {
                    "agent": agent.name,
                    "error_type": "refused",
                    "error_message": "Model refused to answer this query",
                }

            except LLMRateLimitError as e:
                # Backoff harder on rate limits (exponential)
                last_error_type = "rate_limit"
                last_error_message = "Rate limited by LLM provider"
                logger.warning(
                    "[orchestrator] %s rate limited (attempt %d/%d): %s",
                    agent.name, attempt + 1, self.max_retries + 1, e,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(backoff)

            except LLMTimeoutError as e:
                # Transient — retry with linear backoff
                last_error_type = "timeout"
                last_error_message = f"Timed out after {LLM_CALL_TIMEOUT_SECONDS}s (retried {self.max_retries + 1} times)"
                logger.warning(
                    "[orchestrator] %s timed out (attempt %d/%d): %s",
                    agent.name, attempt + 1, self.max_retries + 1, e,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except MemoryUnavailableError as e:
                # Infra issue — retry, might recover
                last_error_type = "memory_unavailable"
                last_error_message = "Memory backend (Redis/Qdrant) unreachable"
                logger.warning(
                    "[orchestrator] %s memory unavailable (attempt %d/%d): %s",
                    agent.name, attempt + 1, self.max_retries + 1, e,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except AgentError as e:
                # Catch-all for other typed agent errors
                last_error_type = "agent_error"
                last_error_message = str(e)
                logger.warning(
                    "[orchestrator] %s failed (attempt %d/%d): %s",
                    agent.name, attempt + 1, self.max_retries + 1, e,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                # Truly unexpected — log at error level, still retry
                last_error_type = "unexpected"
                last_error_message = f"Unexpected error: {type(e).__name__}"
                logger.error(
                    "[orchestrator] %s unexpected error (attempt %d/%d): %s",
                    agent.name, attempt + 1, self.max_retries + 1, e,
                    exc_info=True,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        # All retries exhausted
        self.circuit_breaker.record_failure(agent.name)
        return None, {
            "agent": agent.name,
            "error_type": last_error_type,
            "error_message": last_error_message,
        }

    async def _run_agents_parallel(
        self, agents: dict[str, BaseAgent], query: str, session_id: str,
    ) -> tuple[list[AgentResponse], list[dict]]:
        """Run a group of agents in parallel, return responses + failure details.

        PYTHON CONCEPT — tuple return:
        Returns two values as a tuple: (responses, failed_agents).
        failed_agents is a list of dicts with keys: agent, error_type, error_message.
        TS equivalent: return [responses, failedAgents] with destructuring.
        Rust equivalent: (Vec<AgentResponse>, Vec<FailureInfo>)
        """
        results = await asyncio.gather(
            *[self._run_agent_with_retry(agent, query, session_id) for agent in agents.values()],
        )

        responses = []
        failed = []

        for response, failure_info in results:
            if response is not None:
                responses.append(response)
            elif failure_info is not None:
                failed.append(failure_info)

        return responses, failed

    # =========================================================================
    # NODE FUNCTIONS
    # =========================================================================

    async def _node_run_gatherers(self, state: OrchestratorState) -> dict:
        """Node 1: Run GATHERER agents in parallel (researcher + sentiment).

        These agents fetch fresh data from the web and news.
        Their results are stored in memory (via base agent's run() method),
        so Phase 2 analyzers can access them.

        HYBRID APPROACH — why gatherers first:
        The researcher searches Tavily and stores findings in Qdrant/Mem0.
        When the analyst runs in Phase 2, its memory.parallel_search()
        will find the researcher's fresh data alongside historical data.
        This gives the analyst CURRENT information to analyze.
        """
        query = state["query"]
        session_id = state["session_id"]

        logger.info(
            "[orchestrator] Phase 1: Running %d gatherers in parallel",
            len(self.gatherers),
        )

        responses, failed = await self._run_agents_parallel(
            self.gatherers, query, session_id,
        )

        logger.info(
            "[orchestrator] Phase 1 complete: %d succeeded, %d failed",
            len(responses), len(failed),
        )

        return {
            "agent_responses": responses,
            "failed_agents": failed,
        }

    async def _node_run_analyzers(self, state: OrchestratorState) -> dict:
        """Node 2: Run ANALYZER agents in parallel (analyst + risk).

        These agents analyze data — including fresh data that
        the gatherers just stored in memory.

        Even if all gatherers failed, analyzers still run.
        They'll use whatever is in memory (possibly stale data).
        The quality checker will note the degraded quality.
        """
        query = state["query"]
        session_id = state["session_id"]

        logger.info(
            "[orchestrator] Phase 2: Running %d analyzers in parallel",
            len(self.analyzers),
        )

        responses, failed = await self._run_agents_parallel(
            self.analyzers, query, session_id,
        )

        logger.info(
            "[orchestrator] Phase 2 complete: %d succeeded, %d failed",
            len(responses), len(failed),
        )

        # Merge with Phase 1 results (not replace)
        # PYTHON CONCEPT — list concatenation with +:
        # [1, 2] + [3, 4] = [1, 2, 3, 4]
        # TS equivalent: [...phase1, ...phase2]
        return {
            "agent_responses": state["agent_responses"] + responses,
            "failed_agents": state["failed_agents"] + failed,
        }

    async def _node_detect_conflicts(self, state: OrchestratorState) -> dict:
        """Node 3: Find disagreements between agent responses."""
        conflicts = detect_conflicts(state["agent_responses"])
        return {"conflicts": conflicts}

    async def _node_quality_check(self, state: OrchestratorState) -> dict:
        """Node 4: Assess the quality of the collected responses."""
        quality = assess_quality(
            responses=state["agent_responses"],
            conflicts=state["conflicts"],
            failed_agents=state["failed_agents"],
        )
        return {"quality": quality}

    async def _node_build_report(self, state: OrchestratorState) -> dict:
        """Node 5: Synthesize all agent outputs into a coherent summary via LLM.

        Instead of dumb truncation, we pass all agent outputs to the LLM
        and ask it to produce a unified investment research summary with
        citations. This is one extra LLM call — much cheaper than a full
        agent pipeline (no memory search, no web search, no source gathering).
        """
        elapsed = time.time() - state["start_time"]

        # Build context from all agent responses for the synthesis prompt
        agent_outputs = []
        for response in state["agent_responses"]:
            agent_outputs.append(
                f"=== {response.agent_name.upper()} ===\n"
                f"Confidence: {response.confidence:.2f}\n"
                f"{response.content}"
            )

        conflicts_text = format_conflicts_summary(state["conflicts"])
        quality_text = format_quality_summary(state["quality"])

        # Calculate totals before the synthesis call
        total_tokens = sum(r.tokens_used for r in state["agent_responses"])
        total_cost = sum(r.cost_usd for r in state["agent_responses"])

        # Try LLM-powered synthesis, fall back to basic concatenation if it fails
        summary = await self._synthesize_summary(
            query=state["query"],
            agent_outputs=agent_outputs,
            conflicts_text=conflicts_text,
            quality_text=quality_text,
        )

        # Add synthesis tokens/cost to totals
        total_tokens += self._last_synthesis_tokens
        total_cost += self._last_synthesis_cost

        report = ResearchReport(
            id=str(uuid4()),
            query=state["research_query"],
            summary=summary,
            agent_responses=state["agent_responses"],
            failed_agents=state["failed_agents"],
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            processing_time_seconds=round(elapsed, 2),
        )

        logger.info(
            "[orchestrator] Report built: %d agents, %d tokens, $%.4f, %.1fs",
            len(state["agent_responses"]), total_tokens, total_cost, elapsed,
        )

        return {"report": report}

    async def _synthesize_summary(
        self,
        query: str,
        agent_outputs: list[str],
        conflicts_text: str,
        quality_text: str,
    ) -> str:
        """Use the LLM to synthesize agent outputs into a coherent summary.

        Grabs the LLM from the first available agent (they all share the same
        LLM instance). Falls back to basic concatenation if the LLM call fails.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        self._last_synthesis_tokens = 0
        self._last_synthesis_cost = 0

        # Grab LLM from any agent — they all share the same instance
        any_agent = next(iter(self.agents.values()), None)
        if not any_agent:
            return self._fallback_summary(agent_outputs, conflicts_text, quality_text)

        system_prompt = """You are the SYNTHESIS editor in a multi-agent investment research pipeline.

You receive reports from 4 specialized agents:
- RESEARCHER: raw facts, data points, sources (evidence layer)
- ANALYST: financial interpretation, valuation, metrics (numbers layer)
- SENTIMENT: market mood, narrative, insider activity (psychology layer)
- RISK ASSESSOR: risks, threats, devil's advocate view (adversarial layer)

Your job: Merge these into ONE coherent research summary that a decision-maker can act on.

Synthesis rules:
- NEVER add information that wasn't in the agent reports — you are an editor, not an analyst
- When agents AGREE: state the consensus concisely, cite both
- When agents DISAGREE: present the tension explicitly — "The analyst sees fair valuation at P/E 28, but the risk assessor flags that this assumes 20% growth continuation, which faces [specific threat]"
- Weight reliability: Researcher's confirmed facts > Analyst's metrics > Sentiment signals
- Preserve uncertainty: if an agent flagged low confidence or data gaps, carry that through

Structure (aim for 400-600 words):
1. **Executive Summary** — 2-3 sentences: what is this, what's the verdict, what's the confidence level
2. **Key Findings** — top 3-5 facts from the Researcher, with confirmation tags
3. **Financial Analysis** — Analyst's valuation assessment and key metrics
4. **Market Sentiment** — Sentiment score, narrative, any divergence from fundamentals
5. **Risk Assessment** — Top 2-3 risks from the Risk Assessor with severity ratings
6. **Conclusion** — balanced synthesis: where do the agents agree/disagree? What's the overall picture?

Tone: Professional, balanced, specific. Write for someone who has 2 minutes to read this.
Do NOT give buy/sell recommendations — present the evidence and let the reader decide."""

        human_prompt = f"""Original question: {query}

Here are the reports from our specialized agents:

{"".join(f"{output}" + chr(10) + chr(10) for output in agent_outputs)}
{conflicts_text}

{quality_text}

Synthesize these into a single coherent research summary."""

        try:
            # Use the primary LLM (with fallback chain if available)
            response = await any_agent._call_llm_with_fallback([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ])

            # Track synthesis cost
            usage = response.usage_metadata or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            self._last_synthesis_tokens = input_tokens + output_tokens
            # Use approximate Gemini pricing (much cheaper than Claude)
            self._last_synthesis_cost = (input_tokens + output_tokens) * 0.5 / 1_000_000

            logger.info(
                "[orchestrator] Synthesis complete: %d tokens",
                self._last_synthesis_tokens,
            )
            return response.content

        except Exception as e:
            logger.warning(
                "[orchestrator] LLM synthesis failed, using fallback: %s", e,
            )
            return self._fallback_summary(agent_outputs, conflicts_text, quality_text)

    def _fallback_summary(
        self, agent_outputs: list[str], conflicts_text: str, quality_text: str,
    ) -> str:
        """Basic concatenation fallback if LLM synthesis fails."""
        summary = "\n\n".join(agent_outputs)
        summary += f"\n\n---\n{conflicts_text}\n\n{quality_text}"
        return summary

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def run(self, query: ResearchQuery) -> ResearchReport:
        """Execute the full research pipeline.

        This is what the API/UI calls. One method, one input, one output.

        Args:
            query: The user's research question with settings.

        Returns:
            A complete ResearchReport with all agent analyses.

        LANGGRAPH CONCEPT — ainvoke():
        Runs the compiled graph asynchronously. You pass the initial state,
        and it flows through all nodes, returning the final state.
        """
        session_id = str(uuid4())

        initial_state: OrchestratorState = {
            "query": query.query,
            "research_query": query,
            "session_id": session_id,
            "agent_responses": [],
            "failed_agents": [],
            "conflicts": [],
            "quality": None,
            "report": None,
            "start_time": time.time(),
        }

        logger.info("[orchestrator] Starting research: %s", query.query[:80])

        # Build LangSmith config — attaches metadata to the trace
        # so you can search/filter traces in the dashboard.
        # If LangSmith is not enabled, this is just an inert dict.
        config = get_run_config(
            query=query.query,
            session_id=session_id,
            run_name=f"research: {query.query[:50]}",
        )

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state, config=config)

        report = final_state["report"]
        if report is None:
            raise RuntimeError("Orchestrator completed but no report was generated")

        return report
