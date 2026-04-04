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

LLM_CALL_TIMEOUT_SECONDS = 30


# =============================================================================
# SHARED RETRY LOGIC
# =============================================================================


async def _run_agent_with_retry(agent, query, session_id, user_profile_context, circuit_breaker, max_retries, retry_delay):
    """Run a single agent with retry + circuit breaker.

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

        except AgentError as e:
            last_error_type, last_error_message = "agent_error", str(e)
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))

        except Exception as e:
            last_error_type = "unexpected"
            last_error_message = f"Unexpected error: {type(e).__name__}: {e}"
            logger.error("[tools] %s unexpected error (attempt %d/%d): %s", agent.name, attempt + 1, max_retries + 1, e, exc_info=True)
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (attempt + 1))

    circuit_breaker.record_failure(agent.name)
    return None, {"agent": agent.name, "error_type": last_error_type, "error_message": last_error_message}


# =============================================================================
# TOOL FACTORIES
# =============================================================================


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
        import time as time_mod
        from uuid import uuid4

        from investment_research_system.models.schemas import ResearchReport
        from investment_research_system.orchestrator.conflict import detect_conflicts, format_conflicts_summary
        from investment_research_system.orchestrator.quality import assess_quality, format_quality_summary

        # Merge all_responses + new_responses from this turn's dispatch_agents calls.
        # Handles the case where dispatch_agents and build_report are called
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
            report = ResearchReport(
                id=str(uuid4()),
                query=research_query,
                summary="Unable to gather sufficient data to answer this query. All research agents failed or were not dispatched.",
                agent_responses=[],
                failed_agents=failed_agents,
                total_cost_usd=total_cost,
                total_tokens=total_tokens,
                processing_time_seconds=round(time_mod.time() - start_time, 2),
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

        report = ResearchReport(
            id=str(uuid4()),
            query=research_query,
            summary=summary,
            agent_responses=agent_responses,
            failed_agents=failed_agents,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            processing_time_seconds=round(time_mod.time() - start_time, 2),
        )
        _context["report"] = report
        return "Report built successfully."

    return build_report
