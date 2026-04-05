"""Agent-level tools for the plan-then-execute pattern.

Each agent (researcher, analyst, sentiment, risk) gets these 4 tools
in its ReAct loop. The agent LLM calls them via bind_tools().

PLAN-THEN-EXECUTE PATTERN:
1. create_plan — LLM lists what data it needs (called first)
2. search_web / search_past_research — LLM gathers data per the plan
3. write_analysis — LLM writes its final output (exit tool)

This is more systematic than pure ReAct because the plan prevents
context drift — the agent can't forget what it was looking for.

Tools are created via factory functions that capture dependencies
(tavily, memory) via closure. Same pattern as orchestrator/tools.py.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL FACTORIES
# =============================================================================


def create_plan_tool():
    """Factory: create the create_plan async function.

    Returns:
        An async function that takes (data_needed, _context) and returns str.
    """

    async def create_plan(data_needed: list[dict], _context: dict) -> str:
        """Create a research plan listing what data to gather.

        Call this FIRST before any searches. Each item should have:
        - topic: what to search for (be specific)
        - source: "web" for fresh data or "memory" for past research
        - priority: 1 = must have, 2 = nice to have

        Args:
            data_needed: List of dicts with topic, source, priority.
            _context: Mutable tool context dict.
        """
        if not data_needed:
            return "ERROR: Plan cannot be empty. Provide at least one data item to search for."

        _context["plan"] = data_needed

        # Format plan for LLM to see
        lines = [f"Plan created with {len(data_needed)} items:"]
        for i, item in enumerate(data_needed, 1):
            topic = item.get("topic", "unknown")
            source = item.get("source", "web")
            priority = item.get("priority", 1)
            p_label = "MUST HAVE" if priority == 1 else "NICE TO HAVE"
            lines.append(f"  {i}. [{source.upper()}] {topic} ({p_label})")

        lines.append("\nNow execute this plan by calling search_web or search_past_research for each item.")

        logger.info("[agent_tools] Plan created: %d items", len(data_needed))
        return "\n".join(lines)

    return create_plan


def create_search_web_tool(tavily):
    """Factory: create the search_web async function.

    Args:
        tavily: TavilySearch instance, or None if unavailable.

    Returns:
        An async function that takes (query, topic, time_range) and returns str.
    """

    async def search_web(query: str, topic: str = "finance", time_range: str | None = None) -> str:
        """Search the web for current information using Tavily.

        Args:
            query: What to search for. Be specific.
            topic: "finance", "news", or "general".
            time_range: "day", "week", "month", or None for all time.
        """
        if tavily is None:
            return "Web search unavailable — no Tavily API configured. Use search_past_research instead."

        try:
            results = await asyncio.to_thread(
                tavily.search,
                query=query,
                max_results=5,
                topic=topic,
                time_range=time_range,
            )
        except Exception as e:
            logger.warning("[agent_tools] Web search failed: %s", e)
            return f"Web search failed: {e}. Try a different query or use search_past_research."

        items = results.get("results", [])
        if not items:
            return f"No web results found for: {query}"

        parts = []
        for i, r in enumerate(items, 1):
            title = r.get("title", "Untitled")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            date = r.get("published_date", "")
            parts.append(f"[{i}] {title}\n    {content}\n    Source: {url} ({date})")

        # Include Tavily's AI summary if available
        answer = results.get("answer")
        if answer:
            parts.append(f"\nSummary: {answer}")

        logger.info("[agent_tools] Web search for '%s': %d results", query[:50], len(items))
        return "\n\n".join(parts)

    return search_web


def create_search_past_research_tool(memory):
    """Factory: create the search_past_research async function.

    Args:
        memory: MemoryManager instance with parallel_search().

    Returns:
        An async function that takes (query) and returns str.
    """

    async def search_past_research(query: str) -> str:
        """Search past research in long-term memory (Qdrant + Mem0).

        Returns labeled results: [RESEARCH-N] for past analyses, [FACT-N] for discrete facts.
        """
        try:
            results = await memory.parallel_search(query)
        except Exception as e:
            logger.warning("[agent_tools] Memory search failed: %s", e)
            return f"Memory search failed: {e}. Use search_web instead."

        long_term = results.get("long_term", [])
        semantic = results.get("semantic", [])

        if not long_term and not semantic:
            return "No relevant past research found. Use search_web for fresh data."

        parts = []

        # Filter by relevance threshold (same as base.py _format_memory_context)
        relevant_lt = [item for item in long_term if item.get("score", 0) >= 0.7]
        for i, item in enumerate(relevant_lt[:5], 1):
            text = item.get("text", str(item))[:300]
            score = item.get("score", 0.0)
            parts.append(f"[RESEARCH-{i}] (relevance: {score:.2f}): {text}")

        for i, item in enumerate(semantic[:5], 1):
            text = item.get("memory", str(item))[:200]
            parts.append(f"[FACT-{i}]: {text}")

        logger.info("[agent_tools] Memory search for '%s': %d research, %d facts", query[:50], len(relevant_lt), len(semantic))
        return "\n".join(parts)

    return search_past_research


def create_write_analysis_tool():
    """Factory: create the write_analysis async function.

    Returns:
        An async function that takes (content, _context) and returns str.
    """

    async def write_analysis(content: str, _context: dict) -> str:
        """Write your final analysis. Call this when you have gathered enough data.

        Args:
            content: Your complete analysis text.
            _context: Mutable tool context dict.
        """
        _context["analysis"] = content

        logger.info("[agent_tools] Analysis written: %d chars", len(content))
        return "Analysis complete. Your output has been recorded."

    return write_analysis
