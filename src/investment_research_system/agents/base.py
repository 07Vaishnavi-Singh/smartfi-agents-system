"""Base agent class that all specialized agents inherit from.

Every agent (researcher, analyst, risk, sentiment) shares:
- A connection to Claude via LangChain
- Access to memory via MemoryManager
- Access to web search via TavilySearch
- Token/cost tracking
- A standard run() method that returns AgentResponse

PYTHON CONCEPT — ABSTRACT BASE CLASS (ABC):
An ABC defines a contract — "every subclass MUST implement these methods."
If you forget to implement one, Python throws an error at instantiation time,
not at runtime when you call it. Fail fast.

TS equivalent: abstract class BaseAgent { abstract analyze(...): Promise<string> }
Rust equivalent: trait BaseAgent { fn analyze(&self, ...) -> String; }
"""

import logging
import time
from abc import ABC, abstractmethod

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from investment_research_system.memory.manager import MemoryManager
from investment_research_system.models.schemas import AgentResponse, Source
from investment_research_system.tools.tavily_search import TavilySearch

logger = logging.getLogger(__name__)


# --- Cost constants ---
# Approximate per-token costs for Claude Sonnet (USD)
# These are estimates — check Anthropic's pricing page for current rates.
# Used for tracking, not billing.
COST_PER_INPUT_TOKEN = 3.0 / 1_000_000    # $3 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens
# PYTHON CONCEPT — underscores in numbers:
# 1_000_000 == 1000000. Underscores are visual separators, ignored by Python.
# TS equivalent: same! 1_000_000 works in TS too.
# Rust equivalent: same! 1_000_000u64


class BaseAgent(ABC):
    """Abstract base class for all research agents.

    PYTHON CONCEPT — ABC (Abstract Base Class):
    By inheriting from ABC and using @abstractmethod, we enforce:
    1. You can't instantiate BaseAgent directly — it's abstract
    2. Every subclass MUST implement the abstract methods
    3. If they don't, Python raises TypeError immediately

    This is exactly like `abstract class` in TS or `trait` in Rust.

    HOW SUBCLASSES USE THIS:
    ```python
    class ResearcherAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "researcher"

        @property
        def system_prompt(self) -> str:
            return "You are a financial researcher..."

        def build_query(self, query: str, memory_results: dict) -> str:
            # Customize how this agent frames its question to Claude
            return f"Research this: {query}\\nPast knowledge: {memory_results}"
    ```
    """

    def __init__(
        self,
        llm: ChatAnthropic,
        memory: MemoryManager,
        tavily: TavilySearch | None = None,
    ):
        """Initialize with shared dependencies.

        PYTHON CONCEPT — dependency injection:
        We don't create ChatAnthropic or MemoryManager inside the agent.
        They're passed in from outside. This means:
        - Tests can pass mock versions
        - All agents share the same LLM/memory instance
        - Config lives in ONE place (whoever creates these objects)

        Args:
            llm: LangChain ChatAnthropic instance (shared across agents).
            memory: MemoryManager for searching/storing research.
            tavily: Optional web search tool. Not all agents need it.
        """
        self.llm = llm
        self.memory = memory
        self.tavily = tavily

    # =========================================================================
    # ABSTRACT PROPERTIES — subclasses MUST define these
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent's name. Used in logs, memory storage, and AgentResponse.

        PYTHON CONCEPT — @property:
        Makes a method behave like an attribute. You access it as
        `agent.name` not `agent.name()`. It's a getter without parens.

        TS equivalent: get name(): string { return "researcher" }
        Rust equivalent: fn name(&self) -> &str
        """
        ...
    # PYTHON CONCEPT — `...` (Ellipsis):
    # In abstract methods, `...` means "no implementation here."
    # Same as `pass` but signals intent: "this is a placeholder."
    # TS equivalent: abstract name: string;
    # Rust equivalent: fn name(&self) -> &str;  (no body in trait)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The system prompt that defines this agent's personality and role.

        Each agent gets a different system prompt:
        - Researcher: "You search for and synthesize information..."
        - Analyst: "You analyze financial metrics and valuations..."
        - Risk: "You identify risks and potential downsides..."
        - Sentiment: "You gauge market sentiment from news and social..."
        """
        ...

    # =========================================================================
    # ABSTRACT METHODS — subclasses MUST implement these
    # =========================================================================

    @abstractmethod
    def build_query(self, query: str, memory_results: dict) -> str:
        """Build the prompt to send to Claude, using memory context.

        Each agent frames its question differently:
        - Researcher adds past research context
        - Analyst focuses on financial data
        - Risk agent emphasizes potential downsides
        - Sentiment agent focuses on market mood

        Args:
            query: The user's original question.
            memory_results: Results from MemoryManager.parallel_search().

        Returns:
            The formatted prompt string to send to Claude.
        """
        ...

    @abstractmethod
    def extract_sources(self, query: str) -> list[Source]:
        """Gather sources relevant to this agent's analysis.

        Some agents use Tavily (researcher), others use memory (analyst),
        some use both. Each agent defines its own source-gathering strategy.

        Args:
            query: The user's original question.

        Returns:
            List of Source objects to include in AgentResponse.
        """
        ...

    # =========================================================================
    # CONCRETE METHODS — shared by all agents, no override needed
    # =========================================================================

    async def run(self, query: str, session_id: str) -> AgentResponse:
        """Execute this agent's full pipeline.

        This is the main method the orchestrator calls. It:
        1. Searches memory for relevant past knowledge
        2. Gathers sources (web search, memory, etc.)
        3. Builds a prompt with context
        4. Calls Claude
        5. Tracks tokens and cost
        6. Stores results back to memory
        7. Returns a standardized AgentResponse

        PYTHON CONCEPT — async def:
        This is async because memory and LLM calls are I/O operations.
        The orchestrator can run multiple agents concurrently with
        asyncio.gather() — same as Promise.all() in TS.

        Args:
            query: The user's original research question.
            session_id: Current session ID for memory tracking.

        Returns:
            AgentResponse with analysis, sources, and cost tracking.
        """
        logger.info("[%s] Starting analysis for: %s", self.name, query[:80])
        start_time = time.time()

        # Step 1: Update status in Redis
        await self.memory.update_agent_status(session_id, self.name, "running")

        # Step 2: Search memory for past knowledge
        memory_results = await self.memory.parallel_search(query)
        logger.info(
            "[%s] Memory search returned %d long-term, %d semantic results",
            self.name,
            len(memory_results.get("long_term", [])),
            len(memory_results.get("semantic", [])),
        )

        # Step 3: Gather sources (each agent implements its own strategy)
        sources = self.extract_sources(query)

        # Step 4: Build the prompt with memory context
        user_prompt = self.build_query(query, memory_results)

        # Step 5: Call Claude via LangChain
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # PYTHON CONCEPT — ainvoke (async invoke):
        # LangChain's async version of invoke(). Non-blocking.
        # `invoke()` = synchronous (blocks the event loop)
        # `ainvoke()` = asynchronous (other agents can run while waiting)
        response: AIMessage = await self.llm.ainvoke(messages)

        # Step 6: Calculate tokens and cost
        token_usage = response.usage_metadata or {}
        # PYTHON CONCEPT — duck typing:
        # We don't check the type of usage_metadata. We just call .get()
        # on it. If it's a dict, great. If it's None, we used `or {}`.
        # Python doesn't care about the type — only that it has .get().
        # TS would need: (response.usage_metadata as Record<string, number>)
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)

        elapsed_ms = (time.time() - start_time) * 1000

        # Step 7: Store results in memory for future queries
        await self.memory.store_research(
            content=response.content,
            agent_name=self.name,
            session_id=session_id,
            metadata={"query_context": query},
        )

        # Step 8: Update status to done
        await self.memory.update_agent_status(session_id, self.name, "done")

        logger.info(
            "[%s] Complete in %.0fms | %d tokens | $%.4f",
            self.name, elapsed_ms, total_tokens, cost,
        )

        return AgentResponse(
            agent_name=self.name,
            content=response.content,
            confidence=self._estimate_confidence(response.content, memory_results),
            sources=sources,
            tokens_used=total_tokens,
            cost_usd=cost,
            latency_ms=elapsed_ms,
        )

    def _estimate_confidence(self, content: str, memory_results: dict) -> float:
        """Estimate confidence based on response quality signals.

        Simple heuristic — subclasses can override for smarter logic.
        Factors:
        - Longer responses suggest more thorough analysis
        - Having memory context increases confidence
        - Having sources increases confidence

        Returns a float between 0.0 and 1.0.
        """
        score = 0.5  # base confidence

        # More content = more thorough (up to a point)
        if len(content) > 500:
            score += 0.1
        if len(content) > 1500:
            score += 0.1

        # Had relevant past knowledge
        if memory_results.get("long_term"):
            score += 0.1
        if memory_results.get("semantic"):
            score += 0.1

        # Cap at 0.95 — never claim 100% confidence
        return min(score, 0.95)

    def _format_memory_context(self, memory_results: dict) -> str:
        """Format memory search results into a readable string for the prompt.

        Shared helper — subclasses call this in their build_query().
        Converts the raw memory dict into a text block Claude can understand.

        PYTHON CONCEPT — list comprehension with conditional:
        [x for x in items if condition]
        TS equivalent: items.filter(condition).map(x => ...)
        """
        parts = []

        long_term = memory_results.get("long_term", [])
        if long_term:
            parts.append("=== Past Research ===")
            for item in long_term[:3]:
                # Only include top 3 most relevant results
                text = item.get("text", "")
                score = item.get("score", 0)
                parts.append(f"[relevance: {score:.2f}] {text[:300]}")

        semantic = memory_results.get("semantic", [])
        if semantic:
            parts.append("\n=== Known Facts ===")
            for fact in semantic[:5]:
                # Include up to 5 facts
                memory_text = fact.get("memory", "")
                parts.append(f"- {memory_text}")

        if not parts:
            return "No relevant past knowledge found."

        return "\n".join(parts)
