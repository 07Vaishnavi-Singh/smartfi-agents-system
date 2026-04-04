"""Unified Memory Manager.

Agents use ONLY this class to interact with memory.
They never touch Redis/Qdrant/Mem0 directly.

PYTHON CONCEPT — FACADE PATTERN:
This class is a "facade" — a simple interface over complex subsystems.
Your agents call memory_manager.parallel_search() instead of
separately calling qdrant.search(), mem0.search(), redis.get().

Same pattern in TS: a service class that wraps multiple clients.
Rust equivalent: a trait that abstracts over multiple backends.

WHY: If you swap Qdrant for Pinecone tomorrow, you change THIS file only.
     No agent code changes. This is dependency inversion.
"""

import asyncio
import logging

from investment_research_system.memory.episodic import EpisodicMemory
from investment_research_system.memory.graph import GraphMemory
from investment_research_system.memory.long_term import LongTermMemory
from investment_research_system.memory.semantic import SemanticMemory
from investment_research_system.memory.short_term import ShortTermMemory
from investment_research_system.memory.topic_extractor import extract_topics

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified interface to all memory systems.

    PYTHON CONCEPT — dependency injection via __init__:
    Instead of creating Redis/Qdrant/Mem0 clients inside this class,
    we receive them as parameters. This makes testing easy:
    - Production: pass real clients
    - Tests: pass mock/fake clients

    TS equivalent: constructor(private shortTerm: ShortTermMemory, ...)
    Rust equivalent: fn new(short_term: impl ShortTermMemory, ...)
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        semantic: SemanticMemory,
        episodic: EpisodicMemory | None = None,
        graph: GraphMemory | None = None,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.semantic = semantic
        self.episodic = episodic
        # Graph memory (Neo4j) — used by the orchestrator for profile reads.
        # Profile fetching happens in the orchestrator (once per query), NOT
        # in parallel_search(). This avoids 4x redundant Neo4j reads since
        # all agents share the same user context.
        self.graph = graph

    # =========================================================================
    # SESSION MANAGEMENT (Redis)
    # =========================================================================

    async def create_session(self, session_id: str, query: str) -> None:
        """Start a new research session in Redis + Postgres."""
        await self.short_term.store(
            session_id,
            "metadata",
            {"query": query, "status": "started"},
        )
        if self.episodic:
            await asyncio.to_thread(self.episodic.create_episode, session_id, query)
        logger.info("Session created: %s", session_id)

    async def update_session_status(self, session_id: str, status: str) -> None:
        """Update overall session status."""
        await self.short_term.store(session_id, "status", {"status": status})

    async def update_agent_status(
        self, session_id: str, agent_name: str, status: str
    ) -> None:
        """Track agent progress within a session."""
        await self.short_term.update_agent_status(session_id, agent_name, status)

    async def get_session_state(self, session_id: str) -> dict:
        """Get full session state including agent statuses."""
        metadata = await self.short_term.recall(session_id, "metadata")
        agent_statuses = await self.short_term.get_agent_statuses(session_id)
        return {
            "metadata": metadata or {},
            "agent_statuses": agent_statuses,
        }

    # =========================================================================
    # PARALLEL SEARCH (the core agentic search pattern)
    # =========================================================================

    async def parallel_search(self, query: str) -> dict:
        """Search all memory types simultaneously, with topic-based filtering.

        This is the PARALLEL-FIRST strategy:
        - Extract topics from the query (instant keyword matching)
        - Fire Qdrant (topic-filtered) + Mem0 at the same time
        - Agent evaluates results AFTER both return

        Topic filtering prevents context contamination: if you ask about gold,
        only research tagged with ["gold", "commodities"] is searched in Qdrant.
        ICICI research (tagged "indian_equities") is structurally excluded —
        it never enters the similarity ranking.

        PYTHON CONCEPT — asyncio.gather():
        Runs multiple async functions concurrently (not sequentially).
        TS equivalent: Promise.all([searchQdrant(), searchMem0()])
        Rust equivalent: tokio::join!(search_qdrant(), search_mem0())

        PYTHON CONCEPT — asyncio.to_thread():
        Qdrant and Mem0 clients are synchronous (blocking).
        asyncio.to_thread() runs them in a thread pool so they don't block
        the event loop. This is how you mix sync + async code in Python.
        TS doesn't need this because everything is async by default.
        """
        # Step 1: Extract topics from the query (instant, no LLM call)
        topics = extract_topics(query)
        logger.info("Query topics: %s for query: %s", topics, query[:80])

        # Step 2: Run both searches in parallel. Treat each backend as optional:
        # a Mem0 dimension/config issue should not block all agent analysis.
        # Qdrant gets topic-filtered; Mem0 doesn't support topic filtering
        # natively (it uses its own fact-matching), so it searches broadly.
        qdrant_results, mem0_results = await asyncio.gather(
            asyncio.to_thread(self.long_term.search, query, 5, None, topics),
            asyncio.to_thread(self.semantic.search, query),
            return_exceptions=True,
            # ^ to_thread(function, arg1, arg2) runs the sync function
            #   in a separate thread, returning an awaitable.
        )

        if isinstance(qdrant_results, Exception):
            logger.warning("Long-term memory search failed: %s", qdrant_results)
            qdrant_results = []
        if isinstance(mem0_results, Exception):
            logger.warning("Semantic memory search failed: %s", mem0_results)
            mem0_results = []

        logger.info(
            "Parallel search complete: %d from Qdrant, %d from Mem0 (topics: %s)",
            len(qdrant_results),
            len(mem0_results),
            topics,
        )

        return {
            "long_term": qdrant_results,    # past research chunks (topic-filtered)
            "semantic": mem0_results,        # agent-level facts
            "topics": topics,               # pass topics downstream for storage
        }

    # =========================================================================
    # STORE RESEARCH RESULTS (after a query completes)
    # =========================================================================

    async def store_research(
        self,
        content: str,
        agent_name: str,
        session_id: str,
        metadata: dict | None = None,
        topics: list[str] | None = None,
    ) -> None:
        """Store research results across all relevant memory types.

        Called after an agent produces output:
        1. Qdrant: store the full research text with topic tags (for future semantic search)
        2. Mem0: store extracted facts (for agent-level knowledge)
        3. Redis: update session with latest results

        Args:
            content: The research text to store.
            agent_name: Which agent produced this.
            session_id: Current session ID.
            metadata: Extra metadata (query_context, etc.).
            topics: Topic tags extracted from the query (e.g., ["gold", "commodities"]).
                    Stored in Qdrant payload for future topic-filtered retrieval.

        PYTHON CONCEPT — dict merging with `|` operator (Python 3.9+):
        dict1 | dict2 merges two dicts. Like { ...dict1, ...dict2 } in TS.
        """
        metadata = metadata or {}

        storage_metadata = {
            "agent_name": agent_name,
            "session_id": session_id,
        } | metadata

        # Include topic tags so future searches can filter by topic.
        # If no topics provided, extract them from the query context.
        if topics:
            storage_metadata["topics"] = topics
        elif "query_context" in metadata:
            storage_metadata["topics"] = extract_topics(metadata["query_context"])
        # ^ merges the two dicts. metadata values override if keys conflict.

        # Store in Qdrant (long-term — embedded with contextual enrichment)
        # and Mem0 (semantic) independently so one backend failure doesn't
        # prevent returning agent output.
        try:
            await asyncio.to_thread(self.long_term.store, content, storage_metadata)
        except Exception as e:
            logger.warning("Long-term memory store failed for %s: %s", agent_name, e)

        try:
            await asyncio.to_thread(self.semantic.store, content, agent_name, storage_metadata)
        except Exception as e:
            logger.warning("Semantic memory store failed for %s: %s", agent_name, e)

        # Update session with latest result
        await self.short_term.store(
            session_id,
            f"result:{agent_name}",
            {"content": content[:500], "agent_name": agent_name},
        )

        # Store in Postgres (episodic — decision audit trail)
        if self.episodic:
            await asyncio.to_thread(
                self.episodic.log_decision,
                session_id,
                agent_name,
                "research",
                content[:500],
            )

        logger.info("Research stored for agent %s in session %s", agent_name, session_id)

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def end_session(
        self, session_id: str, total_tokens: int = 0, total_cost_usd: float = 0.0
    ) -> None:
        """Clean up Redis session and mark Postgres episode as completed."""
        await self.short_term.delete_session(session_id)
        if self.episodic:
            await asyncio.to_thread(
                self.episodic.complete_episode, session_id, total_tokens, total_cost_usd
            )
        logger.info("Session ended: %s", session_id)

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def health_check(self) -> dict[str, bool]:
        """Check all memory services are reachable.

        PYTHON CONCEPT — dict comprehension:
        Like list comprehension but creates a dict.
        TS: Object.fromEntries(checks.map(([k, v]) => [k, v]))
        """
        redis_ok = await self.short_term.ping()
        qdrant_ok = await asyncio.to_thread(self.long_term.ping)
        postgres_ok = False
        if self.episodic:
            postgres_ok = await asyncio.to_thread(self.episodic.ping)

        neo4j_ok = False
        if self.graph:
            neo4j_ok = await asyncio.to_thread(self.graph.ping)

        return {
            "redis": redis_ok,
            "qdrant": qdrant_ok,
            "postgres": postgres_ok,
            "neo4j": neo4j_ok,
        }
