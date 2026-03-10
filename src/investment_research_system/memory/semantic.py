"""Semantic memory using Mem0.

Stores agent-level facts and learned knowledge. Auto-deduplicates.
This is the "gets smarter over time" piece.

Example facts stored:
- "NVIDIA P/E is 58 as of March 2026"
- "Users asking about tech stocks usually want risk assessment"
- "AMD and NVIDIA are direct competitors in GPU market"

Mem0 automatically:
- Extracts facts from text
- Merges duplicate facts
- Updates stale facts with newer info
"""

import logging

from mem0 import Memory

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Mem0-backed agent fact store.

    PYTHON CONCEPT — composition vs inheritance:
    We WRAP Mem0's Memory class instead of inheriting from it.
    self.memory = Memory()  ← composition (HAS-A)
    class SemanticMemory(Memory)  ← inheritance (IS-A) — DON'T do this

    Why? Composition is more flexible. If Mem0's API changes, you only
    update this wrapper. Your agents never touch Mem0 directly.
    Same principle in TS/Rust: prefer composition over inheritance.
    """

    def __init__(self, config: dict | None = None):
        """Initialize Mem0.

        PYTHON CONCEPT — dict | None = None:
        Same as `config?: Record<string, any>` in TS.
        The `| None` means it can be None, `= None` makes it optional.

        Mem0 config can specify which LLM/embedding to use.
        None = use Mem0's defaults.
        """
        if config:
            self.memory = Memory.from_config(config)
        else:
            self.memory = Memory()

        logger.info("Semantic memory (Mem0) initialized")

    def store(self, content: str, agent_name: str, metadata: dict | None = None) -> dict:
        """Store a fact. Mem0 auto-deduplicates.

        If you store "NVIDIA P/E is 65" and later store "NVIDIA P/E is 58",
        Mem0 updates the old fact instead of creating a duplicate.

        Args:
            content: The fact or knowledge to store.
            agent_name: Which agent learned this (used as user_id in Mem0).
            metadata: Extra context (query, topic, etc.).

        Returns:
            Mem0's response with the stored/updated memory ID.

        PYTHON CONCEPT — `metadata or {}`:
        None or {} → returns {}
        {"key": "val"} or {} → returns {"key": "val"}
        Python's `or` returns the first truthy value.
        Same as `metadata ?? {}` in TS.
        """
        result = self.memory.add(
            content,
            user_id=agent_name,
            metadata=metadata or {},
        )
        logger.info("Stored fact for %s: %s", agent_name, content[:80])
        return result

    def search(self, query: str, agent_name: str | None = None, limit: int = 5) -> list[dict]:
        """Search facts by meaning.

        Args:
            query: Natural language query.
            agent_name: Filter to facts learned by this agent. None = search all.
            limit: Max results.

        Returns:
            List of matching facts with scores.

        PYTHON CONCEPT — **kwargs (keyword argument unpacking):
        `**kwargs` unpacks a dict as function arguments.
        TS equivalent: fn({ query, limit, ...rest })
        Rust: no direct equivalent

        Example:
            kwargs = {"query": "NVIDIA", "limit": 5}
            self.memory.search(**kwargs)
            # same as: self.memory.search(query="NVIDIA", limit=5)
        """
        kwargs: dict = {"query": query, "limit": limit}
        if agent_name:
            kwargs["user_id"] = agent_name

        results = self.memory.search(**kwargs)
        return results.get("results", [])
        # ^ .get("results", []) = safe access with fallback
        #   TS equivalent: results?.results ?? []

    def get_all(self, agent_name: str) -> list[dict]:
        """Get all facts stored by a specific agent.

        Useful for: "What does the researcher agent know?"
        """
        result = self.memory.get_all(user_id=agent_name)
        return result.get("results", [])

    def delete(self, memory_id: str) -> None:
        """Delete a specific fact by its ID."""
        self.memory.delete(memory_id)
        logger.info("Deleted memory: %s", memory_id)

    def delete_all(self, agent_name: str) -> None:
        """Delete all facts for an agent. Use carefully.

        PYTHON CONCEPT — dangerous methods:
        In production, name dangerous methods clearly.
        Some teams prefix with `dangerous_` or require confirmation.
        At minimum, log a warning.
        """
        logger.warning("Deleting ALL memories for agent: %s", agent_name)
        self.memory.delete_all(user_id=agent_name)
