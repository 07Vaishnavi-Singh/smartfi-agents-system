"""Long-term memory using Qdrant with Contextual Retrieval.

Stores past research as vector embeddings. Retrieves by semantic similarity.
Uses contextual retrieval — enriches text with context before embedding.

PYTHON CONCEPT — IMPORTS ORDER (PEP 8):
1. Standard library (uuid, datetime, logging)
2. Third-party packages (qdrant_client, sentence_transformers)
3. Local imports (from config import settings)
Separated by blank lines. ruff enforces this automatically.
"""

import logging
from datetime import datetime
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# PYTHON CONCEPT — logging:
# Never use print() in production code. Use logging.
# It gives you levels (debug/info/warning/error), formatting, and can be
# redirected to files, monitoring tools, etc.
# TS equivalent: winston or pino logger
# Rust equivalent: log + env_logger crates
logger = logging.getLogger(__name__)
#                          ^ __name__ = this module's name (e.g., "memory.long_term")
#                            So logs show WHERE they came from.


class LongTermMemory:
    """Qdrant-backed knowledge store with contextual retrieval.

    PYTHON CONCEPT — class design:
    In Python production code, classes should:
    1. Take dependencies via __init__ (dependency injection — easy to test)
    2. Have a clear single responsibility
    3. Methods should be short and focused

    This is the same in TS/Rust. Python just uses `self` instead of `this`/`self`.
    """

    COLLECTION_NAME = "research_knowledge"
    # ^ PYTHON CONCEPT — class variable vs instance variable:
    # ALL_CAPS = class-level constant. Shared across all instances.
    # self.something = instance variable. Each instance has its own copy.
    # TS equivalent: static readonly COLLECTION_NAME = "..."
    # Rust equivalent: const COLLECTION_NAME: &str = "..."

    def __init__(self, qdrant_url: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = QdrantClient(url=qdrant_url)

        # Load the embedding model (runs locally on your machine, no API key needed)
        # First run downloads the model (~80MB). After that it's cached.
        logger.info("Loading embedding model: %s", embedding_model)
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Ensure collection exists on startup
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist.

        A "collection" in Qdrant = a "table" in SQL.
        It defines the vector size and distance metric.
        """
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]
        # ^ PYTHON CONCEPT — list comprehension:
        # This is Python's version of .map()
        # TS:     collections.map(c => c.name)
        # Python: [c.name for c in collections]
        # You'll use these EVERYWHERE in Python. They're idiomatic.

        if self.COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    # ^ COSINE = measures angle between vectors.
                    # Texts with similar meaning have vectors pointing in the same direction.
                ),
            )
            logger.info("Created Qdrant collection: %s", self.COLLECTION_NAME)

    def _embed(self, text: str) -> list[float]:
        """Convert text to a vector embedding.

        This is the core of vector search:
        "NVIDIA revenue grew 15%" → [0.023, -0.112, 0.847, ...] (384 numbers)

        Texts with similar meaning produce similar vectors.
        """
        # .tolist() converts numpy array → Python list (Qdrant expects a list)
        return self.embedder.encode(text).tolist()

    def _enrich_with_context(self, text: str, metadata: dict) -> str:
        """Contextual Retrieval — enrich text with context before embedding.

        BEFORE: "Revenue grew 15% YoY"
        AFTER:  "From NVIDIA Q3 2024 analysis by researcher agent.
                 Topic: financial performance. Revenue grew 15% YoY"

        This dramatically improves search accuracy because the embedding
        now carries context about WHAT this text is about and WHERE it came from.
        """
        context_parts = []

        if "agent_name" in metadata:
            context_parts.append(f"Analysis by {metadata['agent_name']} agent")
        if "query_context" in metadata:
            context_parts.append(f"Research query: {metadata['query_context']}")
        if "topic" in metadata:
            context_parts.append(f"Topic: {metadata['topic']}")

        if context_parts:
            context_prefix = ". ".join(context_parts) + ". "
            return context_prefix + text

        return text

    def store(self, text: str, metadata: dict | None = None) -> str:
        """Embed text with contextual enrichment and store in Qdrant.

        Returns the point ID (UUID string) for reference.

        PYTHON CONCEPT — `metadata or {}`:
        If metadata is None, `None or {}` returns {}.
        This is like `metadata ?? {}` in TS (nullish coalescing).
        Python's `or` returns the first truthy value.
        """
        metadata = metadata or {}

        # Step 1: Enrich with context (Anthropic's contextual retrieval technique)
        enriched_text = self._enrich_with_context(text, metadata)

        # Step 2: Embed the enriched text
        embedding = self._embed(enriched_text)

        # Step 3: Create the point (Qdrant's term for a stored record)
        point_id = str(uuid4())
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": text,                      # original text (for display)
                "enriched_text": enriched_text,     # what was actually embedded
                "created_at": datetime.now().isoformat(),
                **metadata,
                # ^ PYTHON CONCEPT — dict unpacking:
                # **metadata spreads the dict into this one.
                # TS equivalent: { text, enrichedText, ...metadata }
                # Rust: no direct equivalent (you'd merge HashMaps)
            },
        )

        # Step 4: Upsert (insert or update) into Qdrant
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

        logger.info("Stored in Qdrant: %s (id: %s)", text[:80], point_id)
        return point_id

    # Minimum cosine similarity to include a result.
    # With all-MiniLM-L6-v2, scores roughly mean:
    #   0.7+ = very similar topic, 0.5-0.7 = related, <0.5 = weakly related
    # 0.7 ensures only closely related past research is retrieved,
    # filtering out cross-topic bleed (e.g., ICICI results for a gold query).
    SCORE_THRESHOLD = 0.7

    def search(self, query: str, limit: int = 5, agent_name: str | None = None) -> list[dict]:
        """Find semantically similar past research.

        Args:
            query: What to search for (natural language).
            limit: Max results to return.
            agent_name: Optional filter — only return results from this agent.

        Returns:
            List of dicts with text, metadata, and similarity score.
        """
        query_vector = self._embed(query)

        # Build optional filter
        query_filter = None
        if agent_name:
            query_filter = Filter(
                must=[FieldCondition(key="agent_name", match=MatchValue(value=agent_name))]
            )

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=self.SCORE_THRESHOLD,
        )

        # PYTHON CONCEPT — list comprehension with transformation:
        # TS: results.points.map(hit => ({ text: hit.payload.text, score: hit.score }))
        # Python: [{"text": hit.payload["text"], "score": hit.score} for hit in results.points]
        return [
            {
                "text": hit.payload.get("text", ""),
                "agent_name": hit.payload.get("agent_name", "unknown"),
                "query_context": hit.payload.get("query_context", ""),
                "created_at": hit.payload.get("created_at", ""),
                "score": hit.score,
                # ^ PYTHON CONCEPT — dict.get(key, default):
                # Like ?. with a fallback in TS: hit.payload?.text ?? ""
                # .get() returns the default if key is missing. Never throws.
            }
            for hit in results.points
        ]

    def delete_by_id(self, point_id: str) -> None:
        """Delete a specific point from Qdrant."""
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[point_id],
        )

    def count(self) -> int:
        """Get total number of stored knowledge entries."""
        info = self.client.get_collection(self.COLLECTION_NAME)
        return info.points_count

    def ping(self) -> bool:
        """Health check."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
