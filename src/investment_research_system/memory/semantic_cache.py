"""Two-tier query cache: Exact (Redis) + Semantic (Qdrant).

Production-grade systems use BOTH cache types because they catch different cases:

  Tier 1 — Exact Cache (Redis):
    Key: normalized query string → Value: report JSON
    Catches: identical or near-identical queries (word-for-word)
    Speed: ~0.1ms (Redis key-value lookup)
    Example: "Should I invest in NVIDIA?" → exact match

  Tier 2 — Semantic Cache (Qdrant):
    Key: query embedding → Value: report JSON
    Catches: semantically similar queries (different wording, same meaning)
    Speed: ~50ms (vector similarity search)
    Example: "Is NVIDIA a good buy?" ≈ "Should I invest in NVIDIA?" (cosine 0.94)

Flow in routes.py:
  1. Check exact cache first (fastest, ~0.1ms)
  2. If miss → check semantic cache (~50ms)
  3. If miss → run orchestrator ($0.30)
  4. On completion → store in BOTH caches

TS equivalent: Redis GET for exact, RediSearch FT.SEARCH for semantic
Rust equivalent: DashMap for exact, HNSW index for semantic
"""

import hashlib
import json
import logging
import time
from uuid import uuid4

import redis

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =============================================================================
# TIER 1: EXACT CACHE (Redis)
# =============================================================================


class ExactCache:
    """Redis-backed exact-match cache for research queries.

    The fastest cache tier — simple key-value lookup (~0.1ms).
    Catches identical or near-identical queries (after normalization).

    Normalization: lowercase, strip whitespace, collapse spaces.
    "Should I invest in  NVIDIA?" → "should i invest in nvidia?"

    Key format: cache:exact:{md5_hash}
    Value: JSON-encoded report
    TTL: 1 hour (configurable)

    PYTHON CONCEPT — hashlib for cache keys:
    We hash the normalized query to get a fixed-length key.
    This prevents issues with very long queries as Redis keys.
    TS equivalent: crypto.createHash('md5').update(query).digest('hex')
    """

    KEY_PREFIX = "cache:exact"

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self._ttl = ttl
        self._redis: redis.Redis | None = None
        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("[exact_cache] Connected to Redis")
        except Exception as e:
            logger.warning("[exact_cache] Redis unavailable: %s", e)
            self._redis = None

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize a query for exact matching.

        Lowercase, strip, collapse multiple spaces.
        "Should I invest in  NVIDIA?" → "should i invest in nvidia?"
        """
        return " ".join(query.lower().strip().split())

    @staticmethod
    def _hash_key(normalized: str) -> str:
        """Create a fixed-length cache key from normalized query."""
        return hashlib.md5(normalized.encode()).hexdigest()

    def search(self, query: str) -> dict | None:
        """Look up an exact-match cached report.

        Returns the cached report dict if found, None on miss.
        ~0.1ms latency (single Redis GET).
        """
        if not self._redis:
            return None

        normalized = self._normalize(query)
        key = f"{self.KEY_PREFIX}:{self._hash_key(normalized)}"

        try:
            cached = self._redis.get(key)
            if cached:
                logger.info("[exact_cache] HIT for: '%s'", query[:60])
                return json.loads(cached)
        except Exception as e:
            logger.warning("[exact_cache] Search failed: %s", e)

        return None

    def store(self, query: str, report: dict) -> None:
        """Store a query-report pair for exact matching.

        The report is stored as JSON with a TTL. Auto-expires.
        """
        if not self._redis:
            return

        normalized = self._normalize(query)
        key = f"{self.KEY_PREFIX}:{self._hash_key(normalized)}"

        try:
            self._redis.set(key, json.dumps(report, default=str), ex=self._ttl)
            logger.info("[exact_cache] Stored for: '%s'", query[:60])
        except Exception as e:
            logger.warning("[exact_cache] Store failed: %s", e)

    def invalidate(self, query: str) -> None:
        """Remove a specific query from the cache."""
        if not self._redis:
            return
        normalized = self._normalize(query)
        key = f"{self.KEY_PREFIX}:{self._hash_key(normalized)}"
        try:
            self._redis.delete(key)
        except Exception:
            pass

    def invalidate_all(self) -> None:
        """Clear all exact cache entries."""
        if not self._redis:
            return
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self.KEY_PREFIX}:*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
            logger.info("[exact_cache] All entries cleared")
        except Exception as e:
            logger.warning("[exact_cache] Invalidate failed: %s", e)


# =============================================================================
# TIER 2: SEMANTIC CACHE (Qdrant)
# =============================================================================


class SemanticCache:
    """Qdrant-backed semantic similarity cache for research reports."""

    COLLECTION_NAME = "query_cache"

    def __init__(
        self,
        qdrant_url: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        default_threshold: float = 0.92,
        max_age_seconds: int = 3600,
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.default_threshold = default_threshold
        self.max_age_seconds = max_age_seconds
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the cache collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created semantic cache collection: %s", self.COLLECTION_NAME)

    def _embed(self, text: str) -> list[float]:
        """Convert query text to a vector embedding."""
        return self.embedder.encode(text).tolist()

    def search(
        self,
        query: str,
        threshold: float | None = None,
        max_age_seconds: int | None = None,
    ) -> dict | None:
        """Search for a semantically similar cached query.

        Returns the cached report dict if a match is found above the threshold
        and within the max age window. Returns None on cache miss.

        Args:
            query: The user's research question.
            threshold: Minimum cosine similarity (0.0-1.0). Default 0.92.
            max_age_seconds: Max age of cached result in seconds. Default 3600.

        Returns:
            The cached report dict, or None if no match.
        """
        threshold = threshold or self.default_threshold
        max_age = max_age_seconds or self.max_age_seconds

        query_vector = self._embed(query)

        try:
            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                limit=1,
                score_threshold=threshold,
            )
        except Exception as e:
            logger.warning("[semantic_cache] Search failed: %s", e)
            return None

        if not results.points:
            return None

        hit = results.points[0]
        cached_at = hit.payload.get("cached_at", 0)

        # TTL check — skip stale results
        if time.time() - cached_at > max_age:
            logger.info(
                "[semantic_cache] Cache hit but expired (age: %.0fs, max: %ds)",
                time.time() - cached_at,
                max_age,
            )
            return None

        logger.info(
            "[semantic_cache] Cache HIT (score: %.3f, query: '%s')",
            hit.score,
            query[:60],
        )

        report_json = hit.payload.get("report")
        if report_json:
            try:
                return json.loads(report_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning("[semantic_cache] Failed to parse cached report")
                return None

        return None

    def store(self, query: str, report: dict) -> str:
        """Cache a query-report pair for future semantic matching.

        Args:
            query: The original research question.
            report: The full ResearchReport as a dict (already model_dump'd).

        Returns:
            The Qdrant point ID.
        """
        query_vector = self._embed(query)
        point_id = str(uuid4())

        point = PointStruct(
            id=point_id,
            vector=query_vector,
            payload={
                "query": query,
                "report": json.dumps(report, default=str),
                "cached_at": time.time(),
            },
        )

        try:
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point],
            )
            logger.info(
                "[semantic_cache] Stored cache entry for: '%s' (id: %s)",
                query[:60],
                point_id,
            )
        except Exception as e:
            logger.warning("[semantic_cache] Failed to store: %s", e)

        return point_id

    def invalidate_all(self) -> None:
        """Clear the entire cache. Useful after model updates or data refreshes."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._ensure_collection()
            logger.info("[semantic_cache] Cache invalidated")
        except Exception as e:
            logger.warning("[semantic_cache] Failed to invalidate: %s", e)

    def count(self) -> int:
        """Number of cached entries."""
        try:
            info = self.client.get_collection(self.COLLECTION_NAME)
            return info.points_count
        except Exception:
            return 0
