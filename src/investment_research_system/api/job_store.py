"""Job store for tracking research jobs — Redis-backed for production.

Stores job state in Redis hashes with auto-expiration (TTL).
Survives pod restarts, supports multiple API pods, and auto-cleans old jobs.

The interface is identical to the old in-memory version — routes.py
doesn't need any changes. Swapped via dependency injection.

TS equivalent: a class wrapping Redis hashes behind a simple CRUD interface
Rust equivalent: a trait JobStore with a Redis impl
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import redis

logger = logging.getLogger(__name__)

# Default TTL for job entries — auto-cleanup after 2 hours.
# Jobs older than this are expired by Redis automatically. No cron job needed.
JOB_TTL_SECONDS = 7200


class JobStore:
    """Redis-backed store for research jobs.

    Each job is stored as a Redis hash at key "job:{job_id}".
    Fields: job_id, status, query, report (JSON string), error, created_at, completed_at.

    Falls back to in-memory dict if Redis is unavailable (graceful degradation).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = JOB_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._redis: redis.Redis | None = None
        self._fallback: dict[str, dict[str, Any]] = {}

        try:
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("[job_store] Connected to Redis")
        except Exception as e:
            logger.warning("[job_store] Redis unavailable, falling back to in-memory: %s", e)
            self._redis = None

    def _key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def create(self, query: str) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "query": query,
            "report": "",
            "error": "",
            "created_at": now,
            "completed_at": "",
        }

        if self._redis:
            try:
                self._redis.hset(self._key(job_id), mapping=job_data)
                self._redis.expire(self._key(job_id), self._ttl)
            except Exception as e:
                logger.warning("[job_store] Redis write failed, using fallback: %s", e)
                self._fallback[job_id] = job_data
        else:
            self._fallback[job_id] = job_data

        logger.info("[job_store] Created job %s", job_id)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID. Returns None if not found or expired."""
        if self._redis:
            try:
                data = self._redis.hgetall(self._key(job_id))
                if not data:
                    return self._fallback.get(job_id)
                # Convert empty strings back to None for API compatibility
                return {
                    "job_id": data.get("job_id", job_id),
                    "status": data.get("status", "unknown"),
                    "query": data.get("query", ""),
                    "report": json.loads(data["report"]) if data.get("report") else None,
                    "error": data.get("error") or None,
                    "created_at": data.get("created_at", ""),
                    "completed_at": data.get("completed_at") or None,
                }
            except Exception as e:
                logger.warning("[job_store] Redis read failed: %s", e)
                return self._fallback.get(job_id)
        return self._fallback.get(job_id)

    def update(
        self,
        job_id: str,
        status: str | None = None,
        report: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update a job's status, report, or error."""
        updates: dict[str, str] = {}
        if status is not None:
            updates["status"] = status
        if report is not None:
            updates["report"] = json.dumps(report, default=str)
        if error is not None:
            updates["error"] = error
        if status in ("completed", "failed"):
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()

        if self._redis:
            try:
                if self._redis.exists(self._key(job_id)):
                    self._redis.hset(self._key(job_id), mapping=updates)
                else:
                    # Job might be in fallback
                    self._update_fallback(job_id, updates)
            except Exception as e:
                logger.warning("[job_store] Redis update failed: %s", e)
                self._update_fallback(job_id, updates)
        else:
            self._update_fallback(job_id, updates)

        logger.info("[job_store] Updated job %s → status=%s", job_id, status)

    def _update_fallback(self, job_id: str, updates: dict) -> None:
        """Update job in the in-memory fallback store."""
        job = self._fallback.get(job_id)
        if job is None:
            logger.warning("[job_store] Tried to update nonexistent job %s", job_id)
            return
        for key, value in updates.items():
            if key == "report" and isinstance(value, str):
                try:
                    job[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    job[key] = value
            else:
                job[key] = value

    def count_active(self) -> int:
        """Count jobs with status 'pending' or 'running'.

        Used for backpressure — reject new jobs if too many are in-flight.
        Note: This scans job keys, which is fine for moderate job counts.
        For high-throughput, maintain a separate counter with INCR/DECR.
        """
        if self._redis:
            try:
                count = 0
                for key in self._redis.scan_iter(match="job:*", count=100):
                    status = self._redis.hget(key, "status")
                    if status in ("pending", "running"):
                        count += 1
                return count
            except Exception:
                pass
        # Fallback
        return sum(1 for j in self._fallback.values() if j.get("status") in ("pending", "running"))
