"""In-memory job store for tracking research jobs.

Stores job state in a Python dict. Designed to be swappable
to Redis later — all access goes through the JobStore class,
so the routes never touch the storage directly.

TS equivalent: a class wrapping a Map<string, Job>
Rust equivalent: Arc<Mutex<HashMap<String, Job>>>
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStore:
    """In-memory store for research jobs.

    All job operations go through this class. To swap to Redis later,
    create a RedisJobStore with the same methods and inject it instead.

    PYTHON CONCEPT — typing.Any:
    The 'report' field holds a dict or None. We use Any because
    the report structure is defined by Pydantic models elsewhere —
    by the time it's stored here, it's already been validated.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, query: str) -> str:
        """Create a new job and return its ID.

        PYTHON CONCEPT — uuid4():
        Generates a random UUID. uuid4 is the most common variant —
        it's random, not based on time or MAC address.
        TS equivalent: crypto.randomUUID()
        """
        job_id = str(uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "query": query,
            "report": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
        logger.info("[job_store] Created job %s", job_id)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID. Returns None if not found."""
        return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        status: str | None = None,
        report: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update a job's status, report, or error.

        PYTHON CONCEPT — optional keyword arguments:
        Only the fields you pass get updated. The rest stay unchanged.
        TS equivalent: Partial<Job> spread into existing object.
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("[job_store] Tried to update nonexistent job %s", job_id)
            return

        if status is not None:
            job["status"] = status
        if report is not None:
            job["report"] = report
        if error is not None:
            job["error"] = error
        if status in ("completed", "failed"):
            job["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info("[job_store] Updated job %s → status=%s", job_id, job.get("status"))
