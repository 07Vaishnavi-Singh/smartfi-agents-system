"""API routes for the investment research system.

Three endpoints:
  POST /research         — submit a research query, get a job_id back
  GET  /research/{id}    — poll for job status and results
  GET  /health           — service health check

PYTHON CONCEPT — APIRouter:
FastAPI's APIRouter is like Express's Router — a group of related routes
that gets mounted onto the main app. Keeps routes separate from app setup.
TS equivalent: express.Router()
Rust equivalent: actix_web::web::scope()

PYTHON CONCEPT — BackgroundTasks:
FastAPI's BackgroundTasks lets you run a function AFTER the response
is sent. The client gets their response immediately, and the heavy
work happens in the background. Like Node's setImmediate or queueMicrotask,
but for long-running work.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from investment_research_system.api.dependencies import get_job_store, get_orchestrator_factory
from investment_research_system.api.job_store import JobStore
from investment_research_system.models.schemas import ResearchDepth

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request / Response Models ---
# These are separate from the internal schemas (models/schemas.py).
# API models define what the CLIENT sees. Internal schemas define
# what flows between agents. They may look similar but evolve independently.


class ResearchRequest(BaseModel):
    """What the client sends to start a research job."""

    query: str = Field(..., min_length=3, max_length=500)
    focus_areas: list[str] = Field(default_factory=list)
    depth: ResearchDepth = ResearchDepth.standard


class JobCreatedResponse(BaseModel):
    """Returned immediately when a job is created."""

    job_id: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Returned when polling for job status."""

    job_id: str
    status: str
    query: str
    report: dict | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


# --- Background Task ---


async def _run_research_job(
    job_id: str, request: ResearchRequest, store: JobStore, orchestrator_factory
) -> None:
    """Run the orchestrator in the background and update the job store.

    This function is called by FastAPI's BackgroundTasks after the
    POST /research response is sent. The client never waits for this.

    PYTHON CONCEPT — async background work:
    Even though FastAPI's BackgroundTasks runs this, we still need
    'await' for async operations. The background task runner handles
    the event loop for us.
    """
    store.update(job_id, status="running")
    logger.info("[api] Job %s started", job_id)

    try:
        from investment_research_system.models.schemas import ResearchQuery

        orchestrator = orchestrator_factory()
        if orchestrator is None:
            store.update(
                job_id,
                status="failed",
                error="Orchestrator not configured — missing API keys or agents",
            )
            return

        query = ResearchQuery(
            query=request.query,
            focus_areas=request.focus_areas,
            depth=request.depth,
        )

        report = await orchestrator.run(query)

        store.update(
            job_id,
            status="completed",
            report=report.model_dump(mode="json"),
        )
        logger.info("[api] Job %s completed", job_id)

    except Exception as e:
        logger.exception("[api] Job %s failed: %s", job_id, e)
        store.update(job_id, status="failed", error=str(e))


# --- Route Handlers ---


@router.post(
    "/research",
    response_model=JobCreatedResponse,
    status_code=202,
)
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
    orchestrator_factory=Depends(get_orchestrator_factory),
) -> JobCreatedResponse:
    """Submit a research query. Returns immediately with a job_id.

    The orchestrator runs in the background. Poll GET /research/{job_id}
    to check progress.

    PYTHON CONCEPT — Depends():
    FastAPI's dependency injection. Depends(get_job_store) calls
    get_job_store() and passes the result as 'store'.
    TS equivalent: @Inject() in NestJS
    """
    job_id = store.create(query=request.query)

    background_tasks.add_task(_run_research_job, job_id, request, store, orchestrator_factory)

    job = store.get(job_id)
    return JobCreatedResponse(
        job_id=job_id,
        status="pending",
        created_at=job["created_at"],
    )


@router.get(
    "/research/{job_id}",
    response_model=JobStatusResponse,
)
async def get_research_status(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> JobStatusResponse:
    """Check the status of a research job.

    Returns the current status. When status is 'completed',
    the 'report' field contains the full ResearchReport.

    PYTHON CONCEPT — HTTPException:
    FastAPI's way to return error responses. Raise it like an exception,
    FastAPI catches it and returns the right HTTP status code.
    TS equivalent: throw new HttpException(404, 'Not found')
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        query=job["query"],
        report=job["report"],
        error=job["error"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """Service health check. Always returns 200 if the server is running."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
