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

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from investment_research_system.api.dependencies import get_job_store, get_orchestrator_factory
from investment_research_system.api.job_store import JobStore
from investment_research_system.errors import PromptInjectionError
from investment_research_system.models.schemas import ResearchDepth
from investment_research_system.models.user import UserProfile, UserProfileUpdate
from investment_research_system.security.input_guard import sanitize_query

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request / Response Models ---
# These are separate from the internal schemas (models/schemas.py).
# API models define what the CLIENT sees. Internal schemas define
# what flows between agents. They may look similar but evolve independently.


class ResearchRequest(BaseModel):
    """What the client sends to start a research job."""

    query: str = Field(..., min_length=3, max_length=500)
    user_id: str | None = None  # optional — enables personalized analysis via KG profile
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
    tracing_enabled: bool
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

        # Build orchestrator off the event loop so heavy model/memory init
        # doesn't block /health and /research/{job_id} polling requests.
        orchestrator = await asyncio.to_thread(orchestrator_factory)
        if orchestrator is None:
            store.update(
                job_id,
                status="failed",
                error="Orchestrator not configured — missing API keys or agents",
            )
            return

        query = ResearchQuery(
            query=request.query,
            user_id=request.user_id,
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
    # SECURITY — Layer 1: Check for prompt injection before any LLM call.
    # Catches known attack patterns early, returns 400 immediately.
    try:
        sanitize_query(request.query)
    except PromptInjectionError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
    from investment_research_system.observability.tracing import is_tracing_enabled

    return HealthResponse(
        status="ok",
        version="0.1.0",
        tracing_enabled=is_tracing_enabled(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# USER PROFILE ENDPOINTS (Knowledge Graph — Graph 1)
# =============================================================================
# These endpoints manage user profiles stored in Neo4j.
# The profile data is injected into agent prompts for personalized analysis.
#
# Flow: Streamlit form → POST /users/onboard → Neo4j graph
#       Research query with user_id → orchestrator fetches profile → agents see it


@router.post("/users/onboard", status_code=201)
async def onboard_user(
    profile: UserProfile,
    orchestrator_factory=Depends(get_orchestrator_factory),
) -> dict:
    """Create a new user profile in Neo4j.

    This is the onboarding endpoint — called once when a user first
    signs up and fills out the financial profile form.

    Creates nodes (User, Goal, Location, etc.) and edges
    (HAS_GOAL, LOCATED_IN, etc.) in a single transaction.

    Returns 409 if user_id already exists.
    """
    # Late import to avoid circular imports — same pattern as create_orchestrator().
    # We need the GraphMemory instance from an orchestrator's agent.
    graph = _get_graph_memory(orchestrator_factory)
    if graph is None:
        raise HTTPException(status_code=503, detail="Neo4j is not available")

    try:
        result = await asyncio.to_thread(graph.create_user_profile, profile)
        return result
    except ValueError as e:
        # User already exists — 409 Conflict
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/users/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    orchestrator_factory=Depends(get_orchestrator_factory),
) -> dict:
    """Get the current user profile from Neo4j.

    Returns the active profile (only edges where to IS NULL).
    Historical data is preserved in the graph but not returned here.
    """
    graph = _get_graph_memory(orchestrator_factory)
    if graph is None:
        raise HTTPException(status_code=503, detail="Neo4j is not available")

    profile = await asyncio.to_thread(graph.get_user_profile, user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    return profile


@router.put("/users/{user_id}/profile")
async def update_user_profile(
    user_id: str,
    updates: UserProfileUpdate,
    orchestrator_factory=Depends(get_orchestrator_factory),
) -> dict:
    """Update user profile fields using append-only temporal versioning.

    Only send the fields that changed — unchanged fields are left as-is.
    Old values are preserved with a `to` date (closed edges), new values
    get `to: null` (active edges).

    TS analogy: This is like PATCH /users/:id — partial update.
    """
    graph = _get_graph_memory(orchestrator_factory)
    if graph is None:
        raise HTTPException(status_code=503, detail="Neo4j is not available")

    try:
        result = await asyncio.to_thread(graph.update_user_profile, user_id, updates)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _get_graph_memory(orchestrator_factory):
    """Helper to get GraphMemory from an orchestrator instance.

    The orchestrator holds agents, agents hold MemoryManager,
    MemoryManager holds GraphMemory. This chain is how we access Neo4j
    without adding a separate dependency injection path for GraphMemory.

    Returns None if Neo4j is not configured or orchestrator creation fails.
    """
    try:
        orchestrator = orchestrator_factory()
        if orchestrator is None:
            return None
        # Grab memory manager from any agent
        any_agent = next(iter(orchestrator.agents.values()), None)
        if any_agent and hasattr(any_agent.memory, "graph"):
            return any_agent.memory.graph
        return None
    except Exception:
        return None
