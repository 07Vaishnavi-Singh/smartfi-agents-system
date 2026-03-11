"""Tests for the FastAPI API layer.

Run: PYTHONPATH=src uv run pytest tests/test_api.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# --- JobStore Unit Tests ---

def test_job_store_create_and_get():
    """Create a job and retrieve it by ID."""
    from investment_research_system.api.job_store import JobStore

    store = JobStore()
    job_id = store.create(query="Should I invest in NVIDIA?")

    job = store.get(job_id)
    assert job is not None
    assert job["status"] == "pending"
    assert job["query"] == "Should I invest in NVIDIA?"
    assert job["report"] is None
    assert job["error"] is None


def test_job_store_update_status():
    """Update a job's status and report."""
    from investment_research_system.api.job_store import JobStore

    store = JobStore()
    job_id = store.create(query="Test query")

    store.update(job_id, status="running")
    assert store.get(job_id)["status"] == "running"

    store.update(job_id, status="completed", report={"summary": "done"})
    job = store.get(job_id)
    assert job["status"] == "completed"
    assert job["report"] == {"summary": "done"}


def test_job_store_update_failed():
    """Mark a job as failed with an error message."""
    from investment_research_system.api.job_store import JobStore

    store = JobStore()
    job_id = store.create(query="Test query")

    store.update(job_id, status="failed", error="LLM timeout")
    job = store.get(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "LLM timeout"


def test_job_store_get_nonexistent():
    """Getting a nonexistent job returns None."""
    from investment_research_system.api.job_store import JobStore

    store = JobStore()
    assert store.get("fake-id") is None


# --- API Route Tests (using FastAPI TestClient) ---

import pytest


@pytest.fixture
def client():
    """Create a test client with a fresh job store per test.

    PYTHON CONCEPT — @pytest.fixture:
    A fixture is a function that provides test setup.
    pytest automatically calls it and passes the result
    to any test that declares it as a parameter.
    TS equivalent: beforeEach() that returns a value.

    PYTHON CONCEPT — dependency_overrides:
    FastAPI lets you swap dependencies for testing.
    We replace the real job store with a fresh one per test.
    """
    from fastapi.testclient import TestClient

    from investment_research_system.api.app import create_app
    from investment_research_system.api.dependencies import get_job_store, get_orchestrator_factory
    from investment_research_system.api.job_store import JobStore

    app = create_app()
    test_store = JobStore()
    app.dependency_overrides[get_job_store] = lambda: test_store
    # Override orchestrator factory so background tasks don't try real API calls.
    # Returns None → job will be marked "failed" (orchestrator unavailable).
    app.dependency_overrides[get_orchestrator_factory] = lambda: lambda: None
    return TestClient(app)


def test_health_endpoint(client):
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "tracing_enabled" in data


def test_create_research_job(client):
    """POST /research creates a job and returns job_id."""
    response = client.post(
        "/research",
        json={"query": "Should I invest in NVIDIA?"},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_create_research_job_with_options(client):
    """POST /research with focus_areas and depth."""
    response = client.post(
        "/research",
        json={
            "query": "Analyze Tesla stock",
            "focus_areas": ["financials", "risk"],
            "depth": "deep",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data


def test_get_job_status_after_create(client):
    """GET /research/{job_id} returns job status after creation.

    NOTE: FastAPI's TestClient runs background tasks synchronously,
    so by the time we poll, the background task has already run.
    With our mock (orchestrator_factory returns None), the job
    transitions to 'failed' because no orchestrator is available.
    In production, the job would be 'pending' or 'running'.
    """
    create_resp = client.post(
        "/research",
        json={"query": "Test query"},
    )
    job_id = create_resp.json()["job_id"]

    status_resp = client.get(f"/research/{job_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["job_id"] == job_id
    # Background task ran synchronously → job is now "failed" (no orchestrator)
    assert data["status"] == "failed"
    assert "not configured" in data["error"]


def test_get_job_status_not_found(client):
    """GET /research/{job_id} returns 404 for nonexistent job."""
    response = client.get("/research/fake-id-12345")
    assert response.status_code == 404


def test_create_research_invalid_body(client):
    """POST /research with missing query returns 422."""
    response = client.post("/research", json={})
    assert response.status_code == 422


# --- Tracing Tests ---

def test_tracing_init_without_key():
    """Tracing should be disabled when no API key is set."""
    import os
    from investment_research_system.observability.tracing import init_tracing, is_tracing_enabled

    # Remove key if present
    old_key = os.environ.pop("LANGSMITH_API_KEY", None)
    os.environ.pop("LANGCHAIN_TRACING_V2", None)

    result = init_tracing()
    assert result is False
    assert is_tracing_enabled() is False

    # Restore if was set
    if old_key:
        os.environ["LANGSMITH_API_KEY"] = old_key


def test_tracing_init_with_key():
    """Tracing should be enabled when API key is set."""
    import os
    from investment_research_system.observability.tracing import init_tracing, is_tracing_enabled

    # Set a fake key
    old_key = os.environ.get("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_API_KEY"] = "fake-key-for-testing"

    result = init_tracing(project_name="test-project")
    assert result is True
    assert is_tracing_enabled() is True
    assert os.environ["LANGCHAIN_PROJECT"] == "test-project"

    # Cleanup
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    if old_key:
        os.environ["LANGSMITH_API_KEY"] = old_key
    else:
        os.environ.pop("LANGSMITH_API_KEY", None)


def test_run_config_structure():
    """get_run_config returns correct structure for LangGraph."""
    from investment_research_system.observability.tracing import get_run_config

    config = get_run_config(
        query="Should I invest in NVIDIA?",
        session_id="test-session-123",
        run_name="test run",
    )

    assert "metadata" in config
    assert config["metadata"]["session_id"] == "test-session-123"
    assert "NVIDIA" in config["metadata"]["query_preview"]
    assert "tags" in config
    assert "investment-research" in config["tags"]
    assert config["run_name"] == "test run"
