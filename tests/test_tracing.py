"""Test that LangSmith tracing and monitoring metrics work end-to-end.

Run: PYTHONPATH=src uv run pytest tests/test_tracing.py -v -s

Uses dummy data — no real API calls needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_tracing_init_sets_env_vars():
    """init_tracing() should set all required env vars for LangChain."""
    from investment_research_system.observability.tracing import init_tracing

    # Save originals
    originals = {
        k: os.environ.pop(k, None)
        for k in ["LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
    }

    # Set fake key and init
    os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_fake_test_key"
    result = init_tracing(project_name="test-monitoring")

    assert result is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "lsv2_pt_fake_test_key"
    assert os.environ["LANGCHAIN_PROJECT"] == "test-monitoring"

    # Cleanup
    for k, v in originals.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_tracing_disabled_without_key():
    """Without LANGSMITH_API_KEY, tracing should not activate."""
    from investment_research_system.observability.tracing import init_tracing, is_tracing_enabled

    originals = {
        k: os.environ.pop(k, None)
        for k in ["LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"]
    }

    result = init_tracing()
    assert result is False
    assert is_tracing_enabled() is False

    for k, v in originals.items():
        if v:
            os.environ[k] = v


def test_run_config_metadata_tagging():
    """Run config should contain metadata that LangSmith uses for filtering."""
    from investment_research_system.observability.tracing import get_run_config

    config = get_run_config(
        query="Should I invest in NVIDIA stock given current AI boom?",
        session_id="session-abc-123",
        run_name="research: NVIDIA analysis",
    )

    # Verify structure matches what LangGraph expects
    assert "metadata" in config
    assert "tags" in config
    assert "run_name" in config

    # Metadata should have query preview (truncated) and session
    meta = config["metadata"]
    assert meta["session_id"] == "session-abc-123"
    assert "NVIDIA" in meta["query_preview"]
    assert len(meta["query_preview"]) <= 100  # truncated

    # Tags for filtering in dashboard
    assert "investment-research" in config["tags"]

    # Run name for display
    assert config["run_name"] == "research: NVIDIA analysis"
    print(f"\n--- Run Config ---\n{config}")


def test_run_config_without_run_name():
    """Run config without run_name should omit the field."""
    from investment_research_system.observability.tracing import get_run_config

    config = get_run_config(query="Test", session_id="s1")
    assert "run_name" not in config
    assert "metadata" in config


def test_orchestrator_passes_config_to_graph():
    """Verify the orchestrator builds and passes tracing config to LangGraph.

    This test mocks the graph to capture what config gets passed,
    ensuring our tracing metadata actually reaches LangGraph.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from investment_research_system.models.schemas import ResearchQuery
    from investment_research_system.orchestrator.graph import ResearchOrchestrator

    # Create a mock agent
    mock_agent = MagicMock()
    mock_agent.name = "researcher"

    # Create orchestrator
    orchestrator = ResearchOrchestrator(agents=[mock_agent], max_retries=0)

    # Capture the config passed to ainvoke
    captured_config = {}

    async def mock_ainvoke(state, config=None):
        captured_config["config"] = config
        # Return a state with a minimal report
        from investment_research_system.models.schemas import ResearchReport
        return {
            **state,
            "report": ResearchReport(
                id="test-id",
                query=state["research_query"],
                summary="Test summary",
                agent_responses=[],
                total_cost_usd=0,
                total_tokens=0,
                processing_time_seconds=0.1,
            ),
        }

    # Patch the compiled graph's ainvoke
    orchestrator.graph = MagicMock()
    orchestrator.graph.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    query = ResearchQuery(query="Should I invest in NVIDIA?")
    asyncio.run(orchestrator.run(query))

    # Verify config was passed
    assert "config" in captured_config
    config = captured_config["config"]
    assert config is not None
    assert "metadata" in config
    assert "NVIDIA" in config["metadata"]["query_preview"]
    assert "tags" in config
    assert config["run_name"].startswith("research:")

    print(f"\n--- Config passed to LangGraph ---\n{config}")


def test_health_endpoint_shows_tracing_status():
    """Health endpoint should report whether tracing is enabled."""
    from fastapi.testclient import TestClient

    from investment_research_system.api.app import create_app
    from investment_research_system.api.dependencies import get_job_store, get_orchestrator_factory
    from investment_research_system.api.job_store import JobStore

    app = create_app()
    app.dependency_overrides[get_job_store] = lambda: JobStore()
    app.dependency_overrides[get_orchestrator_factory] = lambda: lambda: None

    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert "tracing_enabled" in data
    assert isinstance(data["tracing_enabled"], bool)

    print(f"\n--- Health Response ---\n{data}")


def test_full_monitoring_flow_with_dummy_data():
    """End-to-end: init tracing → create job → verify metrics are trackable.

    Simulates what happens in production:
    1. App starts, tracing initializes
    2. Job is created via API
    3. Background task runs (mocked)
    4. Job status shows completion with metrics

    This verifies the full monitoring pipeline is wired correctly.
    """
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock

    from investment_research_system.api.app import create_app
    from investment_research_system.api.dependencies import get_job_store, get_orchestrator_factory
    from investment_research_system.api.job_store import JobStore
    from investment_research_system.models.schemas import AgentResponse, ResearchReport, ResearchQuery

    # Setup: create app with a mock orchestrator that returns dummy data
    app = create_app()
    test_store = JobStore()

    # Build a realistic dummy report
    dummy_report = ResearchReport(
        id="report-001",
        query=ResearchQuery(query="Should I invest in NVIDIA?"),
        summary="NVIDIA shows strong growth driven by AI demand.",
        agent_responses=[
            AgentResponse(
                agent_name="researcher",
                content="NVIDIA revenue grew 94% YoY to $35.1B.",
                confidence=0.85,
                tokens_used=512,
                cost_usd=0.008,
                latency_ms=2100,
            ),
            AgentResponse(
                agent_name="analyst",
                content="P/E ratio of 52 is high but justified by growth.",
                confidence=0.80,
                tokens_used=480,
                cost_usd=0.007,
                latency_ms=1900,
            ),
            AgentResponse(
                agent_name="risk_assessor",
                content="Export controls to China pose 25% revenue risk.",
                confidence=0.75,
                tokens_used=450,
                cost_usd=0.007,
                latency_ms=2000,
            ),
            AgentResponse(
                agent_name="sentiment",
                content="Market sentiment is strongly bullish, score +0.7.",
                confidence=0.82,
                tokens_used=400,
                cost_usd=0.006,
                latency_ms=1800,
            ),
        ],
        total_cost_usd=0.028,
        total_tokens=1842,
        processing_time_seconds=4.5,
    )

    # Mock orchestrator that returns the dummy report
    mock_orchestrator = AsyncMock()
    mock_orchestrator.run = AsyncMock(return_value=dummy_report)

    app.dependency_overrides[get_job_store] = lambda: test_store
    app.dependency_overrides[get_orchestrator_factory] = lambda: lambda: mock_orchestrator

    client = TestClient(app)

    # Step 1: Submit research job
    create_resp = client.post(
        "/research",
        json={"query": "Should I invest in NVIDIA?"},
    )
    assert create_resp.status_code == 202
    job_id = create_resp.json()["job_id"]
    print(f"\n--- Job Created ---\njob_id: {job_id}")

    # Step 2: Poll for results (TestClient runs background task synchronously)
    status_resp = client.get(f"/research/{job_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()

    # Step 3: Verify job completed with metrics
    assert data["status"] == "completed"
    assert data["report"] is not None

    report = data["report"]
    assert report["total_tokens"] == 1842
    assert report["total_cost_usd"] == 0.028
    assert report["processing_time_seconds"] == 4.5
    assert len(report["agent_responses"]) == 4

    # Verify each agent's metrics
    agents_by_name = {r["agent_name"]: r for r in report["agent_responses"]}
    assert agents_by_name["researcher"]["tokens_used"] == 512
    assert agents_by_name["analyst"]["confidence"] == 0.80
    assert agents_by_name["risk_assessor"]["latency_ms"] == 2000
    assert agents_by_name["sentiment"]["cost_usd"] == 0.006

    print(f"\n--- Job Completed ---")
    print(f"Status: {data['status']}")
    print(f"Agents: {len(report['agent_responses'])}")
    print(f"Total tokens: {report['total_tokens']}")
    print(f"Total cost: ${report['total_cost_usd']}")
    print(f"Processing time: {report['processing_time_seconds']}s")
    print(f"\nPer-agent metrics:")
    for agent in report["agent_responses"]:
        print(f"  [{agent['agent_name']}] tokens={agent['tokens_used']}, "
              f"confidence={agent['confidence']}, "
              f"latency={agent['latency_ms']}ms, "
              f"cost=${agent['cost_usd']}")
