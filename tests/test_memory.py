"""Tests for the memory layer.

Uses SQLite in-memory for episodic (no Docker needed).
Uses mocks for Redis/Qdrant/Mem0 so tests run anywhere.

Run: uv run pytest tests/test_memory.py -v
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================================
# database.py + episodic.py tests (SQLite in-memory — no Docker needed)
# ============================================================================


@pytest.fixture
def db_session_factory():
    """Create an in-memory SQLite database for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from investment_research_system.memory.database import Base, create_tables

    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)
    factory = sessionmaker(bind=engine)
    return factory


@pytest.fixture
def episodic(db_session_factory):
    from investment_research_system.memory.episodic import EpisodicMemory

    return EpisodicMemory(session_factory=db_session_factory)


def test_create_episode(episodic):
    """Create an episode and verify it exists."""
    ep = episodic.create_episode("ep-001", "Should I invest in NVIDIA?")
    assert ep.id == "ep-001"
    assert ep.query == "Should I invest in NVIDIA?"
    assert ep.status == "started"


def test_complete_episode(episodic):
    """Complete an episode and check stats are saved."""
    episodic.create_episode("ep-002", "Tesla analysis")
    episodic.complete_episode("ep-002", total_tokens=5000, total_cost_usd=0.05)

    ep = episodic.get_episode("ep-002")
    assert ep.status == "completed"
    assert ep.total_tokens == 5000
    assert ep.total_cost_usd == 0.05
    assert ep.completed_at is not None


def test_fail_episode(episodic):
    """Fail an episode and verify status."""
    episodic.create_episode("ep-003", "Bad query")
    episodic.fail_episode("ep-003", error="API timeout")

    ep = episodic.get_episode("ep-003")
    assert ep.status == "failed"


def test_log_decision(episodic):
    """Log an agent decision and retrieve it."""
    episodic.create_episode("ep-004", "AMD vs Intel")
    decision = episodic.log_decision(
        episode_id="ep-004",
        agent_name="researcher",
        action="web_search",
        result_summary="Found 5 sources about AMD",
        confidence=0.8,
        tokens_used=1200,
        cost_usd=0.01,
    )
    assert decision.agent_name == "researcher"
    assert decision.confidence == 0.8

    decisions = episodic.get_decisions("ep-004")
    assert len(decisions) == 1
    assert decisions[0].action == "web_search"


def test_log_multiple_decisions(episodic):
    """Multiple agents log decisions for the same episode."""
    episodic.create_episode("ep-005", "NVIDIA analysis")
    episodic.log_decision("ep-005", "researcher", "search", "found data")
    episodic.log_decision("ep-005", "analyst", "analyze", "P/E is high")
    episodic.log_decision("ep-005", "risk_assessor", "assess", "export risk noted")

    decisions = episodic.get_decisions("ep-005")
    assert len(decisions) == 3
    agent_names = {d.agent_name for d in decisions}
    assert agent_names == {"researcher", "analyst", "risk_assessor"}


def test_store_knowledge(episodic):
    """Store and retrieve knowledge graph triples."""
    episodic.create_episode("ep-006", "GPU market")
    episodic.store_knowledge(
        episode_id="ep-006",
        subject="NVIDIA",
        predicate="competes_with",
        object_="AMD",
        source_agent="researcher",
        confidence=0.9,
    )
    episodic.store_knowledge(
        episode_id="ep-006",
        subject="NVIDIA",
        predicate="manufactures",
        object_="H100",
        source_agent="researcher",
    )

    entries = episodic.get_knowledge(subject="NVIDIA")
    assert len(entries) == 2
    predicates = {e.predicate for e in entries}
    assert "competes_with" in predicates
    assert "manufactures" in predicates


def test_search_episodes(episodic):
    """Search episodes by query text."""
    episodic.create_episode("ep-010", "Should I invest in NVIDIA?")
    episodic.create_episode("ep-011", "Is Tesla overvalued?")
    episodic.create_episode("ep-012", "Compare NVIDIA and AMD")

    results = episodic.search_episodes("NVIDIA")
    assert len(results) == 2
    queries = {r.query for r in results}
    assert "Should I invest in NVIDIA?" in queries
    assert "Compare NVIDIA and AMD" in queries


def test_get_nonexistent_episode(episodic):
    """Getting a missing episode returns None."""
    assert episodic.get_episode("does-not-exist") is None


def test_episodic_ping(episodic):
    """Ping should return True when DB is reachable."""
    assert episodic.ping() is True


# ============================================================================
# Database model tests
# ============================================================================


def test_tables_created(db_session_factory):
    """All 3 tables should exist after create_tables."""
    from investment_research_system.memory.database import Base

    table_names = set(Base.metadata.tables.keys())
    assert "episodes" in table_names
    assert "agent_decisions" in table_names
    assert "knowledge_entries" in table_names


def test_cascade_delete(db_session_factory):
    """Deleting an episode should cascade-delete its decisions and knowledge."""
    from investment_research_system.memory.database import AgentDecision, Episode, KnowledgeEntry

    with db_session_factory() as session:
        ep = Episode(id="ep-cascade", query="test cascade")
        session.add(ep)
        session.commit()

        session.add(AgentDecision(episode_id="ep-cascade", agent_name="test", action="test"))
        session.add(KnowledgeEntry(
            episode_id="ep-cascade", subject="A", predicate="rel", object_="B"
        ))
        session.commit()

        # Delete the episode
        session.delete(ep)
        session.commit()

        # Children should be gone
        assert session.query(AgentDecision).filter_by(episode_id="ep-cascade").count() == 0
        assert session.query(KnowledgeEntry).filter_by(episode_id="ep-cascade").count() == 0


# ============================================================================
# MemoryManager tests (mocked backends)
# ============================================================================


@pytest.fixture
def mock_memory_manager(tmp_path):
    """MemoryManager with mocked Redis/Qdrant/Mem0 but real episodic (file-based SQLite)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from investment_research_system.memory.database import Base, create_tables
    from investment_research_system.memory.episodic import EpisodicMemory
    from investment_research_system.memory.manager import MemoryManager

    # Use file-based SQLite with StaticPool so all threads share the same connection
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_tables(engine)
    session_factory = sessionmaker(bind=engine)
    real_episodic = EpisodicMemory(session_factory=session_factory)

    short_term = AsyncMock()
    short_term.store = AsyncMock()
    short_term.recall = AsyncMock(return_value=None)
    short_term.update_agent_status = AsyncMock()
    short_term.get_agent_statuses = AsyncMock(return_value={})
    short_term.delete_session = AsyncMock()
    short_term.ping = AsyncMock(return_value=True)

    long_term = MagicMock()
    long_term.search = MagicMock(return_value=[
        {"text": "NVIDIA revenue up 20%", "score": 0.9, "agent_name": "researcher"},
    ])
    long_term.store = MagicMock(return_value="point-123")
    long_term.ping = MagicMock(return_value=True)

    semantic = MagicMock()
    semantic.search = MagicMock(return_value=[
        {"memory": "NVIDIA P/E is 58", "score": 0.85},
    ])
    semantic.store = MagicMock(return_value={"id": "mem-123"})

    return MemoryManager(
        short_term=short_term,
        long_term=long_term,
        semantic=semantic,
        episodic=real_episodic,
    )


@pytest.mark.asyncio
async def test_parallel_search(mock_memory_manager):
    """parallel_search should return results from both Qdrant and Mem0."""
    results = await mock_memory_manager.parallel_search("NVIDIA investment")

    assert "long_term" in results
    assert "semantic" in results
    assert len(results["long_term"]) == 1
    assert results["long_term"][0]["text"] == "NVIDIA revenue up 20%"
    assert len(results["semantic"]) == 1


@pytest.mark.asyncio
async def test_create_session_writes_to_redis_and_postgres(mock_memory_manager):
    """create_session should write to both Redis and Postgres."""
    await mock_memory_manager.create_session("sess-001", "Test query")

    # Redis was called
    mock_memory_manager.short_term.store.assert_called_once()

    # Postgres episode was created
    ep = mock_memory_manager.episodic.get_episode("sess-001")
    assert ep is not None
    assert ep.query == "Test query"
    assert ep.status == "started"


@pytest.mark.asyncio
async def test_store_research_logs_to_all_backends(mock_memory_manager):
    """store_research should write to Qdrant, Mem0, Redis, and Postgres."""
    # Need an episode first for the decision FK
    mock_memory_manager.episodic.create_episode("sess-002", "Research test")

    await mock_memory_manager.store_research(
        content="NVIDIA datacenter revenue grew 25%",
        agent_name="researcher",
        session_id="sess-002",
        metadata={"query_context": "NVIDIA"},
    )

    # Qdrant was called
    mock_memory_manager.long_term.store.assert_called_once()
    # Mem0 was called
    mock_memory_manager.semantic.store.assert_called_once()
    # Redis was called (session result update)
    assert mock_memory_manager.short_term.store.call_count >= 1
    # Postgres decision was logged
    decisions = mock_memory_manager.episodic.get_decisions("sess-002")
    assert len(decisions) == 1
    assert decisions[0].agent_name == "researcher"


@pytest.mark.asyncio
async def test_end_session_cleans_up(mock_memory_manager):
    """end_session should clean Redis and mark Postgres episode completed."""
    mock_memory_manager.episodic.create_episode("sess-003", "Cleanup test")

    await mock_memory_manager.end_session("sess-003", total_tokens=3000, total_cost_usd=0.03)

    # Redis cleaned
    mock_memory_manager.short_term.delete_session.assert_called_once_with("sess-003")
    # Postgres completed
    ep = mock_memory_manager.episodic.get_episode("sess-003")
    assert ep.status == "completed"
    assert ep.total_tokens == 3000


@pytest.mark.asyncio
async def test_health_check(mock_memory_manager):
    """health_check should report status of all backends."""
    health = await mock_memory_manager.health_check()

    assert health["redis"] is True
    assert health["qdrant"] is True
    assert health["postgres"] is True


@pytest.mark.asyncio
async def test_update_agent_status(mock_memory_manager):
    """update_agent_status should forward to Redis."""
    await mock_memory_manager.update_agent_status("sess-x", "researcher", "running")
    mock_memory_manager.short_term.update_agent_status.assert_called_once_with(
        "sess-x", "researcher", "running"
    )


# ============================================================================
# MemoryManager without episodic (graceful degradation)
# ============================================================================


@pytest.mark.asyncio
async def test_manager_works_without_episodic():
    """MemoryManager should work fine when episodic=None."""
    from investment_research_system.memory.manager import MemoryManager

    short_term = AsyncMock()
    short_term.store = AsyncMock()
    short_term.delete_session = AsyncMock()
    short_term.ping = AsyncMock(return_value=True)

    long_term = MagicMock()
    long_term.search = MagicMock(return_value=[])
    long_term.store = MagicMock()
    long_term.ping = MagicMock(return_value=True)

    semantic = MagicMock()
    semantic.search = MagicMock(return_value=[])
    semantic.store = MagicMock()

    manager = MemoryManager(
        short_term=short_term,
        long_term=long_term,
        semantic=semantic,
        episodic=None,  # No Postgres
    )

    # These should all work without errors
    await manager.create_session("s1", "test")
    await manager.store_research("content", "researcher", "s1")
    await manager.end_session("s1")
    results = await manager.parallel_search("test")
    assert "long_term" in results

    health = await manager.health_check()
    assert health["postgres"] is False
