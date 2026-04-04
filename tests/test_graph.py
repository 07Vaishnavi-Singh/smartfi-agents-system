"""Tests for the Neo4j graph memory layer (Knowledge Graph — Graph 1).

Tests GraphMemory without a running Neo4j instance by mocking the driver.
The mock simulates Neo4j's session/transaction API so we can verify:
- Correct Cypher queries are generated
- Profile creation, reading, updating, and formatting all work
- Temporal versioning (append-only) works correctly
- Graceful degradation when user not found

Run: uv run pytest tests/test_graph.py -v

WHY MOCK instead of a real Neo4j?
- Tests must run anywhere (CI, laptop) without Docker
- We're testing OUR code (Cypher generation, data mapping, formatting),
  not Neo4j itself. Neo4j's own test suite covers their engine.
- Integration tests with real Neo4j can be added later.

PYTHON CONCEPT — unittest.mock:
MagicMock creates objects that accept any method call and return another
MagicMock by default. You can configure return values and inspect calls.
TS equivalent: jest.fn() / sinon.stub()
Rust equivalent: mockall crate
"""

import os
import sys
from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from investment_research_system.models.user import (
    FinancialGoal,
    InvestmentType,
    UserProfile,
    UserProfileUpdate,
)


# ============================================================================
# Test fixtures — reusable setup for test functions
# ============================================================================


def _make_sample_profile() -> UserProfile:
    """Create a sample UserProfile for testing.

    PYTHON CONCEPT — helper function in tests:
    Instead of repeating the same UserProfile(...) in every test,
    extract it into a helper. Same as a factory function in TS tests.
    """
    return UserProfile(
        user_id="u-test-001",
        name="Vaiz",
        age_range="23-27",
        gender="M",
        location_city="Bangalore",
        location_country="India",
        risk_tolerance="moderate",
        monthly_budget_range="20-50k",
        budget_currency="INR",
        goals=[
            FinancialGoal(
                type="wealth_building",
                custom_label=None,
                horizon="long",
                horizon_years="5+",
                priority=1,
            ),
            FinancialGoal(
                type="save_for_x",
                custom_label="house",
                horizon="medium",
                horizon_years="1-5",
                priority=2,
            ),
        ],
        existing_investments=[InvestmentType.mutual_funds, InvestmentType.gold],
    )


class FakeNeo4jSession:
    """A fake Neo4j session that records all Cypher queries executed.

    Instead of connecting to a real database, this captures every .run()
    call so we can assert the right queries were generated.

    Also supports configurable return values for read queries via
    the `query_results` dict — map a keyword in the query to the
    result you want returned.

    PYTHON CONCEPT — context manager (__enter__/__exit__):
    Neo4j sessions are used with `with driver.session() as session:`.
    __enter__ returns self, __exit__ cleans up. This fake does the same.
    TS equivalent: implementing Symbol.dispose (using declaration)
    """

    def __init__(self, query_results: dict | None = None):
        self.queries: list[tuple[str, dict]] = []  # [(cypher, params), ...]
        self.query_results = query_results or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, query: str, **params) -> MagicMock:
        """Record the query and return a configurable mock result."""
        self.queries.append((query, params))

        result = MagicMock()

        # Check if any configured keyword matches this query
        for keyword, return_val in self.query_results.items():
            if keyword in query:
                result.single.return_value = return_val
                result.data.return_value = return_val if isinstance(return_val, list) else [return_val]
                return result

        # Default: return None for .single(), empty list for .data()
        result.single.return_value = None
        result.data.return_value = []
        return result

    def execute_write(self, func, *args, **kwargs):
        """Execute a write transaction function, passing this session as tx.

        In real Neo4j, execute_write passes a Transaction object.
        We pass `self` so the function's tx.run() calls get recorded here.
        """
        return func(self, *args, **kwargs)


class FakeNeo4jDriver:
    """A fake Neo4j driver that returns a FakeNeo4jSession."""

    def __init__(self, session: FakeNeo4jSession):
        self._session = session

    def session(self):
        return self._session

    def verify_connectivity(self):
        """Simulate a successful connectivity check."""
        pass


# ============================================================================
# UserProfile model tests
# ============================================================================


class TestUserProfileModel:
    """Tests for the Pydantic UserProfile model validation."""

    def test_valid_profile_creation(self):
        """A well-formed profile should create without errors."""
        profile = _make_sample_profile()
        assert profile.user_id == "u-test-001"
        assert profile.name == "Vaiz"
        assert len(profile.goals) == 2
        assert profile.goals[0].priority == 1

    def test_goals_min_length_enforced(self):
        """Profile must have at least 1 goal."""
        with pytest.raises(Exception):
            # Pydantic should reject an empty goals list
            UserProfile(
                user_id="u-bad",
                name="Bad",
                age_range="23-27",
                gender="M",
                location_city="X",
                location_country="Y",
                risk_tolerance="moderate",
                monthly_budget_range="0-10k",
                goals=[],  # empty — should fail
                existing_investments=[InvestmentType.none],
            )

    def test_goals_max_length_enforced(self):
        """Profile must have at most 3 goals."""
        too_many_goals = [
            FinancialGoal(type=f"goal_{i}", horizon="long", horizon_years="5+", priority=i)
            for i in range(4)  # 4 goals — over the limit
        ]
        with pytest.raises(Exception):
            UserProfile(
                user_id="u-bad",
                name="Bad",
                age_range="23-27",
                gender="M",
                location_city="X",
                location_country="Y",
                risk_tolerance="moderate",
                monthly_budget_range="0-10k",
                goals=too_many_goals,
                existing_investments=[InvestmentType.none],
            )

    def test_default_currency_is_inr(self):
        """Budget currency should default to INR if not specified."""
        profile = _make_sample_profile()
        assert profile.budget_currency == "INR"

    def test_investment_type_enum_values(self):
        """InvestmentType enum should serialize to strings."""
        assert InvestmentType.mutual_funds.value == "mutual_funds"
        assert InvestmentType.crypto.value == "crypto"
        assert InvestmentType.none.value == "none"


class TestUserProfileUpdate:
    """Tests for the partial update model."""

    def test_all_fields_optional(self):
        """UserProfileUpdate with no fields should be valid."""
        update = UserProfileUpdate()
        assert update.name is None
        assert update.age_range is None
        assert update.goals is None

    def test_partial_update(self):
        """Only set the fields that changed."""
        update = UserProfileUpdate(risk_tolerance="aggressive", location_city="Mumbai")
        assert update.risk_tolerance == "aggressive"
        assert update.location_city == "Mumbai"
        assert update.name is None  # not changed


# ============================================================================
# GraphMemory tests (mocked Neo4j driver)
# ============================================================================


class TestGraphMemoryCreate:
    """Tests for GraphMemory.create_user_profile()."""

    def test_create_profile_runs_correct_cypher(self):
        """Creating a profile should generate CREATE/MERGE Cypher queries."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession(query_results={
            # The user-exists check should return None (user doesn't exist)
            "MATCH (u:User": None,
            # Node count query returns a count
            "count(DISTINCT n)": {"node_count": 8},
        })
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        profile = _make_sample_profile()
        result = graph.create_user_profile(profile)

        assert result["user_id"] == "u-test-001"
        assert result["status"] == "created"

        # Verify Cypher queries were generated
        all_queries = " ".join(q for q, _ in session.queries)
        # Should have MERGE for shared nodes
        assert "MERGE" in all_queries
        # Should create User node
        assert "CREATE (u:User" in all_queries
        # Should create goal nodes (2 goals)
        assert all_queries.count("CREATE (goal:Goal") == 2
        # Should create investment edges (2 investments)
        assert all_queries.count("MERGE (a:AssetClass") == 2

    def test_create_duplicate_user_raises_error(self):
        """Creating a profile for an existing user_id should raise ValueError."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession(query_results={
            # User already exists
            "MATCH (u:User": {"id": "u-exists"},
        })
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        profile = _make_sample_profile()
        profile.user_id = "u-exists"

        with pytest.raises(ValueError, match="already exists"):
            graph.create_user_profile(profile)


class TestGraphMemoryRead:
    """Tests for GraphMemory.get_user_profile() and format_profile_for_agents()."""

    def _make_graph_with_profile(self):
        """Helper: create a GraphMemory with a mocked user profile in Neo4j."""
        from investment_research_system.memory.graph import GraphMemory

        # Simulate Neo4j returning a full profile
        session = FakeNeo4jSession(query_results={
            # User exists
            "RETURN u.name": {"name": "Vaiz", "created_at": "2026-03-24"},
            # Active relationships
            "type(r) AS rel_type": [
                {"rel_type": "AGE_RANGE", "props": {"range": "23-27"}, "priority": None},
                {"rel_type": "GENDER", "props": {"value": "M"}, "priority": None},
                {"rel_type": "LOCATED_IN", "props": {"city": "Bangalore", "country": "India"}, "priority": None},
                {"rel_type": "RISK_TOLERANCE", "props": {"level": "moderate"}, "priority": None},
                {"rel_type": "BUDGET", "props": {"range": "20-50k", "currency": "INR"}, "priority": None},
                {"rel_type": "HAS_GOAL", "props": {"type": "wealth_building", "custom_label": None}, "priority": 1},
                {"rel_type": "INVESTS_IN", "props": {"type": "mutual_funds"}, "priority": None},
                {"rel_type": "INVESTS_IN", "props": {"type": "gold"}, "priority": None},
            ],
            # Goal horizons
            "HORIZON": [
                {"goal_type": "wealth_building", "horizon": "long", "years": "5+"},
            ],
        })
        driver = FakeNeo4jDriver(session)
        return GraphMemory(driver)

    def test_get_profile_returns_structured_dict(self):
        """get_user_profile should reconstruct a profile dict from graph data."""
        graph = self._make_graph_with_profile()
        profile = graph.get_user_profile("u-test-001")

        assert profile is not None
        assert profile["user_id"] == "u-test-001"
        assert profile["name"] == "Vaiz"
        assert profile["age_range"] == "23-27"
        assert profile["gender"] == "M"
        assert profile["location_city"] == "Bangalore"
        assert profile["location_country"] == "India"
        assert profile["risk_tolerance"] == "moderate"
        assert profile["monthly_budget_range"] == "20-50k"
        assert profile["budget_currency"] == "INR"
        assert "mutual_funds" in profile["existing_investments"]
        assert "gold" in profile["existing_investments"]
        assert len(profile["goals"]) == 1
        assert profile["goals"][0]["type"] == "wealth_building"

    def test_get_profile_returns_none_for_missing_user(self):
        """get_user_profile should return None if user doesn't exist."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession()  # no query_results → all returns None
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        assert graph.get_user_profile("nonexistent") is None

    def test_format_profile_contains_all_fields(self):
        """format_profile_for_agents should produce a readable text block."""
        graph = self._make_graph_with_profile()
        text = graph.format_profile_for_agents("u-test-001")

        assert "[USER PROFILE]" in text
        assert "23-27" in text
        assert "Bangalore" in text
        assert "India" in text
        assert "moderate" in text
        assert "20-50k" in text
        assert "INR" in text
        assert "mutual_funds" in text
        assert "gold" in text
        assert "wealth_building" in text

    def test_format_profile_returns_empty_for_missing_user(self):
        """format_profile_for_agents should return '' if user not found."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession()
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        assert graph.format_profile_for_agents("nonexistent") == ""

    def test_format_profile_lgbtq_includes_context_note(self):
        """LGBTQ+ users should get an additional financial context note."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession(query_results={
            "RETURN u.name": {"name": "Alex", "created_at": "2026-03-24"},
            "type(r) AS rel_type": [
                {"rel_type": "AGE_RANGE", "props": {"range": "23-27"}, "priority": None},
                {"rel_type": "GENDER", "props": {"value": "LGBTQ+"}, "priority": None},
                {"rel_type": "LOCATED_IN", "props": {"city": "Mumbai", "country": "India"}, "priority": None},
                {"rel_type": "RISK_TOLERANCE", "props": {"level": "moderate"}, "priority": None},
                {"rel_type": "BUDGET", "props": {"range": "10-20k", "currency": "INR"}, "priority": None},
                {"rel_type": "HAS_GOAL", "props": {"type": "wealth_building", "custom_label": None}, "priority": 1},
            ],
            "HORIZON": [
                {"goal_type": "wealth_building", "horizon": "long", "years": "5+"},
            ],
        })
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        text = graph.format_profile_for_agents("u-lgbtq")
        assert "[FINANCIAL CONTEXT NOTE]" in text
        assert "financial independence" in text
        assert "emergency fund" in text


class TestGraphMemoryUpdate:
    """Tests for GraphMemory.update_user_profile()."""

    def test_update_closes_old_edge_and_creates_new(self):
        """Updating risk tolerance should close the old edge and open a new one."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession(query_results={
            # User exists
            "RETURN u.id": {"u.id": "u-test-001"},
        })
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        update = UserProfileUpdate(risk_tolerance="aggressive")
        result = graph.update_user_profile("u-test-001", update)

        assert result["status"] == "updated"
        assert result["fields_updated"] == 1

        # Check that queries include both closing old and creating new
        all_queries = " ".join(q for q, _ in session.queries)
        # Should SET r.to to close old edge
        assert "SET r.to" in all_queries
        # Should MERGE new RiskProfile node
        assert "RiskProfile" in all_queries

    def test_update_nonexistent_user_raises_error(self):
        """Updating a user that doesn't exist should raise ValueError."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession()  # user not found
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        update = UserProfileUpdate(name="New Name")
        with pytest.raises(ValueError, match="not found"):
            graph.update_user_profile("nonexistent", update)

    def test_update_name_only_modifies_user_node(self):
        """Updating name should SET on User node directly (not close/reopen)."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession(query_results={
            "RETURN u.id": {"u.id": "u-test-001"},
        })
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        update = UserProfileUpdate(name="New Name")
        result = graph.update_user_profile("u-test-001", update)

        assert result["fields_updated"] == 1
        # Name update should use SET directly on User node
        all_queries = " ".join(q for q, _ in session.queries)
        assert "SET u.name" in all_queries

    def test_update_goals_replaces_all_goals(self):
        """Updating goals should close all old goals and create fresh ones."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession(query_results={
            "RETURN u.id": {"u.id": "u-test-001"},
        })
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        new_goals = [
            FinancialGoal(type="retirement", horizon="long", horizon_years="5+", priority=1),
        ]
        update = UserProfileUpdate(goals=new_goals)
        result = graph.update_user_profile("u-test-001", update)

        assert result["fields_updated"] == 1
        # Should close old goals (SET r.to) and create new ones
        all_queries = " ".join(q for q, _ in session.queries)
        assert "HAS_GOAL" in all_queries
        assert "CREATE (goal:Goal" in all_queries

    def test_update_multiple_fields(self):
        """Updating multiple fields at once should count each one."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession(query_results={
            "RETURN u.id": {"u.id": "u-test-001"},
        })
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        update = UserProfileUpdate(
            name="Updated Name",
            risk_tolerance="aggressive",
            monthly_budget_range="50k+",
        )
        result = graph.update_user_profile("u-test-001", update)

        # name + risk_tolerance + budget = 3 fields
        assert result["fields_updated"] == 3


class TestGraphMemoryHealth:
    """Tests for GraphMemory.ping() and _ensure_indexes()."""

    def test_ping_returns_true_when_connected(self):
        """ping() should return True when Neo4j is reachable."""
        from investment_research_system.memory.graph import GraphMemory

        driver = FakeNeo4jDriver(FakeNeo4jSession())
        graph = GraphMemory(driver)
        assert graph.ping() is True

    def test_ping_returns_false_on_connection_error(self):
        """ping() should return False when Neo4j is unreachable."""
        from investment_research_system.memory.graph import GraphMemory

        driver = MagicMock()
        driver.verify_connectivity.side_effect = Exception("Connection refused")
        graph = GraphMemory(driver)
        assert graph.ping() is False

    def test_ensure_indexes_runs_without_error(self):
        """_ensure_indexes() should execute CREATE INDEX query."""
        from investment_research_system.memory.graph import GraphMemory

        session = FakeNeo4jSession()
        driver = FakeNeo4jDriver(session)
        graph = GraphMemory(driver)

        graph._ensure_indexes()

        # Should have run the CREATE INDEX query
        assert len(session.queries) == 1
        assert "CREATE INDEX" in session.queries[0][0]


# ============================================================================
# MemoryManager integration with GraphMemory
# ============================================================================


class TestMemoryManagerGraphIntegration:
    """Tests that MemoryManager correctly holds and reports GraphMemory."""

    @pytest.mark.asyncio
    async def test_health_check_includes_neo4j(self):
        """health_check should report neo4j status."""
        from unittest.mock import AsyncMock

        from investment_research_system.memory.graph import GraphMemory
        from investment_research_system.memory.manager import MemoryManager

        short_term = AsyncMock()
        short_term.ping = AsyncMock(return_value=True)
        long_term = MagicMock()
        long_term.ping = MagicMock(return_value=True)
        semantic = MagicMock()

        # Create a graph mock that reports healthy
        graph = MagicMock(spec=GraphMemory)
        graph.ping = MagicMock(return_value=True)

        manager = MemoryManager(
            short_term=short_term,
            long_term=long_term,
            semantic=semantic,
            episodic=None,
            graph=graph,
        )

        health = await manager.health_check()
        assert health["neo4j"] is True

    @pytest.mark.asyncio
    async def test_health_check_neo4j_false_when_not_configured(self):
        """health_check should report neo4j=False when graph=None."""
        from unittest.mock import AsyncMock

        from investment_research_system.memory.manager import MemoryManager

        short_term = AsyncMock()
        short_term.ping = AsyncMock(return_value=True)
        long_term = MagicMock()
        long_term.ping = MagicMock(return_value=True)
        semantic = MagicMock()

        manager = MemoryManager(
            short_term=short_term,
            long_term=long_term,
            semantic=semantic,
            graph=None,  # Neo4j not configured
        )

        health = await manager.health_check()
        assert health["neo4j"] is False
