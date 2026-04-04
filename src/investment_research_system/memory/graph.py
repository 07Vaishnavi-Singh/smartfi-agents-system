"""Neo4j-backed graph memory for user profiles (Knowledge Graph — Graph 1).

Stores user profile data as a property graph: nodes (User, Goal, Location)
connected by edges (HAS_GOAL, LOCATED_IN) with temporal metadata.

WHY A GRAPH DATABASE?
- Relationships are first-class citizens (not JOINs on flat tables)
- Queries like "get everything about this user" are single traversals
- Same Neo4j instance will later hold asset relationships (Graph 3),
  preferences (Graph 2), and source credibility (Graph 4)
- Temporal edges (from/to dates) give us full history for free

ARCHITECTURE DECISION — SYNC DRIVER:
Neo4j has both sync and async Python drivers. We use the SYNC driver here,
matching the pattern in LongTermMemory (Qdrant) and SemanticMemory (Mem0).
The MemoryManager wraps sync calls with asyncio.to_thread() so they don't
block the FastAPI event loop. This keeps all memory backends consistent.

TS equivalent: This is like a Repository class that wraps a database client
and exposes domain-specific methods (createUser, getUserProfile, etc.).
Rust equivalent: A struct with a connection pool that implements domain traits.
"""

import logging
from datetime import date

from neo4j import Driver

from investment_research_system.models.user import (
    UserProfile,
    UserProfileUpdate,
)

logger = logging.getLogger(__name__)


class GraphMemory:
    """Neo4j-backed graph memory for user profiles.

    Uses SYNCHRONOUS Neo4j driver — wrapped with asyncio.to_thread()
    in the orchestrator/MemoryManager layer (same pattern as Qdrant/Mem0).

    PYTHON CONCEPT — composition pattern:
    We hold a reference to the Neo4j driver (self.driver) rather than
    inheriting from it. This is "composition over inheritance" —
    same pattern used in SemanticMemory (wraps Mem0 client).
    TS: constructor(private driver: neo4j.Driver)
    Rust: struct GraphMemory { driver: Driver }
    """

    def __init__(self, driver: Driver):
        """Initialize with a Neo4j driver instance.

        The driver is created in dependencies.py and passed here.
        We don't create connections ourselves — dependency injection.

        Args:
            driver: A neo4j.Driver instance (sync). Created via
                    neo4j.GraphDatabase.driver(uri, auth=(user, pass)).
        """
        self.driver = driver

    # =========================================================================
    # SETUP — called once at startup
    # =========================================================================

    def _ensure_indexes(self) -> None:
        """Create indexes for fast lookups. Called once at app startup.

        WHY INDEXES MATTER:
        Without an index on User.id, Neo4j scans ALL User nodes to find one.
        With an index, it's O(log n) — like adding an index to a SQL column.

        PYTHON CONCEPT — context manager (with statement):
        `with self.driver.session() as session:` is like try/finally in TS:
        it automatically closes the session when the block exits, even on error.
        Rust equivalent: Drop trait (RAII pattern).
        """
        with self.driver.session() as session:
            # CREATE INDEX is idempotent — safe to run on every startup.
            # IF NOT EXISTS prevents errors if index already exists.
            session.run("CREATE INDEX user_id_idx IF NOT EXISTS FOR (u:User) ON (u.id)")
            logger.info("[graph] Neo4j indexes ensured")

    # =========================================================================
    # WRITE — create and update user profiles
    # =========================================================================

    def create_user_profile(self, profile: UserProfile) -> dict:
        """Store a new user's onboarding data as a graph.

        Creates the User node and all related nodes/edges in a SINGLE
        transaction. If anything fails, everything rolls back — atomicity.

        CYPHER CONCEPT — MERGE vs CREATE:
        - MERGE = "find or create" — idempotent, won't duplicate.
          Use for SHARED nodes (AgeRange, Location) that multiple users reference.
        - CREATE = always creates new. Use for user-specific edges.

        Think of MERGE like SQL's INSERT ... ON CONFLICT DO NOTHING,
        or Redis's SETNX (set if not exists).

        Args:
            profile: Validated UserProfile from the API layer.

        Returns:
            dict with user_id, status, and node_count.

        Raises:
            ValueError: If user_id already exists (409 at API layer).
        """
        with self.driver.session() as session:
            # Check if user already exists — fail fast before creating anything
            exists = session.run(
                "MATCH (u:User {id: $uid}) RETURN u.id AS id",
                uid=profile.user_id,
            ).single()

            if exists:
                raise ValueError(f"User {profile.user_id} already exists")

            # Run all writes in a single transaction for atomicity.
            # PYTHON CONCEPT — execute_write():
            # Neo4j's "managed transaction" — handles retries on transient errors
            # (network blips, leader election). You pass a function that receives
            # a transaction (tx), and Neo4j manages commit/rollback.
            result = session.execute_write(
                self._create_profile_tx, profile
            )
            logger.info("[graph] User profile created: %s", profile.user_id)
            return result

    @staticmethod
    def _create_profile_tx(tx, profile: UserProfile) -> dict:
        """Transaction function that creates all nodes and edges.

        PYTHON CONCEPT — @staticmethod:
        A method that doesn't need `self`. It's just a function that lives
        in the class namespace. We use it here because Neo4j's execute_write()
        passes (tx, *args) — there's no `self` in the call chain.
        TS equivalent: static method on a class.
        Rust equivalent: an associated function (no &self parameter).

        CYPHER WALKTHROUGH:
        The query below does everything in one shot:
        1. Creates the User node
        2. MERGE shared nodes (AgeRange, Gender, etc.) — reused across users
        3. CREATE edges from User to each node — each user gets their own edges
        4. Edges have {from: date(), to: null} for temporal versioning
           - to: null means "currently active"
           - When the user updates, we set to = today and create a new edge
        """
        today = date.today().isoformat()

        # --- Step 1: Create User node + basic attribute nodes ---
        # MERGE on shared nodes (AgeRange, Location, etc.) prevents duplicates.
        # Multiple users can be "26-35" — they share the same AgeRange node.
        tx.run(
            """
            CREATE (u:User {id: $uid, name: $name, created_at: $today})

            MERGE (age:AgeRange {range: $age_range})
            CREATE (u)-[:AGE_RANGE {from: $today, to: null}]->(age)

            MERGE (g:Gender {value: $gender})
            CREATE (u)-[:GENDER]->(g)

            MERGE (loc:Location {city: $city, country: $country})
            CREATE (u)-[:LOCATED_IN {from: $today, to: null}]->(loc)

            MERGE (risk:RiskProfile {level: $risk})
            CREATE (u)-[:RISK_TOLERANCE {from: $today, to: null}]->(risk)

            MERGE (budget:Budget {range: $budget_range, currency: $currency})
            CREATE (u)-[:BUDGET {from: $today, to: null}]->(budget)
            """,
            uid=profile.user_id,
            name=profile.name,
            today=today,
            age_range=profile.age_range,
            gender=profile.gender,
            city=profile.location_city,
            country=profile.location_country,
            risk=profile.risk_tolerance,
            budget_range=profile.monthly_budget_range,
            currency=profile.budget_currency,
        )

        # --- Step 2: Create goals (each with its own horizon) ---
        # Goals are user-specific, so we CREATE (not MERGE) both
        # the Goal node and the Horizon node. Each user's "wealth_building"
        # goal is separate from another user's — different priorities, horizons.
        for goal in profile.goals:
            tx.run(
                """
                MATCH (u:User {id: $uid})
                CREATE (goal:Goal {type: $type, custom_label: $label})
                CREATE (u)-[:HAS_GOAL {from: $today, to: null, priority: $priority}]->(goal)
                MERGE (h:Horizon {type: $horizon, years: $years})
                CREATE (goal)-[:HORIZON]->(h)
                """,
                uid=profile.user_id,
                today=today,
                type=goal.type,
                label=goal.custom_label,
                priority=goal.priority,
                horizon=goal.horizon,
                years=goal.horizon_years,
            )

        # --- Step 3: Create investment edges ---
        # MERGE on AssetClass nodes (shared across users — many users invest in stocks).
        # CREATE on edges (each user's INVESTS_IN relationship is their own).
        for inv in profile.existing_investments:
            tx.run(
                """
                MATCH (u:User {id: $uid})
                MERGE (a:AssetClass {type: $asset_type})
                CREATE (u)-[:INVESTS_IN {from: $today, to: null}]->(a)
                """,
                uid=profile.user_id,
                today=today,
                asset_type=inv.value,  # .value gets the string from the enum
            )

        # Count total nodes created for this user (for API response)
        result = tx.run(
            """
            MATCH (u:User {id: $uid})-[r]->(n)
            RETURN count(DISTINCT n) AS node_count
            """,
            uid=profile.user_id,
        ).single()

        return {
            "user_id": profile.user_id,
            "status": "created",
            "node_count": result["node_count"] if result else 0,
        }

    def update_user_profile(self, user_id: str, updates: UserProfileUpdate) -> dict:
        """Update profile fields using append-only temporal versioning.

        APPEND-ONLY PATTERN:
        We NEVER delete or overwrite old data. Instead:
        1. Find current edge (where to IS NULL — meaning "active now")
        2. Close it by setting to = today
        3. Create a new edge with from = today, to = null

        This gives us full history. You can query "what was this user's
        risk tolerance 3 months ago?" by filtering on date ranges.

        TS analogy: Like an event-sourced system where you append events
        instead of mutating state. The current state is derived from
        the latest event (edge with to = null).

        Args:
            user_id: The user to update.
            updates: UserProfileUpdate with only the changed fields.

        Returns:
            dict with user_id, status, and fields_updated count.

        Raises:
            ValueError: If user not found.
        """
        with self.driver.session() as session:
            # Check user exists
            exists = session.run(
                "MATCH (u:User {id: $uid}) RETURN u.id", uid=user_id
            ).single()
            if not exists:
                raise ValueError(f"User {user_id} not found")

            fields_updated = 0
            today = date.today().isoformat()

            # Each field that changed gets its own close-and-reopen cycle.
            # We use a helper to reduce repetition (DRY principle).

            if updates.name is not None:
                session.run(
                    "MATCH (u:User {id: $uid}) SET u.name = $name",
                    uid=user_id,
                    name=updates.name,
                )
                fields_updated += 1

            if updates.age_range is not None:
                self._close_and_reopen(
                    session, user_id, today,
                    rel_type="AGE_RANGE",
                    node_label="AgeRange",
                    node_props={"range": updates.age_range},
                )
                fields_updated += 1

            if updates.location_city is not None or updates.location_country is not None:
                # For location, we need both city and country.
                # If only one is provided, fetch the current other value.
                current = self._get_current_node(
                    session, user_id, "LOCATED_IN"
                )
                city = updates.location_city or (current.get("city", "") if current else "")
                country = updates.location_country or (current.get("country", "") if current else "")
                self._close_and_reopen(
                    session, user_id, today,
                    rel_type="LOCATED_IN",
                    node_label="Location",
                    node_props={"city": city, "country": country},
                )
                fields_updated += 1

            if updates.risk_tolerance is not None:
                self._close_and_reopen(
                    session, user_id, today,
                    rel_type="RISK_TOLERANCE",
                    node_label="RiskProfile",
                    node_props={"level": updates.risk_tolerance},
                )
                fields_updated += 1

            if updates.monthly_budget_range is not None:
                currency = updates.budget_currency or "INR"
                self._close_and_reopen(
                    session, user_id, today,
                    rel_type="BUDGET",
                    node_label="Budget",
                    node_props={"range": updates.monthly_budget_range, "currency": currency},
                )
                fields_updated += 1

            if updates.goals is not None:
                # Close ALL existing goal edges, then create fresh ones.
                # Goals are replaced as a set, not individually.
                session.run(
                    """
                    MATCH (u:User {id: $uid})-[r:HAS_GOAL]->(g:Goal)
                    WHERE r.to IS NULL
                    SET r.to = $today
                    """,
                    uid=user_id,
                    today=today,
                )
                for goal in updates.goals:
                    session.run(
                        """
                        MATCH (u:User {id: $uid})
                        CREATE (goal:Goal {type: $type, custom_label: $label})
                        CREATE (u)-[:HAS_GOAL {from: $today, to: null, priority: $priority}]->(goal)
                        MERGE (h:Horizon {type: $horizon, years: $years})
                        CREATE (goal)-[:HORIZON]->(h)
                        """,
                        uid=user_id,
                        today=today,
                        type=goal.type,
                        label=goal.custom_label,
                        priority=goal.priority,
                        horizon=goal.horizon,
                        years=goal.horizon_years,
                    )
                fields_updated += 1

            if updates.existing_investments is not None:
                # Close all current investment edges, create new ones
                session.run(
                    """
                    MATCH (u:User {id: $uid})-[r:INVESTS_IN]->(a:AssetClass)
                    WHERE r.to IS NULL
                    SET r.to = $today
                    """,
                    uid=user_id,
                    today=today,
                )
                for inv in updates.existing_investments:
                    session.run(
                        """
                        MATCH (u:User {id: $uid})
                        MERGE (a:AssetClass {type: $asset_type})
                        CREATE (u)-[:INVESTS_IN {from: $today, to: null}]->(a)
                        """,
                        uid=user_id,
                        today=today,
                        asset_type=inv.value,
                    )
                fields_updated += 1

            logger.info("[graph] User %s updated (%d fields)", user_id, fields_updated)
            return {
                "user_id": user_id,
                "status": "updated",
                "fields_updated": fields_updated,
            }

    # =========================================================================
    # READ — get user profile
    # =========================================================================

    def get_user_profile(self, user_id: str) -> dict | None:
        """Reconstruct the current user profile from the graph.

        Reads only ACTIVE edges (where to IS NULL). Old/historical edges
        are ignored — they're still in the graph for auditing but don't
        affect the current profile.

        Returns None if user not found (for graceful degradation).

        CYPHER CONCEPT — OPTIONAL MATCH:
        Like a LEFT JOIN in SQL. If the pattern doesn't match, the variables
        are null instead of the entire row being excluded.
        """
        with self.driver.session() as session:
            # Get basic user info
            user = session.run(
                "MATCH (u:User {id: $uid}) RETURN u.name AS name, u.created_at AS created_at",
                uid=user_id,
            ).single()

            if not user:
                return None

            # Get all current (active) relationships in one query.
            # type(r) returns the relationship type as a string ("AGE_RANGE", "LOCATED_IN", etc.)
            # properties(n) returns all properties of the connected node as a map.
            records = session.run(
                """
                MATCH (u:User {id: $uid})-[r]->(n)
                WHERE r.to IS NULL OR NOT exists(r.to)
                RETURN type(r) AS rel_type, properties(n) AS props, r.priority AS priority
                """,
                uid=user_id,
            ).data()
            # .data() returns a list of dicts — like cursor.fetchall() in SQL.

            # Build the profile dict from graph data.
            # Each relationship type maps to a profile field.
            profile = {
                "user_id": user_id,
                "name": user["name"],
                "created_at": user["created_at"],
                "goals": [],
                "existing_investments": [],
            }

            for record in records:
                rel = record["rel_type"]
                props = record["props"]

                if rel == "AGE_RANGE":
                    profile["age_range"] = props.get("range")
                elif rel == "GENDER":
                    profile["gender"] = props.get("value")
                elif rel == "LOCATED_IN":
                    profile["location_city"] = props.get("city")
                    profile["location_country"] = props.get("country")
                elif rel == "RISK_TOLERANCE":
                    profile["risk_tolerance"] = props.get("level")
                elif rel == "BUDGET":
                    profile["monthly_budget_range"] = props.get("range")
                    profile["budget_currency"] = props.get("currency")
                elif rel == "HAS_GOAL":
                    # Goals need their horizon too — fetched separately
                    goal_data = {
                        "type": props.get("type"),
                        "custom_label": props.get("custom_label"),
                        "priority": record.get("priority"),
                    }
                    profile["goals"].append(goal_data)
                elif rel == "INVESTS_IN":
                    profile["existing_investments"].append(props.get("type"))

            # Fetch horizons for goals (separate query because Goal->Horizon
            # is a second hop from User)
            if profile["goals"]:
                goal_horizons = session.run(
                    """
                    MATCH (u:User {id: $uid})-[r:HAS_GOAL]->(g:Goal)-[:HORIZON]->(h:Horizon)
                    WHERE r.to IS NULL
                    RETURN g.type AS goal_type, h.type AS horizon, h.years AS years
                    """,
                    uid=user_id,
                ).data()

                # Build a lookup: goal_type -> horizon info
                horizon_map = {
                    gh["goal_type"]: {"horizon": gh["horizon"], "horizon_years": gh["years"]}
                    for gh in goal_horizons
                }
                for goal in profile["goals"]:
                    h = horizon_map.get(goal["type"], {})
                    goal["horizon"] = h.get("horizon")
                    goal["horizon_years"] = h.get("horizon_years")

                # Sort goals by priority (1 = highest)
                profile["goals"].sort(key=lambda g: g.get("priority") or 99)

            return profile

    def format_profile_for_agents(self, user_id: str) -> str:
        """Format user profile as a text block for agent system prompts.

        This is the key integration point — the orchestrator calls this ONCE,
        then passes the string to all 4 agents. Each agent sees the same
        user context without making 4 separate Neo4j reads.

        Returns "" if user not found (graceful degradation — agents run
        without personalization, same as before KG was added).

        The output looks like:
            [USER PROFILE]
            Age: 28-32 | Gender: M
            Location: Bangalore, India
            Risk tolerance: moderate
            Budget: 20-50k INR/month
            ...
        """
        profile = self.get_user_profile(user_id)
        if not profile:
            return ""

        # Build the text block that gets injected into agent prompts.
        # This sits ABOVE the DATA BLOCK in the prompt, so the agent
        # sees it before the research data — avoids "lost in the middle."
        lines = ["[USER PROFILE]"]

        age = profile.get("age_range", "unknown")
        gender = profile.get("gender", "unknown")
        lines.append(f"Age: {age} | Gender: {gender}")

        city = profile.get("location_city", "")
        country = profile.get("location_country", "")
        if city and country:
            lines.append(f"Location: {city}, {country}")

        risk = profile.get("risk_tolerance", "unknown")
        lines.append(f"Risk tolerance: {risk}")

        budget = profile.get("monthly_budget_range", "unknown")
        currency = profile.get("budget_currency", "INR")
        lines.append(f"Budget: {budget} {currency}/month")

        investments = profile.get("existing_investments", [])
        if investments:
            lines.append(f"Existing investments: {', '.join(investments)}")
        else:
            lines.append("Existing investments: none")

        goals = profile.get("goals", [])
        if goals:
            lines.append("Goals:")
            for g in goals:
                horizon = g.get("horizon_years", "?")
                label = g.get("custom_label") or g.get("type", "unknown")
                lines.append(f"  {g.get('priority', '?')}. {label} (horizon: {horizon} yr)")

        # Add LGBTQ+ financial context note if applicable.
        # This is NOT a judgment — it's practical financial context that helps
        # agents give better advice (e.g., emphasizing emergency funds, liquidity).
        if gender == "LGBTQ+":
            lines.append("")
            lines.append("[FINANCIAL CONTEXT NOTE]")
            lines.append(
                "User may face unique financial pressures including social/family "
                "financial independence needs, higher emergency fund requirements, "
                "and potential housing/relocation costs. Prioritize recommendations "
                "that emphasize financial security, liquidity, and building a strong "
                "safety net alongside growth goals."
            )

        return "\n".join(lines)

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def ping(self) -> bool:
        """Check if Neo4j is reachable.

        Matches the pattern used by LongTermMemory.ping() and
        SemanticMemory — returns bool so MemoryManager can report status.
        """
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    @staticmethod
    def _close_and_reopen(
        session,
        user_id: str,
        today: str,
        rel_type: str,
        node_label: str,
        node_props: dict,
    ) -> None:
        """Close current edge and create a new one — the append-only update pattern.

        This is the core temporal versioning primitive. Every profile update
        goes through this:
        1. Find the active edge (to IS NULL)
        2. Set to = today (close it — it becomes historical)
        3. MERGE the new target node (reuse if it already exists)
        4. CREATE a new edge with from = today, to = null (now active)

        Args:
            session: Active Neo4j session.
            user_id: The user being updated.
            today: ISO date string for temporal tracking.
            rel_type: Neo4j relationship type (e.g., "RISK_TOLERANCE").
            node_label: Target node label (e.g., "RiskProfile").
            node_props: Properties for the target node (e.g., {"level": "aggressive"}).
        """
        # Step 1: Close the current active edge
        session.run(
            f"""
            MATCH (u:User {{id: $uid}})-[r:{rel_type}]->(old)
            WHERE r.to IS NULL
            SET r.to = $today
            """,
            uid=user_id,
            today=today,
        )

        # Step 2: Build MERGE clause for the new node.
        # MERGE needs the properties inline in Cypher — we build the props
        # string dynamically from the dict. Each prop uses a parameter ($prop_X)
        # to prevent Cypher injection (never concatenate user input into queries).
        prop_parts = []
        params = {"uid": user_id, "today": today}
        for i, (key, value) in enumerate(node_props.items()):
            param_name = f"prop_{i}"
            prop_parts.append(f"{key}: ${param_name}")
            params[param_name] = value

        props_str = ", ".join(prop_parts)

        # Step 3: MERGE the new node + CREATE a fresh edge
        session.run(
            f"""
            MATCH (u:User {{id: $uid}})
            MERGE (n:{node_label} {{{props_str}}})
            CREATE (u)-[:{rel_type} {{from: $today, to: null}}]->(n)
            """,
            **params,
        )

    @staticmethod
    def _get_current_node(session, user_id: str, rel_type: str) -> dict | None:
        """Get the current (active) node connected by a relationship type.

        Used when updating partial fields — e.g., user changes city but not
        country, so we need to read the current country before creating the
        new Location node.
        """
        result = session.run(
            f"""
            MATCH (u:User {{id: $uid}})-[r:{rel_type}]->(n)
            WHERE r.to IS NULL
            RETURN properties(n) AS props
            """,
            uid=user_id,
        ).single()

        return result["props"] if result else None
