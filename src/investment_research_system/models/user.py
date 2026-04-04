"""User profile models for the Knowledge Graph (Graph 1).

These Pydantic models define the shape of user data that flows into Neo4j.
The onboarding form collects this data once, then it's stored as a graph —
nodes (User, Goal, Location) connected by edges (HAS_GOAL, LOCATED_IN).

WHY A GRAPH instead of a flat table?
- A user can have multiple goals, each with its own horizon
- Relationships have temporal metadata (from/to dates) for history tracking
- Graph queries like "find all users in Bangalore with aggressive risk" are natural
- Same Neo4j instance will later hold asset relationships, preferences, etc.

TS equivalent: These are like Zod schemas that validate API input before
storing it in the database.
"""

from enum import Enum

from pydantic import BaseModel, Field


# --- Enums ---
# Python enums work like TypeScript's `enum` or Rust's fieldless enum.
# Inheriting from `str` makes them JSON-serializable automatically —
# so FastAPI can accept "stocks" in a JSON body without a custom parser.


class InvestmentType(str, Enum):
    """Asset classes the user already invests in.

    Used during onboarding to understand existing portfolio composition.
    Maps to (:AssetClass) nodes in Neo4j.
    """

    stocks = "stocks"
    mutual_funds = "mutual_funds"
    crypto = "crypto"
    gold = "gold"
    fds = "fds"                  # Fixed Deposits (common in India)
    real_estate = "real_estate"
    bonds = "bonds"
    none = "none"                # user has no existing investments


class FinancialGoal(BaseModel):
    """A single financial goal with its own investment horizon.

    KEY DESIGN DECISION: Horizon belongs to the goal, not the user.
    A user saving for a house (1-5yr) and building wealth (5+yr) has
    two different horizons — attaching horizon to the user loses this.

    In Neo4j this becomes:
        (User)-[:HAS_GOAL {priority: 1}]->(Goal)-[:HORIZON]->(Horizon)
    """

    type: str                        # "wealth_building", "retirement", "passive_income", "trading", "save_for_x"
    custom_label: str | None = None  # only used when type is "save_for_x" — e.g., "house", "car"
    horizon: str                     # "short", "medium", "long"
    horizon_years: str               # "<1", "1-5", "5+" — human-readable version
    priority: int                    # 1 = highest priority goal


class UserProfile(BaseModel):
    """Full user profile collected during onboarding.

    This is the INPUT shape for POST /users/onboard.
    Every field maps to a node + edge in Neo4j Graph 1.

    Validation happens here at the Pydantic layer (not in Neo4j).
    If this model accepts it, we trust it for graph storage.
    """

    user_id: str                     # unique identifier — no auth for now, layered later
    name: str
    age_range: str                   # "18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48-52", "52+"
    gender: str                      # "M", "F", "LGBTQ+", "prefer_not_to_say"
    location_city: str
    location_country: str
    risk_tolerance: str              # "conservative", "moderate", "aggressive"
    monthly_budget_range: str        # "0-10k", "10-20k", "20-50k", "50k+"
    budget_currency: str = "INR"     # default INR for Indian market focus
    goals: list[FinancialGoal] = Field(
        ...,                         # "..." means REQUIRED — Pydantic raises error if missing
        min_length=1,                # must have at least 1 goal
        max_length=3,                # max 3 to keep graph manageable
    )
    existing_investments: list[InvestmentType]


class UserProfileUpdate(BaseModel):
    """Partial update model for PUT /users/{user_id}/profile.

    All fields are optional — only send what changed.
    This pattern is called a "Patch model" in API design.

    TS equivalent: Partial<UserProfile>
    Rust equivalent: struct with all Option<T> fields

    In Neo4j, updates are APPEND-ONLY:
    1. Find current edge (where to IS NULL)
    2. Set to = today (close it)
    3. Create new edge with from = today, to = null
    This means we never lose history — you can always query "what was
    the user's risk tolerance 3 months ago?"
    """

    name: str | None = None
    age_range: str | None = None
    gender: str | None = None
    location_city: str | None = None
    location_country: str | None = None
    risk_tolerance: str | None = None
    monthly_budget_range: str | None = None
    budget_currency: str | None = None
    goals: list[FinancialGoal] | None = None
    existing_investments: list[InvestmentType] | None = None
