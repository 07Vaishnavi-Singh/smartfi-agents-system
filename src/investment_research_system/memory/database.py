"""SQLAlchemy models and database connection for episodic memory (Postgres).

Three tables:
  Episode        — one row per research session
  AgentDecision  — what each agent did during an episode
  KnowledgeEntry — extracted facts as (subject, predicate, object) triples

PYTHON CONCEPT — SQLAlchemy ORM:
SQLAlchemy maps Python classes to database tables.
Each class = a table. Each instance = a row.
TS equivalent: Prisma models
Rust equivalent: Diesel schema
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models.

    PYTHON CONCEPT — DeclarativeBase:
    All your table classes inherit from this. SQLAlchemy uses it to
    track which classes are tables and how they relate to each other.
    """

    pass


class Episode(Base):
    """A complete research session.

    One row per user query. Links to multiple AgentDecisions and KnowledgeEntries.
    """

    __tablename__ = "episodes"

    id = Column(String, primary_key=True)
    query = Column(Text, nullable=False)
    status = Column(String, default="started")
    total_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Relationships — SQLAlchemy auto-joins these when you access them
    decisions = relationship("AgentDecision", back_populates="episode", cascade="all, delete-orphan")
    knowledge_entries = relationship("KnowledgeEntry", back_populates="episode", cascade="all, delete-orphan")


class AgentDecision(Base):
    """What a single agent did during an episode.

    Tracks: which agent, what action it took, what it found, confidence level.
    """

    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    agent_name = Column(String, nullable=False)
    action = Column(String, nullable=False)
    result_summary = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    episode = relationship("Episode", back_populates="decisions")


class KnowledgeEntry(Base):
    """An extracted fact stored as a knowledge graph triple.

    Example: ("NVIDIA", "competes_with", "AMD")
    """

    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    subject = Column(String, nullable=False)
    predicate = Column(String, nullable=False)
    object_ = Column(String, nullable=False)
    source_agent = Column(String, default="")
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    episode = relationship("Episode", back_populates="knowledge_entries")


def get_engine(database_url: str):
    """Create a SQLAlchemy engine (connection pool)."""
    return create_engine(database_url, echo=False)


def get_session_factory(engine):
    """Create a session factory bound to the engine."""
    return sessionmaker(bind=engine)


def create_tables(engine):
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
