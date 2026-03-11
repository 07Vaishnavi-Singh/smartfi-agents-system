"""Episodic memory using PostgreSQL.

Stores the full audit trail of every research session:
- Episodes (who asked what, when)
- Agent decisions (what each agent did and found)
- Knowledge entries (extracted facts as graph triples)

PYTHON CONCEPT — context manager with `with`:
`with session_factory() as session:` ensures the session is always
closed, even if an exception occurs. Like try/finally but cleaner.
TS equivalent: using a disposable pattern or try/finally
Rust equivalent: Drop trait (auto-cleanup when variable goes out of scope)
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import text

from investment_research_system.memory.database import (
    AgentDecision,
    Episode,
    KnowledgeEntry,
)

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """PostgreSQL-backed episodic memory for research sessions."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create_episode(self, episode_id: str, query: str) -> Episode:
        """Start a new research episode."""
        with self.session_factory() as session:
            episode = Episode(id=episode_id, query=query, status="started")
            session.add(episode)
            session.commit()
            session.refresh(episode)
            logger.info("Episode created: %s", episode_id)
            return episode

    def complete_episode(
        self, episode_id: str, total_tokens: int = 0, total_cost_usd: float = 0.0
    ) -> None:
        """Mark an episode as completed with final stats."""
        with self.session_factory() as session:
            episode = session.query(Episode).filter_by(id=episode_id).first()
            if episode:
                episode.status = "completed"
                episode.total_tokens = total_tokens
                episode.total_cost_usd = total_cost_usd
                episode.completed_at = datetime.now(timezone.utc)
                session.commit()

    def fail_episode(self, episode_id: str, error: str = "") -> None:
        """Mark an episode as failed."""
        with self.session_factory() as session:
            episode = session.query(Episode).filter_by(id=episode_id).first()
            if episode:
                episode.status = "failed"
                episode.completed_at = datetime.now(timezone.utc)
                session.commit()

    def log_decision(
        self,
        episode_id: str,
        agent_name: str,
        action: str,
        result_summary: str = "",
        confidence: float = 0.0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> AgentDecision:
        """Log what an agent did during an episode."""
        with self.session_factory() as session:
            decision = AgentDecision(
                episode_id=episode_id,
                agent_name=agent_name,
                action=action,
                result_summary=result_summary,
                confidence=confidence,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
            )
            session.add(decision)
            session.commit()
            session.refresh(decision)
            return decision

    def store_knowledge(
        self,
        episode_id: str,
        subject: str,
        predicate: str,
        object_: str,
        source_agent: str = "",
        confidence: float = 0.0,
    ) -> KnowledgeEntry:
        """Store a knowledge graph triple.

        Example: store_knowledge(ep_id, "NVIDIA", "competes_with", "AMD", "researcher", 0.9)
        """
        with self.session_factory() as session:
            entry = KnowledgeEntry(
                episode_id=episode_id,
                subject=subject,
                predicate=predicate,
                object_=object_,
                source_agent=source_agent,
                confidence=confidence,
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry

    def get_episode(self, episode_id: str) -> Episode | None:
        """Get an episode by ID."""
        with self.session_factory() as session:
            return session.query(Episode).filter_by(id=episode_id).first()

    def get_decisions(self, episode_id: str) -> list[AgentDecision]:
        """Get all agent decisions for an episode."""
        with self.session_factory() as session:
            return session.query(AgentDecision).filter_by(episode_id=episode_id).all()

    def search_episodes(self, query_substring: str, limit: int = 10) -> list[Episode]:
        """Find past episodes by query text (simple LIKE search)."""
        with self.session_factory() as session:
            return (
                session.query(Episode)
                .filter(Episode.query.ilike(f"%{query_substring}%"))
                .order_by(Episode.created_at.desc())
                .limit(limit)
                .all()
            )

    def get_knowledge(self, subject: str | None = None, limit: int = 20) -> list[KnowledgeEntry]:
        """Query the knowledge graph. Optionally filter by subject."""
        with self.session_factory() as session:
            q = session.query(KnowledgeEntry)
            if subject:
                q = q.filter(KnowledgeEntry.subject.ilike(f"%{subject}%"))
            return q.order_by(KnowledgeEntry.created_at.desc()).limit(limit).all()

    def ping(self) -> bool:
        """Health check — verify Postgres is reachable."""
        try:
            with self.session_factory() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
