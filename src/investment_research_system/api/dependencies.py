"""FastAPI dependency injection — wiring up the orchestrator and job store.

PYTHON CONCEPT — DEPENDENCY INJECTION:
Instead of importing singletons directly, FastAPI lets you declare
"this route needs X" and FastAPI provides it. This makes testing easy:
in tests, you swap the real orchestrator for a mock.

TS equivalent: NestJS @Inject() decorators
Rust equivalent: Actix-web's .app_data() / .configure()

FastAPI uses a function that returns the dependency:
    def get_job_store() -> JobStore:
        return _job_store

Then in a route:
    async def create_research(store: JobStore = Depends(get_job_store)):
        ...

In tests, you override it:
    app.dependency_overrides[get_job_store] = lambda: mock_store
"""

import logging

from investment_research_system.api.job_store import JobStore

logger = logging.getLogger(__name__)

# Module-level singleton — created once, reused across requests.
# PYTHON CONCEPT — module-level state:
# Python modules are singletons. When you import this module,
# these objects are created once. Every subsequent import reuses them.
# TS equivalent: a module-scoped let/const
_job_store = JobStore()


def get_job_store() -> JobStore:
    """Provide the shared JobStore instance."""
    return _job_store


def get_orchestrator_factory():
    """Provide the orchestrator factory function as a dependency.

    Returns the create_orchestrator function itself (not its result).
    Routes call the returned function when they need an orchestrator.
    In tests, override this to return a lambda that returns a mock.
    """
    return create_orchestrator


def create_orchestrator():
    """Create a fresh orchestrator with all agents.

    Returns None if required API keys are missing.
    Called per-job (not a singleton) because agents hold per-session state.

    PYTHON CONCEPT — late import:
    We import inside the function to avoid circular imports and
    to only load heavy dependencies when actually needed.
    TS equivalent: dynamic import() inside a function
    """
    try:
        import os

        from dotenv import load_dotenv

        load_dotenv()

        google_key = os.getenv("GOOGLE_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if not google_key or not tavily_key:
            logger.warning("[dependencies] Missing API keys — orchestrator unavailable")
            return None

        from langchain_google_genai import ChatGoogleGenerativeAI

        from investment_research_system.agents.analyst import AnalystAgent
        from investment_research_system.agents.researcher import ResearcherAgent
        from investment_research_system.agents.risk_assessor import RiskAssessorAgent
        from investment_research_system.agents.sentiment import SentimentAgent
        from investment_research_system.orchestrator.graph import ResearchOrchestrator
        from investment_research_system.tools.tavily_search import TavilySearch

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=google_key,
            max_output_tokens=1024,
            temperature=0.7,
        )

        # Real MemoryManager — connects to Redis, Qdrant, Mem0, and optionally Postgres
        from investment_research_system.memory.manager import MemoryManager
        from investment_research_system.memory.short_term import ShortTermMemory
        from investment_research_system.memory.long_term import LongTermMemory
        from investment_research_system.memory.semantic import SemanticMemory

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        short_term = ShortTermMemory(redis_url=redis_url)
        long_term = LongTermMemory(qdrant_url=qdrant_url)
        semantic = SemanticMemory()

        # Postgres is optional — if it's not running, memory still works without it
        episodic = None
        postgres_url = os.getenv("POSTGRES_URL", "postgresql://agent_user:agent_pass@localhost:5432/agent_memory")
        try:
            from investment_research_system.memory.database import create_tables, get_engine, get_session_factory
            from investment_research_system.memory.episodic import EpisodicMemory

            engine = get_engine(postgres_url)
            create_tables(engine)
            session_factory = get_session_factory(engine)
            episodic = EpisodicMemory(session_factory=session_factory)
            logger.info("[dependencies] Postgres episodic memory connected")
        except Exception as e:
            logger.warning("[dependencies] Postgres unavailable, episodic memory disabled: %s", e)

        memory = MemoryManager(
            short_term=short_term,
            long_term=long_term,
            semantic=semantic,
            episodic=episodic,
        )

        tavily = TavilySearch(api_key=tavily_key)

        agents = [
            ResearcherAgent(llm=llm, memory=memory, tavily=tavily),
            AnalystAgent(llm=llm, memory=memory, tavily=tavily),
            RiskAssessorAgent(llm=llm, memory=memory, tavily=tavily),
            SentimentAgent(llm=llm, memory=memory, tavily=tavily),
        ]

        return ResearchOrchestrator(agents=agents, max_retries=1)

    except Exception as e:
        logger.exception("[dependencies] Failed to create orchestrator: %s", e)
        return None
