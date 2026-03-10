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

        # Use mock memory for now — swap to real MemoryManager when infra is ready
        from unittest.mock import AsyncMock, MagicMock

        memory = MagicMock()
        memory.parallel_search = AsyncMock(return_value={
            "long_term": [],
            "semantic": [],
        })
        memory.update_agent_status = AsyncMock()
        memory.store_research = AsyncMock()
        memory.create_session = AsyncMock()
        memory.end_session = AsyncMock()

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
