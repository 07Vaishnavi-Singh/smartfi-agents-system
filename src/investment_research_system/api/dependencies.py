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


def create_llm(model: str, settings):
    """Create LLM instance based on model name from config.

    Factory pattern — the model string in config.py is the single source of truth.
    This makes it trivial to switch models (e.g., cheaper models for low-priority
    queries) without changing any agent code.

    PYTHON CONCEPT — factory function:
    Instead of hardcoding `ChatGoogleGenerativeAI(...)` everywhere, this function
    inspects the model name and creates the right client. The rest of the code
    just calls `create_llm()` and gets back a LangChain chat model.
    TS equivalent: a factory function that returns different class instances
    Rust equivalent: a builder pattern or enum-based construction

    Args:
        model: Model identifier string (e.g., "claude-sonnet-4-20250514", "gemini-2.5-flash-lite").
        settings: Application settings with API keys and model parameters.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If the model string doesn't match any supported provider.
    """
    import os

    if "claude" in model:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    elif "gemini" in model:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    else:
        raise ValueError(
            f"Unsupported model: {model}. "
            f"Set DEFAULT_MODEL in .env to a 'claude-*' or 'gemini-*' model."
        )


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
        from config import settings

        from investment_research_system.agents.analyst import AnalystAgent
        from investment_research_system.agents.researcher import ResearcherAgent
        from investment_research_system.agents.risk_assessor import RiskAssessorAgent
        from investment_research_system.agents.sentiment import SentimentAgent
        from investment_research_system.orchestrator.graph import ResearchOrchestrator
        from investment_research_system.tools.tavily_search import TavilySearch

        # Create LLM from config — no more hardcoded model names
        llm = create_llm(settings.default_model, settings)
        logger.info("[dependencies] LLM created: %s", settings.default_model)

        # Create fallback LLMs from config — tried in order on rate limit
        fallback_llms = []
        for model_name in settings.fallback_models:
            if model_name != settings.default_model:
                try:
                    fallback_llm = create_llm(model_name, settings)
                    fallback_llms.append(fallback_llm)
                except Exception as e:
                    logger.warning("[dependencies] Could not create fallback model %s: %s", model_name, e)
        logger.info("[dependencies] Fallback chain: %s → %s", settings.default_model, [getattr(l, "model", "?") for l in fallback_llms])

        # Real MemoryManager — connects to Redis, Qdrant, Mem0, and optionally Postgres
        from investment_research_system.memory.long_term import LongTermMemory
        from investment_research_system.memory.manager import MemoryManager
        from investment_research_system.memory.semantic import SemanticMemory
        from investment_research_system.memory.short_term import ShortTermMemory

        short_term = ShortTermMemory(redis_url=settings.redis_url)
        long_term = LongTermMemory(qdrant_url=settings.qdrant_url)
        semantic = SemanticMemory()

        # Postgres is optional — if it's not running, memory still works without it
        episodic = None
        try:
            from investment_research_system.memory.database import create_tables, get_engine, get_session_factory
            from investment_research_system.memory.episodic import EpisodicMemory

            engine = get_engine(settings.postgres_url)
            create_tables(engine)
            session_factory = get_session_factory(engine)
            episodic = EpisodicMemory(session_factory=session_factory)
            logger.info("[dependencies] Postgres episodic memory connected")
        except Exception as e:
            logger.warning("[dependencies] Postgres unavailable, episodic memory disabled: %s", e)

        # Neo4j is optional — if it's not running, agents run without
        # user profile personalization (same behavior as before KG was added).
        # Matches the same graceful degradation pattern as Postgres above.
        graph_memory = None
        try:
            from neo4j import GraphDatabase

            from investment_research_system.memory.graph import GraphMemory

            neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            graph_memory = GraphMemory(neo4j_driver)
            graph_memory._ensure_indexes()  # idempotent — safe on every startup
            logger.info("[dependencies] Neo4j graph memory connected")
        except Exception as e:
            logger.warning("[dependencies] Neo4j unavailable, graph memory disabled: %s", e)

        memory = MemoryManager(
            short_term=short_term,
            long_term=long_term,
            semantic=semantic,
            episodic=episodic,
            graph=graph_memory,
        )

        tavily = TavilySearch(api_key=settings.tavily_api_key)

        agents = [
            ResearcherAgent(llm=llm, memory=memory, tavily=tavily, fallback_llms=fallback_llms),
            AnalystAgent(llm=llm, memory=memory, tavily=tavily, fallback_llms=fallback_llms),
            RiskAssessorAgent(llm=llm, memory=memory, tavily=tavily, fallback_llms=fallback_llms),
            SentimentAgent(llm=llm, memory=memory, tavily=tavily, fallback_llms=fallback_llms),
        ]

        # Create orchestrator LLM — may use a different model than agents
        orchestrator_model = settings.orchestrator_model or settings.default_model
        orchestrator_llm = create_llm(orchestrator_model, settings)
        logger.info("[dependencies] Orchestrator LLM created: %s", orchestrator_model)

        return ResearchOrchestrator(
            agents=agents,
            memory_manager=memory,
            orchestrator_llm=orchestrator_llm,
            max_retries=1,
        )

    except Exception as e:
        logger.exception("[dependencies] Failed to create orchestrator: %s", e)
        return None
