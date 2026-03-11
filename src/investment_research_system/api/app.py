"""FastAPI application factory.

PYTHON CONCEPT — APP FACTORY PATTERN:
Instead of creating the app at module level, we use a create_app() function.
This lets tests create fresh app instances with different configs.
TS equivalent: NestJS's NestFactory.create()
Rust equivalent: actix_web::App::new() inside a factory function

PYTHON CONCEPT — CORS MIDDLEWARE:
CORS (Cross-Origin Resource Sharing) controls which domains can call
your API. Without it, a browser on localhost:3000 (your Streamlit UI)
can't call localhost:8000 (your API). The middleware adds the right
headers to every response.
"""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from investment_research_system.api.routes import router
from investment_research_system.observability.tracing import init_tracing

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns a fully configured app with:
    - LangSmith tracing (if API key is set)
    - CORS middleware (allows Streamlit UI to call the API)
    - Research routes mounted
    - Health check endpoint
    """
    # Load .env before anything else
    load_dotenv()

    # Initialize LangSmith tracing — must happen before any LLM calls.
    # If LANGSMITH_API_KEY is missing, tracing is silently disabled.
    tracing_enabled = init_tracing()
    logger.info("[app] LangSmith tracing: %s", "enabled" if tracing_enabled else "disabled")

    app = FastAPI(
        title="Investment Research API",
        description="Multi-agent investment research system with async job processing",
        version="0.1.0",
    )

    # CORS — allow the Streamlit UI (and any local dev tools) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to specific domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routes
    app.include_router(router)

    logger.info("[app] FastAPI application created")
    return app
