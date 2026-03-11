"""LangSmith tracing setup — observability for the entire agent pipeline.

LangSmith automatically traces all LangChain/LangGraph calls when
the right environment variables are set. This module handles:
1. Initializing tracing (setting env vars programmatically)
2. Custom run metadata (tagging traces with query info)
3. A helper to check if tracing is active

HOW LANGSMITH WORKS:
Every time your code calls `llm.ainvoke()` or `graph.ainvoke()`,
LangChain checks if LANGCHAIN_TRACING_V2=true. If so, it sends
the call details (prompt, response, tokens, latency) to LangSmith's
API in the background. You see it all in a web dashboard.

Think of it as:
- console.log() → you read terminal output manually
- LangSmith → structured, searchable, visual traces with timing

WHAT YOU GET FOR FREE (no code changes):
- Every LLM call: prompt sent, response received, tokens, cost
- LangGraph flow: which nodes ran, what state flowed between them
- Errors: stack traces attached to the exact call that failed
- Latency: how long each step took (useful for optimization)

PYTHON CONCEPT — os.environ:
Python's way to get/set environment variables.
os.environ["KEY"] = "value" sets it for the current process.
os.getenv("KEY") reads it (returns None if missing).
TS equivalent: process.env.KEY
Rust equivalent: std::env::var("KEY")
"""

import logging
import os

logger = logging.getLogger(__name__)


def init_tracing(
    project_name: str = "investment-research",
) -> bool:
    """Initialize LangSmith tracing if an API key is available.

    Sets the environment variables that LangChain looks for.
    Call this once at app startup (before any LLM calls).

    Args:
        project_name: The LangSmith project to log traces to.
            All traces for this app will appear under this project
            in the LangSmith dashboard.

    Returns:
        True if tracing was enabled, False if skipped (no API key).

    PYTHON CONCEPT — early return pattern:
    If the API key is missing, we return False immediately.
    No need for an else branch. Same as Rust's early return with `?`.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")

    if not api_key:
        logger.info("[tracing] No LANGSMITH_API_KEY found — tracing disabled")
        return False

    # These env vars are what LangChain/LangGraph checks internally.
    # Setting them here means you don't need to export them in your shell.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name

    logger.info(
        "[tracing] LangSmith tracing enabled — project: %s", project_name
    )
    return True


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently active."""
    return os.getenv("LANGCHAIN_TRACING_V2") == "true"


def get_run_metadata(query: str, session_id: str) -> dict:
    """Build metadata dict to attach to a LangSmith trace.

    Metadata makes traces searchable and filterable in the dashboard.
    For example, you can filter by "show me all traces where
    depth=deep" or "find traces for NVIDIA queries".

    Args:
        query: The user's research question.
        session_id: The current session ID.

    Returns:
        Dict of metadata to pass to LangGraph's ainvoke() config.

    PYTHON CONCEPT — dict literal:
    Python dicts are created with {key: value} syntax.
    TS equivalent: { key: value } (identical!)
    Rust equivalent: HashMap::from([("key", "value")])
    """
    return {
        "query_preview": query[:100],
        "session_id": session_id,
    }


def get_run_config(query: str, session_id: str, run_name: str | None = None) -> dict:
    """Build a LangGraph/LangChain run config with tracing metadata.

    Pass this as the `config` parameter to graph.ainvoke():
        config = get_run_config(query, session_id)
        result = await graph.ainvoke(state, config=config)

    This attaches metadata and tags to the trace so you can
    find it later in the LangSmith dashboard.

    LANGCHAIN CONCEPT — RunnableConfig:
    LangChain's config dict can include:
    - "metadata": key-value pairs attached to the trace
    - "tags": string labels for filtering
    - "run_name": custom name shown in the dashboard
    - "callbacks": custom callback handlers
    """
    config = {
        "metadata": get_run_metadata(query, session_id),
        "tags": ["investment-research"],
    }

    if run_name:
        config["run_name"] = run_name

    return config
