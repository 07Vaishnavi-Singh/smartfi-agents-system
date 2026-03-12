"""Typed exception hierarchy for the research system.

Instead of catching generic `Exception` everywhere, typed errors let the
orchestrator make intelligent recovery decisions:
- Timeout → retry with same LLM
- Rate limit → exponential backoff, or switch provider
- Budget exceeded → hard stop, return partial results
- Refusal → don't retry, the model won't answer this

FUTURE: This hierarchy is the foundation for multi-LLM fallback.
The error type tells the orchestrator WHETHER to fallback, not just
THAT something failed.

Rust equivalent: enum AgentError { Timeout(..), RateLimit(..), ... }
TS equivalent:   class LLMTimeoutError extends AgentError { ... }
"""


class AgentError(Exception):
    """Base exception for all agent-related errors.

    Every typed error carries the agent_name so the orchestrator
    can log, track, and circuit-break per-agent.
    """

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}")


# ---------------------------------------------------------------------------
# LLM errors — problems with the model API itself
# ---------------------------------------------------------------------------

class LLMTimeoutError(AgentError):
    """LLM call exceeded the timeout threshold.

    Recovery: retry (transient), then fallback to another provider.
    """
    pass


class LLMRateLimitError(AgentError):
    """LLM API returned a rate-limit (429) response.

    Recovery: exponential backoff, then fallback to another provider.
    """
    pass


class LLMOutputError(AgentError):
    """LLM returned empty, malformed, or unusable output.

    Recovery: retry with same prompt (sometimes helps), or skip.
    """
    pass


class LLMRefusalError(LLMOutputError):
    """LLM refused to answer the query (content policy, etc.).

    Recovery: do NOT retry — the model won't answer this.
    """
    pass


# ---------------------------------------------------------------------------
# Infrastructure errors — problems with backing services
# ---------------------------------------------------------------------------

class MemoryUnavailableError(AgentError):
    """Memory backend (Redis/Qdrant/Postgres) is unreachable.

    Recovery: continue without memory (degraded mode).
    """
    pass


# ---------------------------------------------------------------------------
# Business logic errors — hard constraints
# ---------------------------------------------------------------------------

class BudgetExceededError(AgentError):
    """Query has exceeded its allocated token/cost budget.

    Recovery: do NOT retry or fallback — return what you have.
    """
    pass


# ---------------------------------------------------------------------------
# Security errors — not agent-scoped
# ---------------------------------------------------------------------------

class PromptInjectionError(Exception):
    """User query contains suspected prompt injection patterns.

    Not an AgentError because it's detected before any agent runs.
    """
    pass
