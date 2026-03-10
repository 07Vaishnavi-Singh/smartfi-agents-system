from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application config — loaded from .env automatically.

    Python style note: In Python, we use blank lines between class-level
    sections for readability. PEP 8 (Python's style guide) says:
    - 2 blank lines before/after top-level classes and functions
    - 1 blank line between methods inside a class
    This is enforced by ruff (your linter).
    """

    # --- API Keys (required — app crashes on startup if missing) ---
    anthropic_api_key: str
    tavily_api_key: str
    # langsmith_api_key: str

    # --- Infrastructure URLs ---
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"

    # --- Model Settings ---
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7

    # --- Cost Controls ---
    max_budget_per_query_usd: float = 0.50

    # --- Embedding Settings ---
    # We'll use a local embedding model (free, no API key needed)
    # sentence-transformers runs on your machine
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384  

    # --- Session Settings ---
    session_ttl_seconds: int = 3600  

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()