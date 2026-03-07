from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    """env config loading and validation"""
    anthropic_api_key: str
    tavily_api_key: str
    langsmith_api_key: str
    class Config:
        env_file = ".env"

settings = Settings()