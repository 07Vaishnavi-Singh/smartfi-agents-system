from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# --- Enums ---
class ResearchDepth(str, Enum):
    quick = "quick"          
    standard = "standard"   
    deep = "deep"          


class SourceType(str, Enum):
    web = "web"           
    memory = "memory"       
    database = "database"   


# --- Core Schemas ---
class Source(BaseModel):
    """Where a piece of information came from."""
    title: str
    url: str | None = None                            
    source_type: SourceType
    reliability_score: float = Field(ge=0.0, le=1.0)                           


class ResearchQuery(BaseModel):
    """What the user sends in. This is the INPUT to the entire system."""
    query: str                                                         
    focus_areas: list[str] = Field(default_factory=list)               
    depth: ResearchDepth = ResearchDepth.standard
    max_budget_usd: float = Field(default=0.50, ge=0.01, le=10.0)     


class AgentResponse(BaseModel):
    """What each agent returns. Every agent (researcher, analyst, risk, sentiment)
    must return this same shape. This is how the orchestrator can treat all agents
    uniformly — it doesn't care which agent produced it."""
    agent_name: str
    content: str                                          # the actual analysis text
    confidence: float = Field(ge=0.0, le=1.0)           
    sources: list[Source] = Field(default_factory=list)
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)       
    timestamp: datetime = Field(default_factory=datetime.now)


class ResearchReport(BaseModel):
    """Final output sent to the user. The orchestrator builds this by combining
    all agent responses, resolving conflicts, and running quality checks."""
    id: str                                                # unique report ID
    query: ResearchQuery                                   # original question
    summary: str                                           # executive summary
    agent_responses: list[AgentResponse]                   # individual agent outputs
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    processing_time_seconds: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=datetime.now)
