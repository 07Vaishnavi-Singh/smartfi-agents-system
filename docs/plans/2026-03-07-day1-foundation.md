# Day 1: Foundation - Project Skeleton, Config & Data Models

## Goal
By end of today, every file exists, config loads from .env, all data models are defined,
database tables are modeled, and a dummy test passes. Nothing "runs" yet - but the
entire skeleton is in place.

---

## Mental Model

```
TODAY YOU ARE BUILDING THE SKELETON:

User asks a question
    |
    v
ResearchQuery (schema)          <-- you define this today
    |
    v
Agents process it               <-- Day 3
    |
    v
AgentResponse (schema)          <-- you define this today
    |
    v
Orchestrator combines them      <-- Day 4
    |
    v
ResearchReport (schema)         <-- you define this today
    |
    v
Stored in memory as:
  - MemoryEntry (schema)        <-- you define this today
  - Episode (db model)          <-- you define this today
  - ConflictRecord (schema)     <-- you define this today
```

Everything downstream depends on these shapes being right.

---

## Tasks (in order)

### Task 1: Create folder structure + __init__.py files
**Time: 10 mins**

Create all directories and empty __init__.py files:

```
src/
├── __init__.py
├── config.py
├── models/
│   ├── __init__.py
│   ├── schemas.py
│   └── database.py
├── memory/
│   ├── __init__.py
│   ├── short_term.py
│   ├── long_term.py
│   ├── episodic.py
│   └── semantic.py
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── researcher.py
│   ├── analyst.py
│   ├── risk_assessor.py
│   └── sentiment.py
├── orchestrator/
│   ├── __init__.py
│   ├── graph.py
│   ├── conflict.py
│   └── quality.py
├── tools/
│   ├── __init__.py
│   ├── tavily_search.py
│   └── mcp_tools.py
├── api/
│   ├── __init__.py
│   └── routes.py
├── ui/
│   └── app.py
└── observability/
    ├── __init__.py
    └── tracing.py
tests/
├── __init__.py
└── test_models.py
```

**How to create in terminal:**
```bash
cd ~/Developer/ai-projects/multi-agent-orchestration

# Create all directories
mkdir -p src/{models,memory,agents,orchestrator,tools,api,ui,observability}
mkdir -p tests

# Create all __init__.py files (makes folders importable as Python packages)
touch src/__init__.py
touch src/{models,memory,agents,orchestrator,tools,api,observability}/__init__.py
touch tests/__init__.py

# Create all module files
touch src/config.py
touch src/models/{schemas.py,database.py}
touch src/memory/{short_term.py,long_term.py,episodic.py,semantic.py}
touch src/agents/{base.py,researcher.py,analyst.py,risk_assessor.py,sentiment.py}
touch src/orchestrator/{graph.py,conflict.py,quality.py}
touch src/tools/{tavily_search.py,mcp_tools.py}
touch src/api/routes.py
touch src/ui/app.py
touch src/observability/tracing.py
touch tests/test_models.py
```

**What you learned:** `mkdir -p` creates nested dirs. `touch` creates empty files.

---

### Task 2: Write config.py
**Time: 15 mins**

This file loads your API keys and settings from `.env`.

**Concepts you'll learn:**
- `pydantic-settings` for type-safe config loading
- Why you never hardcode API keys
- The `BaseSettings` class (reads from .env automatically)

**What it should contain:**
- All API keys (Anthropic, Tavily, LangSmith)
- Database URLs (Redis, Qdrant, Postgres)
- Model settings (which Claude model, max tokens, temperature)
- Cost tracking defaults (max cost per query)

**Key pattern:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    tavily_api_key: str
    # ... more fields

    class Config:
        env_file = ".env"

settings = Settings()  # auto-loads from .env
```

**Note:** You'll need to install pydantic-settings:
```bash
uv add pydantic-settings
```

---

### Task 3: Write schemas.py (Pydantic models)
**Time: 30 mins** (most important task today)

These are the data shapes that flow through your entire system.

**Concepts you'll learn:**
- Pydantic models (like TypeScript interfaces but they validate at runtime)
- Enums for fixed choices
- Optional fields with defaults
- Nested models

**Models to define:**

1. **ResearchQuery** - what the user sends in
   - query: str (the question)
   - focus_areas: list[str] (optional: ["financials", "risk", "sentiment"])
   - depth: enum (quick / standard / deep)
   - max_budget_usd: float (cost limit)

2. **AgentResponse** - what each agent returns
   - agent_name: str
   - content: str (the analysis)
   - confidence: float (0.0 to 1.0)
   - sources: list[Source]
   - tokens_used: int
   - cost_usd: float
   - timestamp: datetime

3. **Source** - where information came from
   - title: str
   - url: str (optional)
   - source_type: enum (web / memory / database)
   - reliability_score: float

4. **ResearchReport** - final output to user
   - query: ResearchQuery (original question)
   - summary: str
   - agent_responses: list[AgentResponse]
   - conflicts: list[ConflictRecord]
   - total_cost_usd: float
   - total_tokens: int
   - processing_time_seconds: float

5. **ConflictRecord** - when agents disagree
   - agents_involved: list[str]
   - topic: str
   - disagreement: str
   - resolution: str
   - resolution_method: enum (weighted_vote / re_query / human_escalation)

6. **MemoryEntry** - what gets stored in long-term memory
   - content: str
   - source_agent: str
   - query_context: str
   - memory_type: enum (fact / analysis / sentiment / risk)
   - created_at: datetime
   - expires_at: datetime (optional)
   - confidence: float

---

### Task 4: Write database.py (SQLAlchemy models)
**Time: 20 mins**

These define your PostgreSQL tables.

**Concepts you'll learn:**
- SQLAlchemy declarative models
- Table relationships (foreign keys)
- How Python classes map to database tables

**Tables to define:**

1. **Episode** - a complete research session
   - id, query, started_at, completed_at, status
   - total_cost, total_tokens
   - Links to: agent_decisions, knowledge_entries

2. **AgentDecision** - what an agent did during an episode
   - id, episode_id (foreign key), agent_name
   - action, reasoning, result
   - tokens_used, cost, latency_ms

3. **KnowledgeEntry** - facts extracted and stored
   - id, episode_id (foreign key)
   - subject, predicate, object (knowledge triple)
   - e.g., ("NVIDIA", "competes_with", "AMD")
   - confidence, source, created_at

---

### Task 5: Write a smoke test
**Time: 10 mins**

**Concepts you'll learn:**
- pytest basics
- How to verify imports work

**What to test:**
- Import Settings from config and verify it loads
- Create a ResearchQuery instance with test data
- Create an AgentResponse instance
- Create a ResearchReport instance
- Verify all validations pass

**Run with:** `pytest tests/ -v`

---

### Task 6: Git commit
**Time: 5 mins**

```bash
git add -A
git commit -m "Day 1: project skeleton, config, schemas, database models"
```

---

## Success Criteria (how you know Day 1 is done)

- [ ] All folders and files exist
- [ ] `config.py` loads settings from `.env` without errors
- [ ] All Pydantic models in `schemas.py` can be instantiated with test data
- [ ] All SQLAlchemy models in `database.py` are defined
- [ ] `pytest tests/ -v` passes
- [ ] Code is committed to git

## What NOT to do today

- Do NOT write any agent logic
- Do NOT connect to any database
- Do NOT install Docker or run services
- Do NOT build the API or UI
- Do NOT overthink the schemas - they can be changed later

---

## Python concepts you'll use today

| Concept | Where | Quick explanation |
|---------|-------|-------------------|
| Pydantic BaseModel | schemas.py | Define data shapes with automatic validation |
| Enum | schemas.py | Fixed set of choices (like TypeScript union types) |
| Optional[T] | schemas.py | Field that can be None |
| datetime | schemas.py | Timestamp handling |
| BaseSettings | config.py | Load config from .env files |
| SQLAlchemy DeclarativeBase | database.py | Map Python classes to DB tables |
| ForeignKey | database.py | Link between tables |
| pytest | test_models.py | Run tests with assertions |

---

## Tomorrow (Day 2 preview)

You'll take the empty memory files and make them actually connect to
Redis, Qdrant, Postgres, and Mem0. The schemas you build today are
what gets stored/retrieved through those connections.
