HLD Architecture
<img width="932" height="593" alt="image" src="https://github.com/user-attachments/assets/c49a586c-f2c5-48e5-ae25-bf4c990690d2" />



LLD Architecture 
<img width="987" height="670" alt="image" src="https://github.com/user-attachments/assets/cf0c7396-1139-4644-ae98-fa289ce89db2" />



LLD description 

 ### API Layer (FastAPI)
  - **Async background tasks** — `POST /research` returns `202 Accepted` with `job_id` immediately, orchestrator runs as a background task, client polls `GET /research/{job_id}` for
  results
  - **4-gate pre-check before any LLM call** — input guard (12 compiled regex patterns for prompt injection → 400), backpressure (>20 concurrent jobs → 429 with `Retry-After: 30`),
  user daily budget ($5/day via Redis TTL auto-reset), two-tier query cache (exact Redis at ~0.1ms + semantic Qdrant at ~50ms → $0.00 for repeat queries)
  - **Job store** — Redis hashes (`job:{job_id}`) with 2-hour TTL, survives pod restarts, graceful fallback to in-memory dict if Redis is down
  - **Dead letter queue** — Redis Stream (`XADD dead_letter:queries MAXLEN 1000`) quarantines queries where all agents failed for manual investigation

  ### Orchestrator (LangGraph StateGraph)
  - **6-node fixed graph** — `fetch_user_profile → search_memory → create_plan → execute_step ⟲ route_after_step → build_final_report`
  - **Plan-then-execute pattern** — 1 LLM call creates a JSON execution plan, code executes each step mechanically, LLM re-engages only on failure (max 2 replans). Reduces
  orchestration LLM calls from 4-5 (pure ReAct) to 1-2
  - **Three-layer agent guardrail** — (1) code-level filter removes agents when memory has sufficient data (≥3 items at ≥0.85 relevance → skip researcher), (2) LLM plans within
  filtered options, (3) `validate_plan()` removes invalid actions/agent names, enforces `build_report` at end, caps at 5 steps
  - **3-strategy JSON parser** — direct `json.loads` → markdown code fence extraction → regex `\[.*\]` → `default_plan()` fallback. Handles cross-provider LLM output quirks (Claude vs
   Gemini)
  - **State accumulation via LangGraph reducers** — `Annotated[list[AgentResponse], operator.add]` concatenates agent results across steps instead of overwriting. Step 1 adds
  `[researcher, sentiment]`, step 2 adds `[analyst, risk]`, final state has all 4
  - **Dual budget enforcement** — Python-level check between orchestrator steps (coarse) + Redis Lua script check before each LLM call inside agents (atomic, prevents TOCTOU)

  ### Agent Architecture (base.py)
  - **4 specialized agents** — `ResearcherAgent` (data gatherer, runs first), `AnalystAgent` (financial interpreter), `RiskAssessorAgent` (deliberate pessimist, pre-mortem thinking),
  `SentimentAgent` (market mood reader)
  - **Plan-then-execute ReAct loop** — max 5 turns per agent, 4 tools: `create_plan` (structured research plan), `search_web` (Tavily with in-session cache), `search_past_research`
  (Qdrant + Mem0 parallel), `write_analysis` (exit tool that signals completion)
  - **7 cost walls inside each agent:**
    1. **Idempotency check** — Redis lookup before any work; if agent already ran for this session, return cached `AgentResponse` ($0.00)
    2. **Cost-aware model routing** — budget <30% remaining → cheapest fallback model; simple query keywords → cheaper model; ~30% savings
    3. **Token estimation + atomic budget** — estimate input (chars/4) + output tokens per agent type, Redis Lua script does atomic read-check-increment preventing TOCTOU race across
  4 parallel agents
    4. **Prompt compression** — after turn 2, summarize middle messages keeping system prompt + user query + last 3 messages; ~40% token reduction
    5. **Tavily search cache** — in-memory dict scoped per orchestrator run; deduplicates when researcher and analyst search the same query
    6. **Checkpointing** — save turn progress to Redis (TTL 5min) after each tool turn; infrastructure for future crash recovery
    7. **Idempotency store** — cache final `AgentResponse` in Redis (TTL 5min) so retries return cached result
  - **Out-of-band tool context** — LangChain tools return strings (for LLM), but structured `AgentResponse` objects flow through a shared `_tool_context` dict for the orchestrator to
  read

  ### Memory Architecture (6 backends behind a facade)
  - **MemoryManager (facade pattern)** — agents call `memory.parallel_search()` / `memory.store_research()`, never touch backends directly. Swap Qdrant for Pinecone → change one file
  - **Redis (short-term)** — session state, atomic budget counters (Lua scripts), idempotency keys, checkpoints, agent status tracking, job store hashes. TTL-based auto-expiry (1-2
  hours). `SCAN` (not `KEYS`) for production-safe iteration
  - **Qdrant (long-term)** — research vectors in `research_knowledge` collection with topic-filtered search (`FieldCondition` + `MatchAny`). Contextual Retrieval: agent name + query +
   topics prepended to text before embedding so vectors encode domain context. Dual threshold: 0.5 with topic filter, 0.7 without
  - **Qdrant (semantic cache)** — separate `query_cache` collection. Cosine threshold ≥0.92, max age 1 hour. Combined with exact Redis cache (MD5 hash, ~0.1ms) for two-tier
  deduplication
  - **Mem0 (semantic facts)** — auto-deduplicating fact store via internal LLM. "NVIDIA P/E is 65" → later "P/E is 52" updates the old fact instead of duplicating. Each agent is a
  separate Mem0 user (multi-tenant isolation). Dimension mismatch recovery with one-time filesystem cleanup
  - **PostgreSQL (episodic)** — decision audit trail via SQLAlchemy ORM. Optional — graceful skip if unavailable. Enables FCRA-compliant reconstruction of what the system knew at any
  point
  - **Neo4j (graph)** — user profiles as knowledge graph with temporal versioning. Old edges get `to` date (closed), new edges stay active (`to: null`). Fetched once by orchestrator,
  shared with all 4 agents
  - **Cross-agent memory bridging** — gatherers (researcher, sentiment) write to Qdrant + Mem0 in step 1; analyzers (analyst, risk) search the same memory in step 2 and find fresh
  data. Eventual consistency enforced by plan ordering

  ### Report Generation
  - **Conflict detection (no LLM)** — keyword-based stance classification (10 bullish + 11 bearish signals), 1.5x ratio threshold to classify stance, pairwise comparison across all
  agents (O(n²), n=4). Severity: HIGH (both confident, avg >0.8), MEDIUM (avg >0.6), LOW. Confidence gap >0.3 flagged separately
  - **Quality assessment (no LLM)** — grade by agent count (4=HIGH, 3=MEDIUM, 2=LOW, 1=MINIMAL, 0=FAILED), downgrade if avg confidence <0.4, disclaimers for failed agents and
  high-severity conflicts
  - **LLM synthesis (1 call)** — merge 4 agent outputs + conflict summary + quality assessment into structured report (Executive Summary → Key Findings → Financial Analysis →
  Sentiment → Risks → Conclusion). Fallback: concatenate raw outputs if synthesis LLM fails

  ### Resilience
  - **Typed error hierarchy** — `AgentError` base with `LLMTimeoutError` (retry once), `LLMRateLimitError` (exponential backoff 2^attempt), `LLMRefusalError` (don't retry),
  `MemoryUnavailableError` (continue degraded), `BudgetExceededError` (hard stop, return partial)
  - **Circuit breaker** — CLOSED → 3 failures → OPEN (skip agent) → 60s timeout → HALF_OPEN (test one call) → success → CLOSED
  - **Graceful degradation** — every dependency categorized as essential (LLM) or optional (Postgres, Neo4j, Qdrant, Redis). Optional backends down → agents work with reduced
  functionality. Philosophy: partial results > complete failure
  - **3-layer prompt injection defense** — (1) 12 compiled regex patterns at API boundary → 400, (2) `<user_query>` XML tags mark trust boundary in prompts, (3) structured JSON output
   contract with explicit refusal thresholds (only refuse genuinely illegal requests, not controversial financial topics)

  ### Security
  - **Input sanitization** — regex guard catches known injection patterns before any LLM call
  - **Trust boundary** — user input wrapped in XML delimiters, system prompt instructs LLM to treat content inside as data, never as instructions
  - **Structured output contract** — LLM must return `{"status": "completed", "analysis": "..."}` or `{"status": "refused", "refusal_reason": "..."}`, eliminates false-positive
  refusals on financial topics

