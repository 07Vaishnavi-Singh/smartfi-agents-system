# Knowledge Graph Architecture — Neo4j

## Overview

Four subgraphs in a single Neo4j instance, each serving a distinct purpose:

```
User query
     │
     ├──→ Graph 1 (Profile)         → WHO is this user?
     ├──→ Graph 2 (Preferences)     → WHAT has this user been doing?
     ├──→ Graph 3 (Asset Relations) → HOW are entities connected?
     ├──→ Graph 4 (Source Cred)     → HOW much to trust each source?
     ├──→ Qdrant + Mem0             → Past research (existing, unchanged)
     │
     ▼
  Agents receive all as context → personalized, ecosystem-aware, trust-scored response
```

---

## Graph 1: User Profile (Slow-Moving, Append-Only)

- **Source:** User-provided (onboarding form + profile edits)
- **Update frequency:** Days / months / years
- **Trust level:** 100% — user explicitly stated this
- **Temporal versioning:** Every edge has `{from: date, to: date|null}`, `to: null` = currently active

### Onboarding Questions

1. Age range (18-25, 26-35, 36-50, 50+)
2. Gender
3. Location (city/country)
4. Financial goal(s) — Wealth building, Retirement, Passive income, Trading, Save for X
5. Risk tolerance — Conservative, Moderate, Aggressive
6. Investment horizon — Short (<1yr), Medium (1-5yr), Long (5+yr)
7. Monthly investable amount (range)
8. Existing investments — Stocks, Mutual funds, Crypto, Gold, FDs, None

### Node Types

```cypher
(:User {id, name, created_at})
(:AgeRange {range: "26-35"})
(:Gender {value: "M"})
(:Location {city: "Bangalore", country: "India"})
(:Goal {type: "Wealth Building", target_amount?, target_date?})
(:RiskProfile {level: "moderate"})
(:Horizon {type: "long", years: "5+"})
(:Budget {range: "20-50k", currency: "INR"})
(:AssetClass {type: "Mutual Funds"})
```

### Relationships

```cypher
(User)-[:AGE_RANGE {from: "2026-01", to: null}]->(AgeRange)
(User)-[:GENDER]->(Gender)
(User)-[:LOCATED_IN {from: "2026-01", to: null}]->(Location)
(User)-[:HAS_GOAL {from: "2026-01", to: null, priority: 1}]->(Goal)
(User)-[:RISK_TOLERANCE {from: "2026-01", to: null}]->(RiskProfile)
(User)-[:HORIZON {from: "2026-01", to: null}]->(Horizon)
(User)-[:BUDGET {from: "2026-01", to: null}]->(Budget)
(User)-[:ALREADY_INVESTS_IN]->(AssetClass)
```

### Update Pattern (Append, Never Overwrite)

```cypher
// User moves from Mumbai to Bangalore:

// Step 1: Close old edge
MATCH (u:User {id: $uid})-[r:LOCATED_IN]->(old:Location)
WHERE r.to IS NULL
SET r.to = date()

// Step 2: Create new edge
MATCH (u:User {id: $uid})
MERGE (loc:Location {city: "Bangalore", country: "India"})
CREATE (u)-[:LOCATED_IN {from: date(), to: null}]->(loc)
```

### Query Pattern (Get Current State)

```cypher
// Always filter by `to IS NULL` to get active profile
MATCH (u:User {id: $uid})-[r]->(n)
WHERE r.to IS NULL
RETURN type(r) AS attribute, properties(n) AS value
```

---

## Graph 2: Learned Preferences (Fast-Moving, Append-Only)

- **Source:** LLM-extracted from every conversation
- **Update frequency:** Every conversation turn
- **Trust level:** Best-effort — LLM-inferred, may be noisy
- **Tracking:** Edges have `{count, first_seen, last_seen}`, count increments on repeat

### Node Types

```cypher
(:Asset {name: "NVIDIA", ticker: "NVDA", type: "stock"})
(:Sector {name: "Tech"})
(:RiskType {type: "Regulation"})
(:ResponseStyle {type: "data_heavy"})
(:Pattern {type: "valuation_focused"})
(:Action {type: "buy"})

// Factual nodes (from agent research outputs)
(:Company {name: "NVIDIA"})
(:Person {name: "Jensen Huang"})
(:Event {name: "China Export Ban", date: "2025-01"})
```

### Relationships — User Preferences

```cypher
(User)-[:INTERESTED_IN {count: 5, first_seen: "2026-03-01", last_seen: "2026-03-24"}]->(Asset)
(User)-[:AVOIDS {reason: "too volatile"}]->(AssetClass)
(User)-[:CONCERNED_ABOUT {count: 3}]->(RiskType)
(User)-[:PREFERS_SECTOR {count: 4}]->(Sector)
(User)-[:PREFERS_STYLE]->(ResponseStyle)
(User)-[:QUERY_PATTERN]->(Pattern)
(User)-[:RECENT_INTENT]->(Action)  // buy, sell, hold, research, compare
```

### Relationships — Factual Knowledge (From Agent Outputs)

```cypher
(Company)-[:COMPETES_WITH]->(Company)        // NVIDIA <> AMD
(Company)-[:BELONGS_TO]->(Sector)            // NVIDIA -> Semiconductors
(Company)-[:HAS_CEO]->(Person)               // NVIDIA -> Jensen Huang
(Company)-[:RISK_FACTOR]->(Event)            // NVIDIA -> China Export Ban
(Asset)-[:INVERSE_CORRELATION]->(Asset)      // Gold <> USD
(Event)-[:IMPACTS]->(Sector)                 // Rate Hike -> Tech
```

### Update Pattern (Increment Counts)

```cypher
// User asks about NVIDIA again:
MATCH (u:User {id: $uid})-[r:INTERESTED_IN]->(a:Asset {name: "NVIDIA"})
SET r.count = r.count + 1, r.last_seen = date()

// First time asking about NVIDIA:
MATCH (u:User {id: $uid})
MERGE (a:Asset {name: "NVIDIA"})
CREATE (u)-[:INTERESTED_IN {count: 1, first_seen: date(), last_seen: date()}]->(a)
```

### Query Pattern (Get Top Preferences)

```cypher
// Top interests
MATCH (u:User {id: $uid})-[r:INTERESTED_IN]->(a)
RETURN a.name, r.count ORDER BY r.count DESC LIMIT 5

// Recent concerns
MATCH (u:User {id: $uid})-[r:CONCERNED_ABOUT]->(risk)
RETURN risk.type, r.count ORDER BY r.last_seen DESC
```

---

## LLM Extraction — How Graph 2 Gets Populated

### From User Queries (Before Agents Run)

```python
QUERY_EXTRACTION_PROMPT = """
From this investment query, extract:
1. Assets mentioned (stocks, commodities, crypto, etc.)
2. Concerns (risk types: regulation, competition, valuation, macro)
3. Intent (buy, sell, hold, research, compare)
4. Sector signals (tech, healthcare, finance, etc.)
5. Response style hints (wants data, wants simple explanation, wants comparison)

Query: {query}
Return JSON:
{
  "assets": [{"name": "...", "type": "stock|crypto|commodity"}],
  "concerns": ["regulation"],
  "intent": "buy",
  "sectors": ["tech"],
  "style_hints": ["data_heavy"]
}
"""
```

### From Agent Outputs (After Agents Finish)

```python
FACT_EXTRACTION_PROMPT = """
From this financial analysis, extract factual relationships as triples:
(subject, relationship, object)

Examples:
- (NVIDIA, competes_with, AMD)
- (NVIDIA, has_ceo, Jensen Huang)
- (Gold, inverse_correlation, USD)

Analysis: {agent_output}
Return JSON: {"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}
"""
```

---

## Integration Points in Existing Code

### Point 1: Extract & Store (Before Agents Run)

**Where:** `MemoryManager.parallel_search()` / `MemoryManager.create_session()`

```python
# Extract intent from query → upsert into Graph 2
await self.graph.track_query(user_id, query, topics)
```

### Point 2: Retrieve & Inject (Alongside Existing Memory Search)

**Where:** `MemoryManager.parallel_search()`

```python
# Add Neo4j query alongside Qdrant + Mem0
qdrant_results, mem0_results, user_profile, user_prefs = await asyncio.gather(
    self.long_term.search(query, topics=topics),
    self.semantic.search(query),
    self.graph.get_user_profile(user_id),       # Graph 1
    self.graph.get_user_preferences(user_id),   # Graph 2
)
```

### Point 3: Learn & Update (After Agents Finish)

**Where:** `MemoryManager.store_research()`

```python
# Extract factual triples from agent output → store in Graph 2
await self.graph.store_facts(content, agent_name)
```

---

## How Agents Receive Graph Context

Injected into the agent system prompt as a labeled section:

```
[USER PROFILE — from Graph 1]
Age: 26-35 | Location: Bangalore, India
Risk: Moderate | Horizon: 5+ years
Budget: 20-50k INR/month
Goal: Wealth Building
Existing investments: Mutual Funds

[USER PREFERENCES — from Graph 2]
High interest: NVIDIA (5 queries), Gold (3 queries)
Avoids: Crypto
Key concern: Regulatory risk
Response style: Prefers data-heavy analysis
Recent intent: Buy

[BEHAVIORAL NOTE]
User STATES moderate risk tolerance. Behavioral signals suggest
higher risk appetite (frequently asks about volatile assets).
Respect stated preference but acknowledge the pattern if relevant.
```

### How This Changes Each Agent

| Agent | Without KG | With KG |
|-------|-----------|---------|
| **Researcher** | Generic facts | Prioritizes topics user tracks |
| **Analyst** | Standard metrics | Frames analysis for user's budget/horizon |
| **Risk Assessor** | All risks equally | Leads with user's top concerns (regulation) |
| **Sentiment** | General sentiment | Focuses on assets user is interested in |

---

## Behavioral Inference (Advanced — Phase 2)

The system can detect mismatches between stated and observed behavior:

```cypher
// User said "moderate" but keeps asking about crypto and meme stocks
MATCH (u:User {id: $uid})-[:RISK_TOLERANCE]->(r:RiskProfile {level: "moderate"})
MATCH (u)-[i:INTERESTED_IN]->(a:Asset)
WHERE a.type IN ["crypto", "meme_stock"]
WITH u, r, count(i) AS risky_count
WHERE risky_count > 3
// Flag: stated=moderate, observed=aggressive
```

Use this to add a behavioral note to agent context, NOT to override the user's stated preference.

---

## Infrastructure

### Docker Compose Addition

```yaml
neo4j:
  image: neo4j:5-community
  ports:
    - "7474:7474"   # Browser UI (explore graph visually)
    - "7687:7687"   # Bolt protocol (Python driver connects here)
  environment:
    NEO4J_AUTH: neo4j/agent_pass
  volumes:
    - neo4j_data:/data
  restart: unless-stopped
```

### Python Dependency

```
neo4j  # async support built-in via AsyncDriver
```

### New File

```
src/investment_research_system/memory/
  └── graph.py    # Neo4j client: store/query/update for both subgraphs
```

### Config Addition (config.py)

```python
neo4j_uri: str = "bolt://localhost:7687"
neo4j_user: str = "neo4j"
neo4j_password: str = "agent_pass"
```

---

## Graph 3: Asset/Entity Relationship Graph

The #1 use case for KGs in production. Used by Bloomberg, Goldman Sachs, JPMorgan, Palantir, Google.

- **Source:** LLM-extracted from agent research outputs + can be seeded from public data (SEC filings, company databases)
- **Update frequency:** After every research run
- **Trust level:** Medium — LLM-extracted, but improves with repeated confirmation across queries
- **Purpose:** When user asks about NVIDIA, agents automatically discover and research AMD, TSMC, supply chain risks — ecosystem-aware research instead of single-stock isolation

### Node Types

```cypher
(:Company {name: "NVIDIA", ticker: "NVDA", market_cap: "3.2T"})
(:Company {name: "AMD", ticker: "AMD"})
(:Company {name: "TSMC", ticker: "TSM"})
(:Person {name: "Jensen Huang", role: "CEO"})
(:Sector {name: "Semiconductors"})
(:Product {name: "H100", type: "GPU"})
(:Event {name: "China Export Ban", date: "2025-01", type: "regulatory"})
(:Filing {type: "10-K", date: "2025-03", company: "NVIDIA"})
(:Index {name: "S&P 500"})
```

### Relationships

```cypher
// Corporate structure
(Company)-[:SUBSIDIARY_OF]->(Parent)               // Instagram -> Meta
(Company)-[:ACQUIRED {date, amount}]->(Company)     // Microsoft -> Activision
(Person)-[:CEO_OF {from, to}]->(Company)            // Jensen Huang -> NVIDIA
(Person)-[:BOARD_MEMBER_OF]->(Company)
(Company)-[:FILED]->(Filing)                        // NVIDIA -> 10-K

// Market relationships
(Company)-[:COMPETES_WITH]->(Company)               // NVIDIA <> AMD
(Company)-[:SUPPLIED_BY]->(Company)                 // NVIDIA <- TSMC
(Company)-[:PARTNER_OF]->(Company)                  // NVIDIA <> Microsoft (Azure GPU)
(Company)-[:CUSTOMER_OF]->(Company)                 // Meta -> NVIDIA (GPU buyer)
(Company)-[:BELONGS_TO]->(Sector)                   // NVIDIA -> Semiconductors
(Company)-[:LISTED_IN]->(Index)                     // NVIDIA -> S&P 500

// Product relationships
(Company)-[:MAKES]->(Product)                       // NVIDIA -> H100
(Product)-[:COMPETES_WITH]->(Product)               // H100 <> MI300X

// Risk/event relationships
(Event)-[:IMPACTS]->(Company)                       // China Export Ban -> NVIDIA
(Event)-[:IMPACTS]->(Sector)                        // Rate Hike -> Tech
(Company)-[:EXPOSED_TO]->(Event)                    // NVIDIA -> China Export Ban
```

### How It Makes Agents Smarter

```python
# In parallel_search(), before agents run:
related = await graph.get_related_entities("NVIDIA", depth=2)
# Returns:
# {
#   "competitors": ["AMD", "Intel"],
#   "suppliers": ["TSMC", "Samsung Foundry"],
#   "partners": ["Microsoft", "AWS"],
#   "risks": ["China Export Ban", "Antitrust"],
#   "sector": "Semiconductors"
# }

# Researcher's query becomes enriched:
# "NVIDIA analysis. Also consider: AMD (competitor), TSMC (key supplier,
#  single point of failure), China Export Ban (regulatory risk affecting
#  ~25% of revenue)"
```

### Query Patterns

```cypher
// Get full ecosystem for a company (2-hop)
MATCH (c:Company {name: $name})-[r1]-(connected)-[r2]-(second_hop)
WHERE type(r1) IN ["COMPETES_WITH", "SUPPLIED_BY", "PARTNER_OF", "EXPOSED_TO"]
RETURN c, r1, connected, r2, second_hop

// Find supply chain risk — who depends on TSMC?
MATCH (c:Company)-[:SUPPLIED_BY]->(tsmc:Company {name: "TSMC"})
RETURN c.name AS dependent_company

// Find all companies exposed to a specific event
MATCH (c:Company)-[:EXPOSED_TO]->(e:Event {name: $event})
RETURN c.name, c.ticker

// Competitive landscape
MATCH (c:Company {name: $name})-[:BELONGS_TO]->(s:Sector)<-[:BELONGS_TO]-(peer:Company)
WHERE peer.name <> c.name
RETURN peer.name, peer.ticker ORDER BY peer.market_cap DESC
```

### Population Strategy: Schema-Guided LLM Extraction

This is the core design pattern — **developer pre-defines the allowed relationship types (edges), LLM fills in the nodes using live web data**. This sits between fully manual KGs and fully LLM-generated chaos:

```
Fully manual (traditional)     Schema-guided (our approach)     Fully LLM (chaos)
──────────────────────────     ────────────────────────────     ─────────────────
Humans define everything       Human defines edges,             LLM invents everything
                               LLM fills nodes
Rigid, expensive               Structured + flexible            Noisy, inconsistent
Bloomberg-style                Production-grade ✅               Research/prototype only
```

#### Step 1: Pre-Define Allowed Relationship Types (One-Time, Developer)

```python
# This is the "schema" — the LLM MUST only use these relationship types
ALLOWED_RELATIONSHIPS = [
    "COMPETES_WITH",
    "SUPPLIED_BY",
    "PARTNER_OF",
    "CEO_OF",
    "BELONGS_TO_SECTOR",
    "EXPOSED_TO_RISK",
    "SUBSIDIARY_OF",
    "MAKES_PRODUCT",
    "CUSTOMER_OF",
    "LISTED_IN",
]
```

#### Step 2: Define Search Queries Per Relationship Type

```python
# Each relationship type maps to a Tavily search query template
RELATIONSHIPS_TO_FILL = [
    ("COMPETES_WITH",      "{company} competitors 2026"),
    ("SUPPLIED_BY",        "{company} supply chain key suppliers"),
    ("PARTNER_OF",         "{company} strategic partnerships"),
    ("CEO_OF",             "{company} CEO leadership team"),
    ("BELONGS_TO_SECTOR",  "{company} industry sector classification"),
    ("EXPOSED_TO_RISK",    "{company} major risks regulatory threats"),
    ("MAKES_PRODUCT",      "{company} main products revenue segments"),
    ("CUSTOMER_OF",        "{company} biggest customers revenue"),
]
```

#### Step 3: LLM Extraction Prompt (Schema-Constrained)

```python
GRAPH_EXTRACTION_PROMPT = """
You are a financial knowledge graph builder.

You MUST ONLY use these relationship types:
{allowed_relationships}

For the company "{company}", extract entities for this specific relationship:
  Relationship: {relationship_type}

Search results from the web:
{tavily_results}

Rules:
- Only extract what is EXPLICITLY stated in the search results
- Do NOT guess or infer relationships
- Include confidence score (0-1) based on source quality
- Include date/recency of the information
- If search results don't contain relevant info, return empty objects list

Return JSON:
{{
  "subject": "{company}",
  "relationship": "{relationship_type}",
  "objects": [
    {{
      "name": "AMD",
      "properties": {{"ticker": "AMD", "context": "direct GPU competitor"}},
      "confidence": 0.95,
      "source": "bloomberg.com"
    }}
  ]
}}
"""
```

The key constraint: **"You MUST ONLY use these relationship types"** — this prevents the LLM from inventing random edge types, ensuring schema consistency.

#### Step 4: Multi-Call Loop (Parallel Execution)

```python
async def build_entity_graph(company: str, graph_client, tavily_client, llm):
    """Schema-guided multi-LLM-call loop to populate graph for a company.

    Fires all Tavily searches in parallel, then all LLM extractions in parallel.
    8 sequential calls = ~40 seconds. 8 parallel calls = ~5 seconds.
    """

    # Step 1: Fire ALL Tavily searches in parallel
    search_tasks = [
        tavily_client.search(query.format(company=company))
        for _, query in RELATIONSHIPS_TO_FILL
    ]
    all_search_results = await asyncio.gather(*search_tasks)

    # Step 2: Fire ALL LLM extractions in parallel
    extraction_tasks = [
        llm.invoke(GRAPH_EXTRACTION_PROMPT.format(
            company=company,
            relationship_type=rel_type,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            tavily_results=results,
        ))
        for (rel_type, _), results in zip(RELATIONSHIPS_TO_FILL, all_search_results)
    ]
    all_extracted = await asyncio.gather(*extraction_tasks)

    # Step 3: Batch upsert into Neo4j
    for extracted in all_extracted:
        parsed = parse_extraction(extracted)
        for entity in parsed.objects:
            await graph_client.upsert_relationship(
                subject=company,
                relationship=parsed.relationship,
                object_=entity.name,
                properties={
                    **entity.properties,
                    "confidence": entity.confidence,
                    "source": entity.source,
                    "extracted_at": datetime.now().isoformat(),
                },
            )
```

#### Step 5: When Does the Loop Run?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Population Triggers                           │
│                                                                  │
│  Trigger A: Lazy Load (on first query about a company)          │
│  ──────────────────────────────────────────────────              │
│  User asks about NVIDIA for the first time                      │
│  → Graph has no nodes for NVIDIA                                │
│  → Trigger build_entity_graph("NVIDIA")                         │
│  → ~5 seconds (parallel), cached in Neo4j                       │
│  → All future NVIDIA queries are instant (read from graph)      │
│                                                                  │
│  Trigger B: Background Enrichment (after every research run)    │
│  ──────────────────────────────────────────────────              │
│  Research run completes for NVIDIA                              │
│  → Agent outputs mention AMD, TSMC (new entities discovered)    │
│  → Background task: build_entity_graph("AMD")                   │
│  → Background task: build_entity_graph("TSMC")                  │
│  → Graph grows organically — connected entities get enriched    │
│                                                                  │
│  Trigger C: Batch Seed (bootstrap on system startup)            │
│  ──────────────────────────────────────────────────              │
│  On first deployment, seed top 50 companies:                    │
│  → for company in TOP_50: build_entity_graph(company)           │
│  → ~5 min total with parallel calls                             │
│  → System starts with a rich base graph                         │
│                                                                  │
│  Trigger D: Staleness Refresh (periodic)                        │
│  ──────────────────────────────────────────────────              │
│  Cron job checks: any company node older than 7 days?           │
│  → Re-run build_entity_graph() for stale entries                │
│  → Ensures data stays fresh (CEOs change, acquisitions happen)  │
│                                                                  │
│  Best approach: A + B combined.                                  │
│  Lazy load on first query, background-enrich connected entities. │
└─────────────────────────────────────────────────────────────────┘
```

#### Neo4j Upsert Pattern (Idempotent)

```cypher
// MERGE = create if not exists, match if exists (idempotent)
// This is safe to call repeatedly — won't create duplicates

MERGE (c1:Company {name: $subject})
MERGE (c2:Company {name: $object})
MERGE (c1)-[r:COMPETES_WITH]->(c2)
SET r.confidence = $confidence,
    r.source = $source,
    r.extracted_at = datetime(),
    r.updated_at = datetime()
```

#### Cost Analysis

```
Per company graph build:
  8 Tavily searches    × $0.001 each  = $0.008
  8 LLM calls (flash)  × $0.001 each  = $0.008
  Total: ~$0.016 per company

  Top 50 seed: ~$0.80

  With parallel execution: ~5 seconds per company
  Top 50 seed: ~5 minutes (10 companies at a time)
```

#### Refresh Strategy (Keeping Data Fresh)

```python
async def refresh_stale_entities(graph_client, tavily_client, llm, max_age_days=7):
    """Find and refresh company nodes older than max_age_days."""
    stale = await graph_client.query("""
        MATCH (c:Company)
        WHERE c.last_refreshed < datetime() - duration({days: $max_age})
        RETURN c.name ORDER BY c.last_refreshed ASC LIMIT 10
    """, max_age=max_age_days)

    for company in stale:
        await build_entity_graph(company.name, graph_client, tavily_client, llm)
        # Update refresh timestamp
        await graph_client.query("""
            MATCH (c:Company {name: $name})
            SET c.last_refreshed = datetime()
        """, name=company.name)
```

#### Why This Is Production-Grade

| Concern | How Schema-Guided Extraction Handles It |
|---------|----------------------------------------|
| **Schema consistency** | Pre-defined edges — LLM can't invent random relationship types |
| **Data quality** | Tavily gives real-time web data, not LLM hallucinations |
| **Confidence tracking** | Each extraction includes confidence score + source URL |
| **Staleness** | `extracted_at` + `last_refreshed` timestamps + periodic refresh |
| **Cost** | ~$0.016 per company, cached in Neo4j — one-time cost per entity |
| **Latency** | Parallel calls: ~5 seconds. Cached reads: <10ms |
| **Idempotency** | MERGE-based upserts — safe to re-run without duplicates |
| **Entity resolution** | Schema constraints + MERGE reduce "Apple company vs fruit" problems |
| **Observability** | Every extraction tagged with source, confidence, timestamp |

#### How Real Companies Do This

- **Bloomberg** — predefined ontology (thousands of relationship types), automated extraction from SEC filings + news feeds
- **Diffbot** — crawls the web, uses ML to extract entities into a predefined schema (10B+ entities)
- **Google Knowledge Graph** — predefined schema (inherited from Freebase), automated extraction fills it at scale
- **Palantir** — customer-defined ontologies, data pipelines populate nodes automatically

Our approach is the same pattern at portfolio scale — predefined schema, LLM-powered extraction, live web data via Tavily.

---

## Graph 4: Source Credibility Tracking

Used by Reuters, AP, financial firms, and any production RAG system. Required for compliance in finance.

- **Source:** Built from Tavily search results + tracked over time
- **Update frequency:** After every research run
- **Trust level:** Starts neutral, builds confidence over time through repeated citation
- **Purpose:** Populate the existing `reliability_score` field in the `Source` schema with real data instead of defaults

### Node Types

```cypher
(:Source {domain: "bloomberg.com", name: "Bloomberg"})
(:Source {domain: "seekingalpha.com", name: "Seeking Alpha"})
(:SourceCategory {type: "Tier 1 — Wire Services"})        // Reuters, AP, Bloomberg
(:SourceCategory {type: "Tier 2 — Major Financial"})       // WSJ, FT, CNBC
(:SourceCategory {type: "Tier 3 — Analyst/Opinion"})       // Seeking Alpha, Motley Fool
(:SourceCategory {type: "Tier 4 — Social/Unverified"})     // Reddit, Twitter, blogs
(:Claim {text: "NVIDIA revenue grew 94% YoY", date: "2026-03"})
(:Report {id: "report_uuid", query: "NVIDIA analysis"})
```

### Relationships

```cypher
// Source classification
(Source)-[:CATEGORY]->(SourceCategory)                     // Bloomberg -> Tier 1
(Source)-[:SPECIALIZES_IN]->(Sector)                       // Bloomberg -> Finance

// Citation tracking
(Source)-[:CITED_IN {date}]->(Report)                      // Bloomberg cited in report X
(Claim)-[:SOURCED_FROM]->(Source)                          // "94% growth" came from Bloomberg
(Claim)-[:VERIFIED_BY {count: 3}]->(VerificationStatus)   // cross-referenced by 3 sources

// Credibility scoring (updated over time)
(Source)-[:CREDIBILITY {
  score: 0.92,                  // 0-1 scale
  citations_total: 45,          // how often cited
  citations_recent_30d: 12,     // recent activity
  last_cited: "2026-03-24",
  accuracy_tracked: 0.88        // when verifiable claims were checked
}]->(CredibilityScore)

// Track when sources got it wrong
(Source)-[:MADE_CLAIM]->(Claim)
(Claim)-[:OUTCOME {result: "correct" | "incorrect" | "unverified"}]->(Outcome)
```

### How It Plugs Into Existing Code

Your `Source` schema already has `reliability_score`:

```python
class Source(BaseModel):
    title: str
    url: str | None = None
    source_type: SourceType
    reliability_score: float  # 0.0-1.0  ← THIS FIELD, currently defaulted
```

Wire it up:

```python
# In researcher.py extract_sources():
for source in tavily_results:
    domain = extract_domain(source.url)  # "bloomberg.com"
    credibility = await graph.get_source_credibility(domain)
    source.reliability_score = credibility.score  # now real data!

# In store_research() (after agents finish):
for source in agent_response.sources:
    await graph.track_citation(source.url, report_id)
    # Credibility score gradually updates based on citation frequency
```

### Credibility Score Calculation

```python
def calculate_credibility(source_node) -> float:
    base_score = TIER_SCORES[source_node.category]  # Tier 1: 0.9, Tier 2: 0.75, etc.
    citation_boost = min(source_node.citations_total * 0.005, 0.1)  # max +0.1
    accuracy_factor = source_node.accuracy_tracked or base_score
    recency_factor = 1.0 if cited_in_last_30_days else 0.95

    return min(base_score + citation_boost, 1.0) * accuracy_factor * recency_factor
```

### Initial Tier Seeding

```cypher
// Seed known sources with baseline credibility
CREATE (s:Source {domain: "reuters.com", name: "Reuters"})
  -[:CATEGORY]->(:SourceCategory {type: "Tier 1"})
CREATE (s)-[:CREDIBILITY {score: 0.95, citations_total: 0}]->(:CredibilityScore)

// Tier defaults:
// Tier 1 (Wire/Official):    0.90-0.95  — Reuters, AP, Bloomberg, SEC.gov
// Tier 2 (Major Financial):  0.75-0.85  — WSJ, FT, CNBC, Yahoo Finance
// Tier 3 (Analyst/Opinion):  0.55-0.70  — Seeking Alpha, Motley Fool, analyst blogs
// Tier 4 (Social/Unverified): 0.20-0.40 — Reddit, Twitter, unknown blogs
```

### Query Patterns

```cypher
// Get credibility for a source
MATCH (s:Source {domain: $domain})-[c:CREDIBILITY]->(score)
RETURN c.score, c.citations_total, c.accuracy_tracked

// Find most-cited sources across all reports
MATCH (s:Source)-[r:CITED_IN]->(report)
RETURN s.name, s.domain, count(r) AS citations ORDER BY citations DESC LIMIT 10

// Cross-verify a claim — how many sources back it?
MATCH (claim:Claim {text: $claim_text})-[:SOURCED_FROM]->(s:Source)
RETURN count(s) AS source_count, collect(s.name) AS sources

// Find sources that were wrong
MATCH (s:Source)-[:MADE_CLAIM]->(c:Claim)-[:OUTCOME]->(o {result: "incorrect"})
RETURN s.name, count(c) AS wrong_claims ORDER BY wrong_claims DESC
```

### Why This Matters (Interview Talking Point)

> "In financial AI, you can't just retrieve and generate — you need provenance. Our source credibility graph tracks every citation across all research runs. Tier 1 sources like Bloomberg start with a 0.95 baseline, while unknown blogs start at 0.3. Over time, the scores adjust based on citation frequency and accuracy tracking. This feeds directly into the `reliability_score` field on every source in every report, so the user knows exactly how much to trust each claim."

---

## Implementation Priority

| Priority | Graph | Why |
|----------|-------|-----|
| **1** | Graph 1 — User Profile | Core personalization, onboarding flow |
| **2** | Graph 2 — Learned Preferences | Dynamic personalization from conversations |
| **3** | Graph 3 — Asset Relationships | Biggest research quality improvement, #1 production KG use case |
| **4** | Graph 4 — Source Credibility | Wires into existing `reliability_score` field, production compliance |

All 4 live in a single Neo4j instance, queried in parallel during `MemoryManager.parallel_search()`.

---

## Graph 1 vs Graph 2 — Summary

| | Graph 1 (Profile) | Graph 2 (Preferences) |
|---|---|---|
| **Source** | User fills form | LLM extracts from chats |
| **Updates** | Days/months/years | Every conversation |
| **Accuracy** | 100% (user-provided) | Noisy (LLM-inferred) |
| **Versioning** | Temporal (`from`/`to` dates) | Count-based (`count`, `last_seen`) |
| **On conflict** | Graph 1 wins | Treated as signal, not truth |
| **If deleted** | User re-onboards | System relearns over time |
| **Query cost** | Cheap (fixed structure) | Heavier (growing graph) |
