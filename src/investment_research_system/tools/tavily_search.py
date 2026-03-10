"""Tavily web search tool for live information retrieval.

Wraps the Tavily API to give agents access to current web data.
Used when memory (Qdrant/Mem0) doesn't have fresh enough info.

PYTHON CONCEPT — WRAPPER PATTERN:
Same idea as your memory classes: wrap an external client, expose
only the methods your agents need. If Tavily changes their API,
you update ONE file.

TS equivalent: a service class around an SDK client.
Rust equivalent: a newtype wrapper with impl methods.
"""

import logging
from datetime import datetime

from tavily import TavilyClient

from investment_research_system.models.schemas import Source, SourceType

logger = logging.getLogger(__name__)


class TavilySearch:
    """Web search tool backed by Tavily's AI search API.

    Tavily is purpose-built for AI agents — it returns clean,
    summarized results instead of raw HTML. Think of it as
    "Google but the results are already parsed for you."

    PYTHON CONCEPT — Literal types in Tavily's API:
    Tavily uses Literal["basic", "advanced"] for search_depth.
    This is like TS union types: type SearchDepth = "basic" | "advanced"
    Python's Literal does the same thing — restricts to specific values.
    """

    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        topic: str = "finance",
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict:
        """Search the web and return structured results.

        Args:
            query: Natural language search query.
            max_results: How many results to return (1-20).
            search_depth: "basic" (fast, cheaper) or "advanced" (deeper, costs more).
            topic: "finance" optimizes for financial sources.
                   Also: "general", "news".
            time_range: Filter by recency — "day", "week", "month", "year".
                        None = no time filter.
            include_domains: Only search these domains (e.g., ["reuters.com"]).
            exclude_domains: Skip these domains.

        Returns:
            Dict with "results" (list of articles) and "answer" (AI summary).

        PYTHON CONCEPT — keyword arguments with defaults:
        All params after `query` have defaults, so you can call:
            search("NVIDIA stock")                    # uses all defaults
            search("NVIDIA stock", max_results=10)    # override one
            search("NVIDIA stock", topic="news", time_range="week")  # override many
        TS equivalent: function search(query: string, opts?: { maxResults?: number, ... })
        Python doesn't need an options object — named params ARE the options.
        """
        logger.info("Tavily search: %s (depth=%s, topic=%s)", query, search_depth, topic)

        raw = self.client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer="basic",
            # ^ include_answer makes Tavily return an AI-generated summary
            #   of all results. Useful for agents to get a quick overview.
        )

        # Transform raw Tavily response into our format
        results = []
        for item in raw.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "published_date": item.get("published_date"),
            })

        return {
            "query": query,
            "answer": raw.get("answer", ""),
            "results": results,
            "result_count": len(results),
            "searched_at": datetime.now().isoformat(),
        }

    def search_as_sources(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        topic: str = "finance",
        time_range: str | None = None,
    ) -> list[Source]:
        """Search and return results as Source schema objects.

        Convenience method — agents need Source objects for their AgentResponse.
        This converts Tavily results directly into the schema your system uses.

        PYTHON CONCEPT — method reuse:
        This method calls self.search() and transforms the output.
        Don't duplicate logic — call the method that already does the work.
        TS: same pattern, just this.search() instead of self.search().
        """
        raw = self.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            time_range=time_range,
        )

        return [
            Source(
                title=result["title"],
                url=result["url"],
                source_type=SourceType.web,
                reliability_score=min(result["score"], 1.0),
                # ^ Tavily scores can occasionally exceed 1.0,
                #   but our schema constrains to [0.0, 1.0].
            )
            for result in raw["results"]
        ]

    def search_news(self, query: str, max_results: int = 5, time_range: str = "week") -> dict:
        """Shortcut for news-specific searches.

        Pre-configures topic="news" and defaults to last week.
        Agents call this when they need recent headlines, not deep research.
        """
        return self.search(
            query=query,
            max_results=max_results,
            topic="news",
            time_range=time_range,
        )

    def search_financials(
        self,
        query: str,
        max_results: int = 5,
        include_domains: list[str] | None = None,
    ) -> dict:
        """Shortcut for financial data searches.

        Uses advanced search depth for better financial source extraction.
        Optionally restrict to trusted financial domains.
        """
        # Default to trusted financial sources if none specified
        domains = include_domains or [
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "finance.yahoo.com",
            "seekingalpha.com",
            "fool.com",
            "marketwatch.com",
        ]

        return self.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            topic="finance",
            include_domains=domains,
        )

    def ping(self) -> bool:
        """Health check — verify Tavily API is reachable."""
        try:
            self.client.search("test", max_results=1)
            return True
        except Exception:
            return False
