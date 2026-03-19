"""Topic extraction for memory retrieval filtering.

Classifies financial queries into topic tags so that memory search
can filter by topic BEFORE ranking by vector similarity. This prevents
context contamination (e.g., ICICI results appearing in a gold query).

Uses a hybrid approach:
1. Keyword mapping (instant, free, deterministic)
2. Falls back to a broad "general" tag for unrecognized queries

PYTHON CONCEPT — module-level constants:
TOPIC_KEYWORDS is a dict defined at module level. It's loaded once
when the module is imported, not on every function call.
TS equivalent: const TOPIC_KEYWORDS = { ... } at top of file
Rust equivalent: lazy_static! or const in a module
"""

import logging
import re

logger = logging.getLogger(__name__)

# ─── Topic taxonomy ───────────────────────────────────────────────
# Each topic maps to keywords that identify it in a query.
# Keywords are checked case-insensitively against the query text.
#
# WHY a flat taxonomy (not hierarchical)?
# Hierarchical (Asset Class → Sector → Industry → Company) is what
# Bloomberg/Refinitiv use, but it requires entity resolution and a
# maintained ontology. A flat taxonomy is good enough for our retrieval
# filtering — we just need to separate unrelated topics, not build
# a full financial knowledge graph.
#
# HOW TO EXTEND:
# Just add a new key + keywords. No code changes needed elsewhere —
# the extractor returns whatever tags match.

TOPIC_KEYWORDS: dict[str, list[str]] = {
    # Precious metals
    "gold": [
        "gold", "bullion", "XAU", "gold ETF", "gold futures",
        "gold market", "sovereign gold bond", "SGB", "gold bar",
        "gold coin", "gold price",
    ],
    "silver": ["silver", "XAG", "silver ETF", "silver futures"],

    # Commodities
    "commodities": [
        "commodity", "crude oil", "brent", "WTI", "natural gas",
        "copper", "platinum", "palladium", "agricultural",
    ],

    # Indian equities
    "indian_equities": [
        "ICICI", "HDFC", "Reliance", "TCS", "Infosys", "Wipro",
        "Nifty", "Sensex", "NSE", "BSE", "Tata", "Bajaj", "Kotak",
        "SBI", "Axis Bank", "Adani", "Bharti", "Airtel", "Maruti",
        "HUL", "ITC", "LIC",
    ],

    # US equities
    "us_equities": [
        "NVIDIA", "Apple", "Microsoft", "Google", "Alphabet",
        "Amazon", "Meta", "Tesla", "AMD", "Intel",
        "S&P 500", "SPX", "Nasdaq", "Dow Jones", "DJIA",
        "NYSE", "FAANG", "MAANG",
    ],

    # Tech sector (overlaps with equities — that's fine, a query can have multiple tags)
    "tech_stocks": [
        "semiconductor", "GPU", "AI chip", "cloud computing",
        "SaaS", "tech stock", "NVDA", "AAPL", "MSFT", "GOOGL",
        "AMZN", "META", "TSLA",
    ],

    # Crypto
    "crypto": [
        "bitcoin", "BTC", "ethereum", "ETH", "crypto",
        "blockchain", "solana", "SOL", "altcoin", "defi",
        "stablecoin", "USDT", "USDC",
    ],

    # Fixed income
    "fixed_income": [
        "bond", "treasury", "T-bill", "yield curve",
        "fixed income", "debt", "government bond", "corporate bond",
        "interest rate", "coupon",
    ],

    # Forex
    "forex": [
        "forex", "USD", "EUR", "GBP", "JPY", "INR",
        "currency", "exchange rate", "dollar", "rupee",
    ],

    # Mutual funds & ETFs
    "funds": [
        "mutual fund", "ETF", "index fund", "SIP",
        "expense ratio", "NAV", "AUM", "fund manager",
    ],

    # Macro / economy
    "macro": [
        "GDP", "inflation", "recession", "federal reserve",
        "RBI", "monetary policy", "fiscal", "CPI", "unemployment",
        "trade war", "tariff", "geopolitical",
    ],

    # Real estate
    "real_estate": [
        "real estate", "REIT", "housing", "mortgage",
        "property", "rental yield",
    ],
}

# ─── Reverse index for faster lookup ──────────────────────────────
# Pre-compute a list of (compiled_regex, topic) tuples sorted by
# keyword length descending. Longer keywords are checked first so
# "gold ETF" matches before "gold" — giving more specific tags priority.
#
# WHY regex with word boundaries (\b)?
# Plain substring matching causes false positives:
#   "bitcoin" contains "itc" → falsely matches ITC (Indian Tobacco Company)
#   "analysis" contains "sis" → could match a ticker
# Word boundaries ensure "ITC" only matches as a standalone word,
# not as part of "bitcoin". This is critical for short ticker symbols.
#
# PYTHON CONCEPT — re.compile():
# Pre-compiling regex patterns is faster than re.search(pattern, text)
# on every call. The compiled pattern object is reused across calls.
# TS equivalent: const pattern = new RegExp(`\\b${keyword}\\b`, 'i')
# Rust equivalent: regex::Regex::new(&format!(r"\b{}\b", keyword))
_KEYWORD_INDEX: list[tuple[re.Pattern, str]] = sorted(
    [
        (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), topic)
        for topic, keywords in TOPIC_KEYWORDS.items()
        for kw in keywords
    ],
    key=lambda pair: len(pair[0].pattern),
    reverse=True,  # longest keywords first
)


def extract_topics(query: str) -> list[str]:
    """Extract topic tags from a financial query using keyword matching.

    Returns a list of topic tags (e.g., ["gold", "commodities"]).
    If no keywords match, returns ["general"] as a fallback — this ensures
    every piece of stored research has at least one tag.

    The matching uses word-boundary regex (\\b) to prevent false positives.
    For example, "bitcoin" won't match the keyword "ITC" because "itc" in
    "bitcoin" is not a standalone word.

    Multi-word keywords (like "gold ETF") are checked first so they
    take priority over single-word matches.

    Args:
        query: The user's research question.

    Returns:
        List of 1+ topic tags. Never empty.

    PYTHON CONCEPT — set():
    We use a set to avoid duplicate topics (if multiple keywords from
    the same topic match). Then convert to a sorted list for determinism.
    TS equivalent: [...new Set(topics)].sort()
    Rust equivalent: BTreeSet or HashSet
    """
    matched_topics: set[str] = set()

    for pattern, topic in _KEYWORD_INDEX:
        if pattern.search(query):
            matched_topics.add(topic)

    if not matched_topics:
        logger.debug("No topic keywords matched for query: %s", query[:80])
        return ["general"]

    result = sorted(matched_topics)
    logger.info("Extracted topics %s from query: %s", result, query[:80])
    return result
