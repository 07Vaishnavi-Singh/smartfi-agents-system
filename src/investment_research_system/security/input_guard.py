"""Input guard — Layer 1 of prompt injection defense.

Scans user queries for known prompt injection patterns BEFORE they
reach any LLM call. If a pattern matches, raises PromptInjectionError
immediately — no tokens wasted, no risk of manipulation.

This is the LLM equivalent of a Web Application Firewall (WAF).
It catches obvious/low-effort attacks. Sophisticated attackers can
rephrase to bypass regex, which is why Layer 2 (XML delimiters in
the prompt) and Layer 3 (structured output validation) exist.

OWASP LLM01: Prompt Injection is the #1 LLM vulnerability.
In a financial research system, a successful injection could
manipulate investment recommendations.

TS equivalent: express middleware that validates request body
Rust equivalent: tower middleware layer
"""

import logging
import re

from investment_research_system.errors import PromptInjectionError

logger = logging.getLogger(__name__)

# Patterns that indicate prompt injection attempts.
# Each is a compiled regex for performance (compiled once at import time).
# IGNORECASE is set so "Ignore Previous Instructions" matches too.
#
# These are deliberately conservative — we'd rather miss a clever attack
# (caught by Layer 2/3) than block a legitimate query about prompt engineering.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|all|above|prior|my|your|the)\s+instructions",
        r"disregard\s+(previous|all|above|prior|your|the)\s+instructions",
        r"forget\s+(previous|all|above|prior|your|the)\s+instructions",
        r"(reveal|show|output|print|display)\s+(your|the)\s+(system|initial|original)\s+prompt",
        r"what\s+(is|are)\s+your\s+(system|initial|original)\s+(prompt|instructions)",
        r"you\s+are\s+now\s+a",
        r"pretend\s+(you\s+are|to\s+be)",
        r"act\s+as\s+(if\s+you\s+are|a|an)",
        r"override\s+(your|all|the)\s+(rules|instructions|guidelines)",
        r"do\s+not\s+follow\s+(your|the)\s+(rules|instructions|guidelines)",
        r"new\s+instructions?\s*:",
        r"<\s*/?\s*system\s*>",
    ]
]


def sanitize_query(query: str) -> str:
    """Check a user query for prompt injection patterns.

    This is Layer 1 of defense — fast regex check before the LLM call.
    Should be called at the API boundary (routes.py) so injections are
    caught as early as possible.

    Args:
        query: The raw user query string.

    Returns:
        The original query string (unchanged) if no injection detected.

    Raises:
        PromptInjectionError: If the query matches a known injection pattern.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning(
                "[input_guard] Prompt injection detected: %s",
                query[:100],
            )
            raise PromptInjectionError(
                "Query blocked by input guard: suspected prompt injection"
            )
    return query
