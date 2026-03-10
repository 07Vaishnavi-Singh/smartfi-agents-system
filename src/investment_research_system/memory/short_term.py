"""Short-term memory using Redis.

Handles session state and agent coordination. Data auto-expires.

PYTHON CONCEPT — DOCSTRINGS:
The first string in a module/class/function is a "docstring".
It's not a comment — it's stored as metadata, shown by IDEs and help().
TS equivalent: JSDoc /** ... */
Rust equivalent: /// doc comments
"""

import json
from typing import Any

import redis.asyncio as redis


class ShortTermMemory:
    """Redis-backed session state and agent coordination."""

    def __init__(self, redis_url: str, default_ttl: int = 3600):
        # PYTHON CONCEPT — self:
        # Python has no implicit `this`. You MUST pass `self` as the first
        # parameter to every instance method. It's explicit — no magic.
        # TS: this.client     → Python: self.client
        # Rust: self.client   → Python: self.client (same!)
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = default_ttl

    def _make_key(self, session_id: str, key: str) -> str:
        """Build a namespaced Redis key.

        PYTHON CONCEPT — underscore prefix:
        _method means "private by convention". Python has NO true private.
        It's a signal to devs: "don't call this from outside the class".
        TS equivalent: private makeKey(...)
        Rust equivalent: fn (without `pub`)
        """
        return f"session:{session_id}:{key}"

    async def store(self, session_id: str, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with auto-expiration.

        PYTHON CONCEPT — async/await:
        Almost identical to TS. The only difference:
        TS:  async store(...): Promise<void>
        Py:  async def store(...) -> None
        """
        full_key = self._make_key(session_id, key)

        # json.dumps() = JSON.stringify() in TS
        # default=str handles datetime by converting to string automatically
        serialized = json.dumps(value, default=str)

        await self.client.set(full_key, serialized, ex=ttl or self.default_ttl)
        #                                           ^ ex = expire after N seconds

    async def recall(self, session_id: str, key: str) -> Any | None:
        """Retrieve a value. Returns None if expired or doesn't exist."""
        full_key = self._make_key(session_id, key)
        data = await self.client.get(full_key)

        # PYTHON CONCEPT — `is` vs `==`:
        # `is` checks identity (same object in memory). `==` checks equality.
        # For None, ALWAYS use `is None` / `is not None`. Never `== None`.
        if data is None:
            return None

        return json.loads(data)  # json.loads() = JSON.parse() in TS

    async def update_agent_status(self, session_id: str, agent_name: str, status: str) -> None:
        """Track which agents are running/done in this session.

        Uses Redis Hash — like a JS object where you can set one property
        without reading/writing the whole thing.
        """
        agents_key = self._make_key(session_id, "agent_status")
        await self.client.hset(agents_key, agent_name, status)
        await self.client.expire(agents_key, self.default_ttl)

    async def get_agent_statuses(self, session_id: str) -> dict[str, str]:
        """Get all agent statuses for a session.

        Returns {} if no agents tracked yet, or:
        {"researcher": "done", "sentiment": "running"}
        """
        agents_key = self._make_key(session_id, "agent_status")
        return await self.client.hgetall(agents_key)

    async def delete_session(self, session_id: str) -> None:
        """Clean up all keys for a session.

        Uses SCAN instead of KEYS — KEYS blocks Redis on large datasets.
        SCAN iterates in batches. Always use SCAN in production.
        """
        pattern = f"session:{session_id}:*"
        async for key in self.client.scan_iter(match=pattern):
            await self.client.delete(key)

    async def ping(self) -> bool:
        """Health check — verify Redis is reachable.

        PYTHON CONCEPT — try/except:
        Same as try/catch in TS. Python calls them "exceptions" not "errors".
        You catch SPECIFIC exception types, not generic Exception.
        """
        try:
            await self.client.ping()
            return True
        except redis.ConnectionError:
            return False
