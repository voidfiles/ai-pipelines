"""Monkey-patch for claude-agent-sdk's message parser.

The SDK's ``parse_message`` raises ``MessageParseError`` on unknown message
types (e.g. ``rate_limit_event``). Because ``process_query`` does
``yield parse_message(data)``, the exception fires *inside* the async
generator frame, which kills the entire generator chain and its task group.
Catching the error on the consumer side is too late: the generator is
already dead and will only yield ``StopAsyncIteration`` from that point on.

This patch wraps ``parse_message`` so unknown/unparseable messages return
``None`` instead of raising. Consumers must skip ``None`` values.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)
_patched = False


def apply() -> None:
    """Patch ``parse_message`` once. Safe to call multiple times."""
    global _patched  # noqa: PLW0603
    if _patched:
        return

    from claude_agent_sdk._errors import MessageParseError
    import claude_agent_sdk._internal.message_parser as _mp
    import claude_agent_sdk._internal.client as _client

    _original = _mp.parse_message

    def _safe_parse_message(data: dict) -> object:
        try:
            return _original(data)
        except MessageParseError as exc:
            _log.debug("Skipping unparseable SDK message: %s", exc)
            return None

    _mp.parse_message = _safe_parse_message  # type: ignore[assignment]
    # client.py imports parse_message at module level, so we need
    # to replace its local reference too.
    _client.parse_message = _safe_parse_message  # type: ignore[assignment]
    _patched = True
