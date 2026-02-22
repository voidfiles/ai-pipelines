"""Jinja2 template rendering for prompt steps.

Templates use {{ args.field }} syntax. StrictUndefined ensures
missing variables blow up immediately instead of silently rendering
empty strings.
"""

from __future__ import annotations

from typing import Any

import jinja2

_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
)


def render_template(template_str: str, variables: dict[str, Any]) -> str:
    """Render a Jinja2 template string with variables available as ``args``.

    Args:
        template_str: Jinja2 template (e.g., "Hello {{ args.name }}").
        variables: Dict of values accessible via ``{{ args.key }}``.

    Returns:
        Rendered string.

    Raises:
        jinja2.UndefinedError: If the template references a variable
            that doesn't exist in *variables*.
    """
    template = _ENV.from_string(template_str)
    return template.render(args=variables)
