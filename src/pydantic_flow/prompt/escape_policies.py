"""Escape policy implementations for template rendering."""

from html import escape


def no_escape(s: str) -> str:
    """Return string without escaping."""
    return s


def escape_html(s: str) -> str:
    """Escape HTML special characters."""
    return escape(s)
