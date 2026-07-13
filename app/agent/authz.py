"""
Static tool-authorization allowlist for the LangGraph agent.

A small, honest echo of the access-control pattern from the InfraSteward
platform work: before any MCP tool is actually invoked, check whether the
calling role is permitted to use it. No policy engine, no dynamic rules,
no per-tenant scoping - just a fixed mapping, checked on every dispatch.
Fails closed: a role not listed below gets no tools at all.
"""


class ToolNotAuthorized(Exception):
    """Raised when a role attempts to call a tool it is not allowed to use."""


ROLE_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "reader": {"search_documents"},
    "ingest_agent": {"search_documents", "ingest_document"},
}


def check_tool_authorized(role: str, tool_name: str) -> None:
    """Raise ToolNotAuthorized if `role` may not call `tool_name`."""
    allowed = ROLE_TOOL_ALLOWLIST.get(role, set())
    if tool_name not in allowed:
        raise ToolNotAuthorized(
            f"role '{role}' is not authorized to call tool '{tool_name}'"
        )
