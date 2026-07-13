"""
Unit tests for the tool authorization allowlist. No external dependencies -
these should always run, on any machine, with no services up.
"""

import pytest

from app.agent.authz import ROLE_TOOL_ALLOWLIST, ToolNotAuthorized, check_tool_authorized


def test_reader_can_search():
    check_tool_authorized("reader", "search_documents")


def test_reader_cannot_ingest():
    with pytest.raises(ToolNotAuthorized):
        check_tool_authorized("reader", "ingest_document")


def test_ingest_agent_can_do_both():
    check_tool_authorized("ingest_agent", "search_documents")
    check_tool_authorized("ingest_agent", "ingest_document")


def test_unknown_role_gets_nothing():
    with pytest.raises(ToolNotAuthorized):
        check_tool_authorized("nonexistent_role", "search_documents")


def test_unknown_tool_is_denied_for_every_role():
    for role in ROLE_TOOL_ALLOWLIST:
        with pytest.raises(ToolNotAuthorized):
            check_tool_authorized(role, "delete_everything")
