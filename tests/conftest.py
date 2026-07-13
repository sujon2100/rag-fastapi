"""
Shared fixtures for the test suite.

Tests marked @pytest.mark.integration require a real local Ollama (with the
llama3 model pulled) and real Pinecone credentials in .env. They are skipped
by default - opt in with `pytest --run-integration`.
"""

import os
import socket
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked 'integration' (needs real Ollama + Pinecone)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs real Ollama + Pinecone, opt-in only")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="needs --run-integration (real Ollama + Pinecone)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def _wait_for_port(host: str, port: int, timeout: float = 20) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def mcp_server():
    """Spawns the real MCP server (app/mcp/server.py) as a subprocess on :8001."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.mcp.server"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if not _wait_for_port("127.0.0.1", 8001, timeout=20):
        proc.terminate()
        output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        pytest.fail(f"MCP server did not come up on port 8001:\n{output}")
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
