"""
Shared fixtures for the test suite.

Tests marked @pytest.mark.integration require a real local Ollama (with
qwen2.5:1.5b pulled) and real Pinecone credentials in .env. They write to the
"integration-test" namespace, never the default one. Opt in with
`pytest --run-integration`.

Tests marked @pytest.mark.functional start the real MCP server and the real
FastAPI app as subprocesses against the eval-v1 namespace and talk to them
over HTTP. Opt in with `pytest --run-functional`.
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
    parser.addoption(
        "--run-functional",
        action="store_true",
        default=False,
        help="run tests marked 'functional' (starts the real API + MCP server)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs real Ollama + Pinecone, opt-in only")
    config.addinivalue_line("markers", "functional: real API + MCP server over HTTP, opt-in only")


def pytest_collection_modifyitems(config, items):
    for marker, flag in (("integration", "--run-integration"), ("functional", "--run-functional")):
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


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
        env={**os.environ, "PINECONE_NAMESPACE": "integration-test"},
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


def _start(cmd, env, port, timeout=90):
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not _wait_for_port("127.0.0.1", port, timeout=timeout):
        proc.terminate()
        output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        pytest.fail(f"{cmd} did not open port {port}:\n{output[-2000:]}")
    return proc


@pytest.fixture(scope="session")
def live_api():
    """
    Real MCP server on :8021 and real uvicorn on :8020, both pointed at the
    eval-v1 namespace. Yields the API base URL.
    """
    env = {
        **os.environ,
        "PINECONE_NAMESPACE": "eval-v1",
        "MCP_SERVER_PORT": "8021",
        "MCP_SERVER_URL": "http://127.0.0.1:8021/mcp",
    }
    mcp = _start([sys.executable, "-m", "app.mcp.server"], env, 8021)
    api = _start([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8020"], env, 8020)
    yield "http://127.0.0.1:8020"
    for proc in (api, mcp):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
