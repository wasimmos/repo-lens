"""Tests for RepoLens Web & Mobile server and REST API."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import pytest

from src.web.server import RepoLensHandler, get_local_ip


@pytest.fixture(scope="module")
def web_server():
    """Start an ephemeral test HTTP server on port 0."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), RepoLensHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


def test_get_local_ip():
    ip = get_local_ip()
    assert isinstance(ip, str)
    assert len(ip) >= 7  # e.g. "1.1.1.1"


def test_serve_static_index(web_server):
    req = urllib.request.Request(f"{web_server}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "RepoLens" in content
        assert "Codebase Intelligence" in content


def test_serve_static_assets(web_server):
    # CSS
    with urllib.request.urlopen(f"{web_server}/static/styles.css") as resp:
        assert resp.status == 200
        assert "text/css" in resp.headers.get("Content-Type", "")

    # JS
    with urllib.request.urlopen(f"{web_server}/static/app.js") as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type", "")

    # Manifest
    with urllib.request.urlopen(f"{web_server}/manifest.json") as resp:
        assert resp.status == 200
        assert "json" in resp.headers.get("Content-Type", "")


def test_api_status(web_server):
    with urllib.request.urlopen(f"{web_server}/api/status") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "active_provider" in data
        assert "active_model" in data
        assert "lan_ip" in data
        assert "server_port" in data


def test_api_repos(web_server):
    with urllib.request.urlopen(f"{web_server}/api/repos") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "repos" in data
        assert isinstance(data["repos"], list)


def test_api_providers(web_server):
    with urllib.request.urlopen(f"{web_server}/api/providers") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "providers" in data
        assert len(data["providers"]) >= 8


def test_api_repos_use(web_server, tmp_path: Path):
    with patch("src.cli.ACTIVE_REPO_FILE", tmp_path / ".active_repo"):
        body = json.dumps({"repo": "my_test_repo"}).encode("utf-8")
        req = urllib.request.Request(
            f"{web_server}/api/repos/use",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert "my_test_repo" in data["active_repo"]


def test_api_search(web_server):
    mock_records = [
        {"name": "hello_world", "kind": "function", "file": "src/app.py", "start_line": 1, "end_line": 5, "docstring": "greeting", "code": "def hello_world(): pass"},
        {"name": "other_func", "kind": "function", "file": "src/other.py", "start_line": 1, "end_line": 5, "docstring": "other", "code": "def other_func(): pass"},
    ]
    with patch("src.web.server.load_index", return_value=mock_records):
        body = json.dumps({"query": "hello", "repo": "test_repo"}).encode("utf-8")
        req = urllib.request.Request(
            f"{web_server}/api/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["count"] >= 1
            assert data["results"][0]["name"] == "hello_world"


def test_api_ask(web_server):
    mock_records = [
        {"name": "auth_token", "kind": "function", "file": "auth.py", "start_line": 10, "end_line": 20, "docstring": "", "code": "def auth_token(): pass"}
    ]
    mock_answer = "Authentication works via tokens [auth.py:10-20]."
    with patch("src.web.server.load_index", return_value=mock_records), \
         patch("src.llm.answer.ask_ollama", return_value=mock_answer):
        body = json.dumps({"question": "How does auth work?", "repo": "test_repo"}).encode("utf-8")
        req = urllib.request.Request(
            f"{web_server}/api/ask",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert "auth.py:10-20" in data["answer"]
            assert len(data["evidence"]) >= 1
