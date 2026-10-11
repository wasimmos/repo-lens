"""End-to-end integration tests for RepoLens pipeline."""

from pathlib import Path
from unittest.mock import patch
import pytest

from src.ingestion.repo_loader import load_repository
from src.ingestion.indexer import index_repository, load_index
from src.retrieval.search import search
from src.llm.answer import answer_question, build_context, validate_citations


def test_full_rag_pipeline_end_to_end(tmp_path: Path):
    # 1. Create a simulated repository structure
    repo_dir = tmp_path / "mock_project"
    repo_dir.mkdir()

    # Python module
    (repo_dir / "service.py").write_text(
        '''"""User management module."""

class UserManager:
    """Manages active user sessions."""

    def __init__(self, db_client):
        self.db = db_client

    def create_user(self, username: str, email: str) -> dict:
        """Create a new user and persist to database."""
        return {"username": username, "email": email, "active": True}
''',
        encoding="utf-8",
    )

    # Documentation
    (repo_dir / "README.md").write_text(
        """# Mock Project

## Architecture
The system uses `UserManager` to handle user accounts.

## Getting Started
Run `pip install mock_project` to begin.
""",
        encoding="utf-8",
    )

    # Configuration
    (repo_dir / "pyproject.toml").write_text(
        """[project]
name = "mock_project"
version = "1.0.0"
dependencies = ["pydantic"]
""",
        encoding="utf-8",
    )

    # 2. Ingest / load repository
    repo_info = load_repository(str(repo_dir))
    assert repo_info.id == "mock_project"
    assert repo_info.path == repo_dir.resolve()

    # 3. Index repository
    index_file = tmp_path / "test_index.json"
    saved_path, records, stats = index_repository(
        repo_info, output_file=index_file
    )

    assert saved_path == index_file
    assert stats["scanned_files"] == 3
    assert stats["total_records"] >= 5

    # 4. Search index
    search_res = search("create_user", records, top_k=2)
    assert len(search_res) >= 1
    top_score, top_rec = search_res[0]
    assert "create_user" in top_rec["name"]
    assert top_rec["file"] == "service.py"

    # Search documentation
    doc_res = search("how to get started and install", records, top_k=2)
    assert len(doc_res) >= 1
    assert any(r[1]["kind"] == "documentation" for r in doc_res)

    # 5. Answer generation with mocked LLM
    mock_llm_response = """To create a user, instantiate `UserManager` and call `create_user`.

Sources:
- service.py:9-11
"""
    with patch("src.llm.answer.ask_ollama", return_value=mock_llm_response):
        answer, results = answer_question(
            "How do I create a user?",
            records=records,
            top_k=2,
        )

        assert "service.py:9-11" in answer
        assert "Verified citations: [service.py:9-11]" in answer
        assert len(results) > 0


def test_syntax_error_resilience(tmp_path: Path):
    """Ensure that malformed Python files do not crash indexing."""
    repo_dir = tmp_path / "broken_repo"
    repo_dir.mkdir()
    (repo_dir / "broken.py").write_text("def broken_func(:\n    pass\n")
    (repo_dir / "valid.py").write_text("def valid_func():\n    return 42\n")

    index_file = tmp_path / "broken_idx.json"
    _, records, stats = index_repository(repo_dir, output_file=index_file)

    names = [r["name"] for r in records]
    assert "valid_func" in names
    assert "broken_func" not in names


def test_empty_query_resilience(tmp_path: Path):
    """Ensure search handles empty or whitespace queries gracefully."""
    records = [{"name": "func", "kind": "function", "file": "f.py", "start_line": 1, "end_line": 2, "code": "pass"}]
    assert search("", records) == []
    assert search("   ", records) == []
    assert search("??? !!!", records) == []
