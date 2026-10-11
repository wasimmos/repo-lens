"""Tests for repo_loader module."""

from pathlib import Path
import pytest

from src.ingestion.repo_loader import (
    NotFoundError,
    normalize_repo_source,
    load_repository,
    list_cached_repos,
    sanitize_identifier,
)


def test_sanitize_identifier():
    assert sanitize_identifier("pallets/flask") == "pallets_flask"
    assert sanitize_identifier("my-cool.repo!") == "my-cool.repo"
    assert sanitize_identifier("   ") == "unnamed_repo"


def test_normalize_repo_source_github_url():
    repo_id, repo_name, resolved, is_remote = normalize_repo_source(
        "https://github.com/pallets/flask.git"
    )
    assert repo_id == "pallets_flask"
    assert repo_name == "flask"
    assert resolved == "https://github.com/pallets/flask.git"
    assert is_remote is True


def test_normalize_repo_source_shorthand():
    repo_id, repo_name, resolved, is_remote = normalize_repo_source("psf/requests")
    assert repo_id == "psf_requests"
    assert repo_name == "requests"
    assert resolved == "https://github.com/psf/requests.git"
    assert is_remote is True


def test_normalize_repo_source_ssh():
    repo_id, repo_name, resolved, is_remote = normalize_repo_source(
        "git@github.com:octocat/Hello-World.git"
    )
    assert repo_id == "octocat_Hello-World"
    assert repo_name == "Hello-World"
    assert resolved == "git@github.com:octocat/Hello-World.git"
    assert is_remote is True


def test_normalize_repo_source_local(tmp_path: Path):
    local_dir = tmp_path / "my_project"
    local_dir.mkdir()
    repo_id, repo_name, resolved, is_remote = normalize_repo_source(str(local_dir))
    assert repo_id == "my_project"
    assert repo_name == "my_project"
    assert resolved == str(local_dir.resolve())
    assert is_remote is False


def test_normalize_empty_source_raises():
    with pytest.raises(NotFoundError):
        normalize_repo_source("")


def test_load_local_repository(tmp_path: Path):
    local_dir = tmp_path / "test_repo"
    local_dir.mkdir()
    (local_dir / "README.md").write_text("# Test Repo")

    info = load_repository(str(local_dir))
    assert info.id == "test_repo"
    assert info.name == "test_repo"
    assert info.path == local_dir.resolve()
    assert info.is_git is False


def test_load_nonexistent_local_repo():
    with pytest.raises(NotFoundError):
        load_repository("/non/existent/path/that/does/not/exist")


def test_list_cached_repos(tmp_path: Path):
    storage_dir = tmp_path / "repos"
    storage_dir.mkdir()
    fake_repo = storage_dir / "my_cached_repo"
    fake_repo.mkdir()
    (fake_repo / "README.md").write_text("# Test")

    repos = list_cached_repos(storage_dir=storage_dir)
    matched = [r for r in repos if r.id == "my_cached_repo"]
    assert len(matched) == 1
    assert matched[0].name == "my_cached_repo"

