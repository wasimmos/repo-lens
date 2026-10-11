"""Tests for CLI module."""

import argparse
from pathlib import Path
from unittest.mock import patch
import pytest

from src.cli import (
    build_parser,
    cmd_list,
    cmd_status,
    cmd_use,
    get_active_repo,
    get_source_link,
    resolve_repo_id,
    set_active_repo,
)


def test_build_parser_subcommands():
    parser = build_parser()

    # Ingest & Alias (clone)
    args = parser.parse_args(["ingest", "pallets/flask", "--force", "--python-only"])
    assert args.command == "ingest"
    assert args.source == "pallets/flask"
    assert args.force is True
    assert args.python_only is True

    args_clone = parser.parse_args(["clone", "psf/requests"])
    assert args_clone.command == "clone"
    assert args_clone.source == "psf/requests"

    # Search (nargs="+")
    args = parser.parse_args(["search", "route", "decorator", "--repo", "sample_repo", "--top-k", "3", "--hybrid"])
    assert args.command == "search"
    assert args.query == ["route", "decorator"]
    assert args.repo == "sample_repo"
    assert args.top_k == 3
    assert args.hybrid is True

    # Ask (nargs="+")
    args = parser.parse_args(["ask", "What", "is", "Flask?", "--model", "qwen2.5-coder:3b", "--hybrid"])
    assert args.command == "ask"
    assert args.question == ["What", "is", "Flask?"]
    assert args.model == "qwen2.5-coder:3b"
    assert args.hybrid is True

    # Chat & Alias
    args = parser.parse_args(["chat", "--hybrid"])
    assert args.command == "chat"
    assert args.hybrid is True

    # Use / Select
    args_use = parser.parse_args(["use", "sample_repo"])
    assert args_use.command == "use"
    assert args_use.repo == "sample_repo"

    # List & Status
    assert parser.parse_args(["list"]).command == "list"
    assert parser.parse_args(["repos"]).command == "repos"
    assert parser.parse_args(["status"]).command == "status"


def test_active_repo_management(tmp_path: Path):
    with patch("src.cli.ACTIVE_REPO_FILE", tmp_path / ".active_repo"):
        assert get_active_repo() is None
        set_active_repo("test_project")
        assert get_active_repo() == "test_project"


def test_resolve_repo_id():
    # Requested repo
    repo_id, desc = resolve_repo_id("sample_repo")
    assert repo_id == "sample_repo"

    # Partial match
    repo_id_partial, _ = resolve_repo_id("sample")
    assert "sample" in repo_id_partial


def test_cmd_status_runs(capsys):
    parser = build_parser()
    args = parser.parse_args(["status"])
    cmd_status(args)
    captured = capsys.readouterr()
    assert "RepoLens System Status" in captured.out
    assert "Python:" in captured.out


def test_cmd_list_runs(capsys):
    parser = build_parser()
    args = parser.parse_args(["list"])
    cmd_list(args)
    captured = capsys.readouterr()
    assert "Available Repositories" in captured.out


def test_get_source_link():
    rec = {"file": "src/app.py", "start_line": 10, "end_line": 20}
    link, term = get_source_link(rec, web_url="https://github.com/pallets/flask", branch="main")
    assert link == "https://github.com/pallets/flask/blob/main/src/app.py#L10-L20"
    assert "src/app.py:10-20" in term
