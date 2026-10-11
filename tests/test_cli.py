"""Tests for CLI module."""

import argparse
from unittest.mock import patch
import pytest

from src.cli import build_parser, cmd_list, cmd_status


def test_build_parser_subcommands():
    parser = build_parser()
    
    # Ingest
    args = parser.parse_args(["ingest", "pallets/flask", "--force", "--python-only"])
    assert args.command == "ingest"
    assert args.source == "pallets/flask"
    assert args.force is True
    assert args.python_only is True

    # Search
    args = parser.parse_args(["search", "route", "--repo", "sample_repo", "--top-k", "3", "--hybrid"])
    assert args.command == "search"
    assert args.query == "route"
    assert args.repo == "sample_repo"
    assert args.top_k == 3
    assert args.hybrid is True

    # Ask
    args = parser.parse_args(["ask", "What is Flask?", "--model", "qwen2.5-coder:3b", "--hybrid"])
    assert args.command == "ask"
    assert args.question == "What is Flask?"
    assert args.model == "qwen2.5-coder:3b"
    assert args.hybrid is True

    # Chat
    args = parser.parse_args(["chat", "--hybrid"])
    assert args.command == "chat"
    assert args.hybrid is True

    # List & Status
    assert parser.parse_args(["list"]).command == "list"
    assert parser.parse_args(["status"]).command == "status"


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
    assert "Cached Repositories" in captured.out
    assert "Available Indexes" in captured.out


def test_get_source_link():
    from src.cli import get_source_link

    rec = {"file": "src/app.py", "start_line": 10, "end_line": 20}
    link, term = get_source_link(rec, web_url="https://github.com/pallets/flask", branch="main")
    assert link == "https://github.com/pallets/flask/blob/main/src/app.py#L10-L20"
    assert "src/app.py:10-20" in term
