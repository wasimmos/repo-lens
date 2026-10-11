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
    args = parser.parse_args(["search", "route", "--repo", "sample_repo", "--top-k", "3"])
    assert args.command == "search"
    assert args.query == "route"
    assert args.repo == "sample_repo"
    assert args.top_k == 3

    # Ask
    args = parser.parse_args(["ask", "What is Flask?", "--model", "qwen2.5-coder:3b"])
    assert args.command == "ask"
    assert args.question == "What is Flask?"
    assert args.model == "qwen2.5-coder:3b"

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
