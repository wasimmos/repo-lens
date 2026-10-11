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

    # Provider & Model
    args_prov = parser.parse_args(["provider", "openai", "--model", "gpt-4o", "--api-key", "sk-123"])
    assert args_prov.command == "provider"
    assert args_prov.name == "openai"
    assert args_prov.model == "gpt-4o"
    assert args_prov.api_key == "sk-123"

    args_agent = parser.parse_args(["agent", "anthropic"])
    assert args_agent.command == "agent"
    assert args_agent.name == "anthropic"

    args_mod = parser.parse_args(["model", "llama3.2"])
    assert args_mod.command == "model"
    assert args_mod.model_name == "llama3.2"

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


def test_cmd_provider_and_model_runs(capsys, tmp_path: Path):
    from src.cli import cmd_model, cmd_provider

    fake_config = tmp_path / "config.json"
    with patch("src.llm.provider.CONFIG_FILE", fake_config), \
         patch("src.llm.provider.DATA_DIR", tmp_path):
        parser = build_parser()

        # List providers
        args = parser.parse_args(["provider"])
        cmd_provider(args)
        out = capsys.readouterr().out
        assert "LLM Providers & Agents" in out
        assert "ollama" in out

        # Switch provider
        args = parser.parse_args(["provider", "openai", "--model", "gpt-4o"])
        cmd_provider(args)
        out = capsys.readouterr().out
        assert "Switched active provider to 'openai'" in out

        # Model command
        args = parser.parse_args(["model", "gpt-4o-mini"])
        cmd_model(args)
        out = capsys.readouterr().out
        assert "Active model for provider 'openai' set to: gpt-4o-mini" in out

