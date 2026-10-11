"""Tests for retrieval and search module."""

from pathlib import Path
import pytest

from src.retrieval.search import (
    split_identifier,
    tokenize,
    search,
    load_index,
    list_available_indexes,
)


def test_split_identifier_snake_case():
    parts = split_identifier("get_user_by_id")
    assert "get_user_by_id" in parts
    assert "get" in parts
    assert "user" in parts
    assert "id" in parts


def test_split_identifier_camel_case():
    parts = split_identifier("FlaskClient")
    assert "flaskclient" in parts
    assert "flask" in parts
    assert "client" in parts


def test_split_identifier_numbers():
    parts = split_identifier("HTTP404Error")
    assert "http404error" in parts
    assert "http" in parts
    assert "404" in parts
    assert "error" in parts


def test_tokenize_code_aware():
    tokens = tokenize("def parse_url_request(url): pass")
    assert "parse_url_request" in tokens
    assert "parse" in tokens
    assert "url" in tokens
    assert "request" in tokens


def test_search_exact_symbol_boost():
    records = [
        {
            "name": "unrelated_func",
            "kind": "function",
            "file": "utils.py",
            "start_line": 1,
            "end_line": 5,
            "docstring": "Some other thing",
            "code": "def unrelated_func(): pass",
        },
        {
            "name": "calculate_tax",
            "kind": "function",
            "file": "tax.py",
            "start_line": 10,
            "end_line": 20,
            "docstring": "Computes sales tax",
            "code": "def calculate_tax(amount): return amount * 0.1",
        },
    ]

    results = search("calculate_tax", records, top_k=2)
    assert len(results) >= 1
    top_score, top_record = results[0]
    assert top_record["name"] == "calculate_tax"
    assert top_score > 5.0


def test_search_file_path_boost():
    records = [
        {
            "name": "login",
            "kind": "function",
            "file": "legacy/old.py",
            "start_line": 1,
            "end_line": 5,
            "docstring": "Login",
            "code": "def login(): pass",
        },
        {
            "name": "login",
            "kind": "function",
            "file": "src/auth.py",
            "start_line": 10,
            "end_line": 20,
            "docstring": "Login",
            "code": "def login(): pass",
        },
    ]

    results = search("login in auth.py", records, top_k=2)
    assert len(results) == 2
    assert results[0][1]["file"] == "src/auth.py"
    assert results[0][0] > results[1][0]


def test_search_documentation_boost():
    records = [
        {
            "name": "install_helper",
            "kind": "function",
            "file": "setup.py",
            "start_line": 1,
            "end_line": 5,
            "docstring": "",
            "code": "def install_helper(): pass",
        },
        {
            "name": "Installation",
            "kind": "documentation",
            "file": "README.md",
            "start_line": 10,
            "end_line": 25,
            "docstring": "How to install the package",
            "code": "pip install repolens",
        },
    ]

    results = search("how to install", records, top_k=2)
    assert len(results) >= 1
    assert results[0][1]["kind"] == "documentation"


def test_load_nonexistent_index():
    with pytest.raises(FileNotFoundError):
        load_index(repo_id="nonexistent_repo_xyz_123")


def test_list_available_indexes():
    indexes = list_available_indexes()
    assert isinstance(indexes, list)
    assert len(indexes) > 0
