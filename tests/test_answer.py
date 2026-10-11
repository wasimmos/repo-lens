"""Tests for LLM answering and citation verification module."""

from src.llm.answer import validate_citations, build_context


def test_validate_citations_bracketed():
    results = [
        (10.0, {"file": "src/app.py", "start_line": 10, "end_line": 50}),
    ]
    answer = "The app factory is configured here [src/app.py:15-30]."
    validated = validate_citations(answer, results)
    assert "Verified citations: [src/app.py:15-30]" in validated


def test_validate_citations_markdown_list():
    results = [
        (10.0, {"file": "src/helpers.py", "start_line": 200, "end_line": 250}),
        (8.0, {"file": "src/app.py", "start_line": 50, "end_line": 100}),
    ]
    answer = """`url_for` requires an active context.

Sources:
- src/helpers.py:200-250
- src/app.py:55-80
"""
    validated = validate_citations(answer, results)
    assert "Verified citations:" in validated
    assert "[src/helpers.py:200-250]" in validated
    assert "[src/app.py:55-80]" in validated


def test_validate_citations_unverified():
    results = [
        (10.0, {"file": "src/app.py", "start_line": 10, "end_line": 50}),
    ]
    # File not in results, or range outside allowed bounds
    answer = "Claims made here [fake/path.py:1-10] and [src/app.py:1-200]."
    validated = validate_citations(answer, results)
    assert "Unverified citations:" in validated
    assert "[fake/path.py:1-10]" in validated
    assert "[src/app.py:1-200]" in validated


def test_build_context():
    results = [
        (
            9.5,
            {
                "file": "calc.py",
                "start_line": 1,
                "end_line": 5,
                "kind": "function",
                "name": "add",
                "docstring": "Adds two numbers",
                "code": "def add(a, b): return a + b",
            },
        )
    ]
    context = build_context(results)
    assert "calc.py:1-5" in context
    assert "Symbol: function add" in context
    assert "Adds two numbers" in context


def test_validate_citations_with_github_links():
    results = [
        (10.0, {"file": "src/app.py", "start_line": 10, "end_line": 50}),
    ]
    answer = "Factory configured at [src/app.py:15-30]."
    validated = validate_citations(
        answer,
        results,
        web_url="https://github.com/pallets/flask",
        branch="main",
    )
    expected_link = "[src/app.py:15-30](https://github.com/pallets/flask/blob/main/src/app.py#L15-L30)"
    assert expected_link in validated
