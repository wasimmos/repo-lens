"""Tests for indexer module."""

from pathlib import Path
import pytest

from src.ingestion.indexer import (
    extract_python_symbols,
    extract_markdown_chunks,
    extract_general_file_chunks,
    should_index_file,
    is_binary_file,
    index_repository,
    load_index,
)


def test_extract_python_symbols(tmp_path: Path):
    code = '''"""Module docstring."""

def standalone_func(a, b):
    """Calculates sum."""
    return a + b

class MyService:
    """Service class."""

    def __init__(self):
        self.val = 1

    async def fetch_data(self):
        """Fetches async data."""
        return "data"
'''
    py_file = tmp_path / "service.py"
    py_file.write_text(code, encoding="utf-8")

    records = extract_python_symbols(py_file, tmp_path)

    names = [r["name"] for r in records]
    assert "service" in names  # module docstring
    assert "standalone_func" in names
    assert "MyService" in names
    assert "MyService.__init__" in names
    assert "MyService.fetch_data" in names

    # Check method kind and qualified name
    fetch_rec = next(r for r in records if r["name"] == "MyService.fetch_data")
    assert fetch_rec["kind"] == "method"
    assert fetch_rec["start_line"] > fetch_rec["end_line"] - 10
    assert "Fetches async data" in fetch_rec["docstring"]

    # Check class docstring
    class_rec = next(r for r in records if r["name"] == "MyService")
    assert class_rec["kind"] == "class"
    assert "Service class" in class_rec["docstring"]


def test_extract_markdown_chunks(tmp_path: Path):
    md_content = """# Overview
This is an overview of the project.

## Installation
Run `pip install repo-lens` to install.

## Usage
Import and use as follows:
```python
import repolens
```
"""
    md_file = tmp_path / "README.md"
    md_file.write_text(md_content, encoding="utf-8")

    records = extract_markdown_chunks(md_file, tmp_path)
    assert len(records) >= 3

    titles = [r["name"] for r in records]
    assert "Overview" in titles
    assert "Installation" in titles
    assert "Usage" in titles

    inst_rec = next(r for r in records if r["name"] == "Installation")
    assert inst_rec["kind"] == "documentation"
    assert "pip install" in inst_rec["code"]
    assert inst_rec["start_line"] == 4
    assert inst_rec["end_line"] == 6


def test_extract_general_config_file(tmp_path: Path):
    toml_content = """[project]
name = "demo"
version = "0.1.0"
dependencies = ["requests"]
"""
    cfg_file = tmp_path / "pyproject.toml"
    cfg_file.write_text(toml_content, encoding="utf-8")

    records = extract_general_file_chunks(cfg_file, tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "config"
    assert rec["name"] == "pyproject.toml"
    assert rec["start_line"] == 1
    assert rec["end_line"] == 4
    assert "requests" in rec["code"]


def test_binary_file_detection(tmp_path: Path):
    bin_file = tmp_path / "test.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03\x04")
    assert is_binary_file(bin_file) is True

    text_file = tmp_path / "test.txt"
    text_file.write_text("Hello world")
    assert is_binary_file(text_file) is False


def test_should_index_file(tmp_path: Path):
    assert should_index_file(tmp_path / "app.py") is True
    assert should_index_file(tmp_path / "README.md") is True
    assert should_index_file(tmp_path / "app.pyc") is False
    assert should_index_file(tmp_path / "package-lock.json") is False
    assert should_index_file(tmp_path / "image.png") is False


def test_index_repository_end_to_end(tmp_path: Path):
    repo_dir = tmp_path / "sample_codebase"
    repo_dir.mkdir()
    (repo_dir / "calc.py").write_text("def add(x, y):\n    return x + y\n")
    (repo_dir / "README.md").write_text("# Project\nA calculator project.\n")

    output_idx = tmp_path / "index.json"
    index_file, records, stats = index_repository(
        repo_dir,
        output_file=output_idx,
        repo_id="test_calc",
    )

    assert index_file.exists()
    assert stats["total_records"] >= 2
    assert stats["scanned_files"] == 2

    loaded = load_index(index_file)
    assert len(loaded) == len(records)
    names = [r["name"] for r in loaded]
    assert "add" in names
    assert "Project" in names


def test_index_repository_python_only_and_metadata(tmp_path: Path):
    repo_dir = tmp_path / "mixed_codebase"
    repo_dir.mkdir()
    (repo_dir / "calc.py").write_text("def mul(a, b):\n    return a * b\n")
    (repo_dir / "notes.md").write_text("# Notes\nSome notes.\n")

    output_idx = tmp_path / "out.json"
    index_file, records, stats = index_repository(
        repo_dir,
        output_file=output_idx,
        repo_id="mixed_test",
        python_only=True,
        web_url="https://github.com/example/mixed",
        branch="develop",
    )

    assert stats["web_url"] == "https://github.com/example/mixed"
    assert stats["branch"] == "develop"
    assert "markdown" not in stats["file_types"]
    assert stats["file_types"].get("python") == 1
    names = [r["name"] for r in records]
    assert "mul" in names
    assert "Notes" not in names

