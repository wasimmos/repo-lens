"""Codebase indexer for RepoLens.

Parses source code, documentation, and configuration files across
repositories into structured records with precise line numbers for RAG retrieval.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ingestion.repo_loader import DEFAULT_STORAGE_DIR, RepoInfo, load_repository
REPO_DIR = PROJECT_DIR / "data" / "sample_repo"
OUTPUT_FILE = PROJECT_DIR / "data" / "index.json"
INDEXES_DIR = PROJECT_DIR / "data" / "indexes"

IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "node_modules", "build", "dist", ".tox", ".eggs",
    ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    "target", ".next", ".nuxt", "coverage", ".coverage",
    "site-packages",
}

IGNORED_EXTENSIONS = {
    # Binaries and compiled
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bin",
    # Images and media
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    # Documents and fonts
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar", ".woff", ".woff2", ".ttf", ".eot",
    # Lock files that are too verbose / minified files
    ".min.js", ".min.css", ".map",
}

IGNORED_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
}

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB


def is_binary_file(file_path: Path) -> bool:
    """Check whether a file is binary by looking for null bytes in the first 1KB."""
    try:
        with file_path.open("rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except (OSError, PermissionError):
        return True


class PythonSymbolVisitor(ast.NodeVisitor):
    """AST visitor that extracts functions, classes, and methods with qualified names."""

    def __init__(self, lines: list[str], rel_path: str):
        self.lines = lines
        self.rel_path = rel_path
        self.records: list[dict[str, Any]] = []
        self.class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start = node.lineno
        end = getattr(node, "end_lineno", start)

        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(d.lineno for d in decorators))

        raw_docstring = ast.get_docstring(node) or ""
        header_end = node.body[0].lineno - 1 if node.body else end
        code = "\n".join(self.lines[start - 1 : header_end])

        docstring = raw_docstring[:400]
        if len(raw_docstring) > 400:
            docstring += "... [truncated]"

        full_name = ".".join(self.class_stack + [node.name]) if self.class_stack else node.name

        self.records.append({
            "name": full_name,
            "kind": "class",
            "file": self.rel_path,
            "start_line": start,
            "end_line": end,
            "docstring": docstring,
            "code": code,
            "language": "python",
        })

        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node, is_async=True)

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        start = node.lineno
        end = getattr(node, "end_lineno", start)

        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(d.lineno for d in decorators))

        raw_docstring = ast.get_docstring(node) or ""
        code = "\n".join(self.lines[start - 1 : end])

        if self.class_stack:
            full_name = ".".join(self.class_stack + [node.name])
            kind = "method"
        else:
            full_name = node.name
            kind = "async_function" if is_async else "function"

        self.records.append({
            "name": full_name,
            "kind": kind,
            "file": self.rel_path,
            "start_line": start,
            "end_line": end,
            "docstring": raw_docstring,
            "code": code,
            "language": "python",
        })

        self.generic_visit(node)


def extract_python_symbols(file_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    """Extract Python symbols (classes, methods, functions, module docstring) using AST."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError) as error:
        print(f"Skipping unreadable file {file_path}: {error}")
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as error:
        print(f"Skipping syntax error in {file_path}: {error}")
        return []

    lines = source.splitlines()
    rel_path = file_path.relative_to(repo_root).as_posix()

    visitor = PythonSymbolVisitor(lines, rel_path)
    visitor.visit(tree)
    records = visitor.records

    # Also capture top-level module docstring if significant
    module_doc = ast.get_docstring(tree)
    if module_doc:
        doc_lines = len(module_doc.splitlines())
        records.insert(0, {
            "name": file_path.stem,
            "kind": "module",
            "file": rel_path,
            "start_line": 1,
            "end_line": max(doc_lines + 2, 1),
            "docstring": module_doc[:500],
            "code": "\n".join(lines[:min(len(lines), doc_lines + 5)]),
            "language": "python",
        })

    return records


def extract_markdown_chunks(file_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    """Chunk Markdown documentation files by headers and paragraphs."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return []

    lines = content.splitlines()
    if not lines:
        return []

    rel_path = file_path.relative_to(repo_root).as_posix()
    records: list[dict[str, Any]] = []

    # Find heading positions
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    sections: list[tuple[int, str, int]] = []  # (line_no_1indexed, title, level)

    for i, line in enumerate(lines, start=1):
        match = heading_pattern.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append((i, title, level))

    if not sections:
        # No headings: chunk into 50-line blocks with 10-line overlap
        return chunk_text_by_lines(
            lines, rel_path, kind="documentation", name=file_path.name, language="markdown"
        )

    # Process sections
    for idx, (start_line, title, _) in enumerate(sections):
        if idx + 1 < len(sections):
            end_line = sections[idx + 1][0] - 1
        else:
            end_line = len(lines)

        if end_line < start_line:
            end_line = start_line

        section_lines = lines[start_line - 1 : end_line]
        section_code = "\n".join(section_lines)

        # Extract first non-heading paragraph as docstring/summary
        docstring = ""
        for s_line in section_lines[1:]:
            s_clean = s_line.strip()
            if s_clean and not s_clean.startswith("#"):
                docstring = s_clean[:300]
                break

        # If section is small to medium (<= 80 lines), keep as one chunk
        if len(section_lines) <= 80:
            records.append({
                "name": title,
                "kind": "documentation",
                "file": rel_path,
                "start_line": start_line,
                "end_line": end_line,
                "docstring": docstring or title,
                "code": section_code,
                "language": "markdown",
            })
        else:
            # Sub-chunk large section while preserving line numbers
            sub_records = chunk_text_by_lines(
                lines,
                rel_path,
                kind="documentation",
                name=title,
                language="markdown",
                line_offset=start_line,
                total_lines=len(section_lines),
            )
            records.extend(sub_records)

    return records


def chunk_text_by_lines(
    all_lines: list[str],
    rel_path: str,
    kind: str,
    name: str,
    language: str,
    line_offset: int = 1,
    total_lines: Optional[int] = None,
    chunk_size: int = 50,
    overlap: int = 10,
) -> list[dict[str, Any]]:
    """Generic sliding window chunker for arbitrary text or code files."""
    records: list[dict[str, Any]] = []
    lines_subset = (
        all_lines[line_offset - 1 : line_offset - 1 + total_lines]
        if total_lines is not None
        else all_lines[line_offset - 1 :]
    )

    n = len(lines_subset)
    if n == 0:
        return []

    step = max(chunk_size - overlap, 1)
    part = 1

    for start_idx in range(0, n, step):
        end_idx = min(start_idx + chunk_size, n)
        chunk_lines = lines_subset[start_idx:end_idx]

        actual_start_line = line_offset + start_idx
        actual_end_line = line_offset + end_idx - 1
        if actual_end_line < actual_start_line:
            actual_end_line = actual_start_line

        chunk_name = name if n <= chunk_size else f"{name} (p{part})"
        first_meaningful_line = next((l.strip() for l in chunk_lines if l.strip()), "")

        records.append({
            "name": chunk_name,
            "kind": kind,
            "file": rel_path,
            "start_line": actual_start_line,
            "end_line": actual_end_line,
            "docstring": first_meaningful_line[:200],
            "code": "\n".join(chunk_lines),
            "language": language,
        })

        part += 1
        if end_idx >= n:
            break

    return records


def extract_general_file_chunks(
    file_path: Path, repo_root: Path
) -> list[dict[str, Any]]:
    """Extract chunks from documentation, configuration, or other code files."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return []

    lines = content.splitlines()
    if not lines:
        return []

    rel_path = file_path.relative_to(repo_root).as_posix()
    ext = file_path.suffix.lower()
    filename_lower = file_path.name.lower()

    # Determine kind and language
    if ext in {".md", ".markdown"}:
        return extract_markdown_chunks(file_path, repo_root)

    if ext in {".rst", ".txt"}:
        kind = "documentation"
        language = ext.lstrip(".")
    elif (
        ext in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}
        or filename_lower in {"dockerfile", "makefile", "requirements.txt", "gemfile"}
    ):
        kind = "config"
        language = ext.lstrip(".") or filename_lower
    else:
        kind = "code"
        language = ext.lstrip(".")

    # Keep small files (< 80 lines) whole
    if len(lines) <= 80:
        first_line = next((l.strip() for l in lines if l.strip()), file_path.name)
        return [{
            "name": file_path.name,
            "kind": kind,
            "file": rel_path,
            "start_line": 1,
            "end_line": len(lines),
            "docstring": first_line[:200],
            "code": content,
            "language": language,
        }]

    return chunk_text_by_lines(
        lines,
        rel_path,
        kind=kind,
        name=file_path.name,
        language=language,
        chunk_size=60,
        overlap=12,
    )


def extract_symbols(
    file_path: Path, repo_root: Optional[Path] = None
) -> list[dict[str, Any]]:
    """Backward-compatible function to extract symbols/records from a file."""
    root = repo_root or REPO_DIR
    if file_path.suffix.lower() == ".py":
        return extract_python_symbols(file_path, root)
    return extract_general_file_chunks(file_path, root)


def should_index_file(file_path: Path) -> bool:
    """Determine if a file should be indexed."""
    if file_path.name in IGNORED_FILENAMES:
        return False

    suffix = file_path.suffix.lower()
    if suffix in IGNORED_EXTENSIONS:
        return False

    for part in file_path.parts:
        if part in IGNORED_DIRS:
            return False

    if file_path.exists():
        try:
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                return False
        except (OSError, PermissionError):
            return False

        if is_binary_file(file_path):
            return False

    return True


def index_repository(
    repo_target: Path | str | RepoInfo,
    output_file: Optional[Path] = None,
    repo_id: Optional[str] = None,
    include_all_files: bool = True,
    python_only: bool = False,
    web_url: Optional[str] = None,
    branch: Optional[str] = None,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """Index a target repository and save records to disk.

    Args:
        repo_target: Path to repo, GitHub URL/shorthand, or RepoInfo object.
        output_file: Custom output json path.
        repo_id: Custom repo identifier.
        include_all_files: If True, indexes Python, docs, configs, and other code.
        python_only: If True, only indexes Python files (sets include_all_files=False).
        web_url: Optional remote GitHub web URL for citations.
        branch: Optional git branch name for citations.

    Returns:
        tuple of (index_file_path, records, stats)
    """
    if python_only:
        include_all_files = False

    if isinstance(repo_target, RepoInfo):
        repo_info = repo_target
        repo_dir = repo_info.path
        target_id = repo_id or repo_info.id
    elif isinstance(repo_target, (Path, str)) and Path(repo_target).exists():
        repo_dir = Path(repo_target).resolve()
        target_id = repo_id or repo_dir.name
        repo_info = None
    else:
        repo_info = load_repository(str(repo_target))
        repo_dir = repo_info.path
        target_id = repo_id or repo_info.id

    if not repo_dir.exists() or not repo_dir.is_dir():
        raise FileNotFoundError(f"Repository directory does not exist: {repo_dir}")

    records: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}
    scanned_files = 0

    # Scan files
    for root, dirs, files in os.walk(repo_dir):
        # Filter directories in-place to avoid traversing ignored trees
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        for filename in files:
            file_path = Path(root) / filename
            if not should_index_file(file_path):
                continue

            ext = file_path.suffix.lower()
            scanned_files += 1

            if ext == ".py":
                syms = extract_python_symbols(file_path, repo_dir)
                records.extend(syms)
                file_counts["python"] = file_counts.get("python", 0) + 1
            elif include_all_files:
                chunks = extract_general_file_chunks(file_path, repo_dir)
                records.extend(chunks)
                cat = "markdown" if ext in {".md", ".markdown"} else (ext.lstrip(".") or "other")
                file_counts[cat] = file_counts.get(cat, 0) + 1

    # Determine index target paths
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    specific_index_file = output_file or (INDEXES_DIR / f"{target_id}.json")
    specific_index_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to specific repository index
    index_payload = records

    specific_index_file.write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Also update global default index.json for backward compatibility
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Extract web URL and git metadata
    if repo_info:
        web_url = web_url or repo_info.web_url
        branch = branch or repo_info.branch
        commit_sha = repo_info.commit_sha
    else:
        from src.ingestion.repo_loader import get_git_metadata
        commit_sha, git_branch, git_web_url = get_git_metadata(repo_dir)
        web_url = web_url or git_web_url
        branch = branch or git_branch

    stats = {
        "repo_id": target_id,
        "repo_path": str(repo_dir),
        "web_url": web_url,
        "branch": branch or "main",
        "commit_sha": commit_sha,
        "total_records": len(records),
        "scanned_files": scanned_files,
        "file_types": file_counts,
    }

    # Save metadata sidecar
    meta_target = specific_index_file.with_name(f"{target_id}_meta.json")
    meta_target.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    global_meta = OUTPUT_FILE.parent / "index_meta.json"
    global_meta.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return specific_index_file, records, stats


def get_repo_metadata(repo_id: Optional[str] = None) -> dict[str, Any]:
    """Retrieve metadata (including web_url, branch) for an indexed repository."""
    if repo_id and repo_id != "default":
        meta_file = INDEXES_DIR / f"{repo_id}_meta.json"
        if meta_file.exists():
            try:
                return json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    global_meta = OUTPUT_FILE.parent / "index_meta.json"
    if global_meta.exists():
        try:
            return json.loads(global_meta.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {}


def get_index_path(repo_id: Optional[str] = None) -> Path:
    """Get the path to an index file for a given repo ID or default index."""
    if repo_id:
        target = INDEXES_DIR / f"{repo_id}.json"
        if target.exists():
            return target
    return OUTPUT_FILE


def load_index(
    index_path: Optional[Path | str] = None,
    repo_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Load indexed records from file with backward-compatibility support."""
    if index_path:
        target = Path(index_path)
    elif repo_id:
        target = get_index_path(repo_id)
    else:
        target = OUTPUT_FILE

    if not target.exists():
        raise FileNotFoundError(
            f"Index not found: {target}\n"
            "Run python src/ingestion/indexer.py first."
        )

    with target.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return []


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="RepoLens Codebase Indexer - parses code and docs for RAG search"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(REPO_DIR),
        help="Repository path or GitHub URL (default: data/sample_repo)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom output JSON path",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only index Python files (skip markdown and configs)",
    )
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    print(f"Indexing repository: {args.target}")

    index_file, records, stats = index_repository(
        args.target,
        output_file=out_path,
        include_all_files=not args.python_only,
    )

    print("\n--- Indexing Complete ---")
    print(f"Repository:      {stats['repo_path']}")
    print(f"Repository ID:   {stats['repo_id']}")
    print(f"Files scanned:   {stats['scanned_files']}")
    print(f"Total passages:  {stats['total_records']}")
    print(f"Index saved to:  {index_file}")
    print(f"Global index at: {OUTPUT_FILE}")
    print("\nFile breakdown:")
    for ftype, count in stats["file_types"].items():
        print(f"  - {ftype:12}: {count} files")

    print("\nSample indexed items:")
    for item in records[:8]:
        print(
            f"  [{item['kind']:13}] {item['name']:30} "
            f"{item['file']}:{item['start_line']}-{item['end_line']}"
        )


if __name__ == "__main__":
    main()
