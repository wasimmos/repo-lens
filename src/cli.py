"""RepoLens CLI - Unified Command Line Interface for GitHub Codebase RAG.

Provides subcommands to ingest, index, search, ask questions, and chat with
any GitHub or local repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ingestion.indexer import get_repo_metadata, index_repository
from src.ingestion.repo_loader import list_cached_repos, load_repository
from src.llm.answer import answer_question
from src.llm.provider import MODEL, OLLAMA_URL, check_ollama_available
from src.retrieval.search import display_results, list_available_indexes, load_index, search


def get_source_link(
    rec: dict,
    web_url: Optional[str] = None,
    branch: str = "main",
    repo_path: Optional[str] = None,
) -> tuple[str, str]:
    """Return (direct_url, osc8_terminal_link) for a record."""
    file_path = rec.get("file", "")
    start = rec.get("start_line", 1)
    end = rec.get("end_line", 1)
    loc = f"{file_path}:{start}-{end}"

    link = ""
    if web_url:
        clean_base = web_url.rstrip("/")
        link = f"{clean_base}/blob/{branch}/{file_path}#L{start}-L{end}"
    elif repo_path:
        abs_p = (Path(repo_path) / file_path).resolve()
        link = f"file://{abs_p}#L{start}"

    terminal_loc = f"\033]8;;{link}\033\\{loc}\033]8;;\033\\" if link else loc
    return link, terminal_loc


def cmd_ingest(args: argparse.Namespace) -> None:
    """Clone/load and index a repository."""
    print(f"Ingesting repository from: {args.source}")
    info = load_repository(args.source, force_clone=args.force)

    print(f"\nRepository loaded at: {info.path}")
    print(f"ID: {info.id} | Branch: {info.branch or 'N/A'}")

    print("\nStarting indexing...")
    index_file, records, stats = index_repository(info, include_all_files=not args.python_only)

    print("\n=== Ingestion Successful ===")
    print(f"Repository ID:  {stats['repo_id']}")
    print(f"Files scanned:  {stats['scanned_files']}")
    print(f"Total passages: {stats['total_records']}")
    print(f"Index location: {index_file}")
    print("\nBreakdown by type:")
    for ext, count in stats["file_types"].items():
        print(f"  - {ext:12}: {count} files")


def cmd_search(args: argparse.Namespace) -> None:
    """Search code and documentation passages in an indexed repository."""
    records = load_index(repo_id=args.repo)
    meta = get_repo_metadata(args.repo)
    repo_name = args.repo or "global index"
    mode_desc = "hybrid" if getattr(args, "hybrid", False) else "BM25"
    print(f"Searching [{repo_name}] ({len(records)} passages, mode: {mode_desc}) for: '{args.query}'")

    results = search(args.query, records, top_k=args.top_k, hybrid=getattr(args, "hybrid", False))
    display_results(results, meta=meta)


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question about a repository and receive a cited response."""
    repo_name = args.repo or "global index"
    model = args.model or MODEL

    meta = get_repo_metadata(args.repo)
    web_url = meta.get("web_url")
    branch = meta.get("branch", "main")
    repo_path = meta.get("repo_path")

    print(f"Querying [{repo_name}] using model '{model}'...")
    answer, results = answer_question(
        args.question,
        repo_id=args.repo,
        top_k=args.top_k,
        model=model,
        hybrid=getattr(args, "hybrid", False),
    )

    if results:
        print("\n--- Retrieved Evidence ---")
        for score, rec in results:
            link, term_loc = get_source_link(rec, web_url, branch, repo_path)
            print(f"- [{rec.get('kind', 'code')}] {rec.get('name', 'unnamed')}")
            print(f"  Location: {term_loc}")
            if link:
                print(f"  Link:     {link}")

    print("\n--- RepoLens Answer ---")
    print(answer)
    print()


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive multi-turn Q&A terminal loop."""
    repo_name = args.repo or "global index"
    model = args.model or MODEL
    records = load_index(repo_id=args.repo)
    meta = get_repo_metadata(args.repo)
    web_url = meta.get("web_url")
    branch = meta.get("branch", "main")
    repo_path = meta.get("repo_path")

    print(f"=== RepoLens Interactive Chat ===")
    print(f"Target repository: [{repo_name}] ({len(records)} passages)")
    print(f"Active LLM:        {model}")
    print("Type 'exit' or 'quit' to end session.\n")

    while True:
        try:
            query = input("RepoLens > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        try:
            answer, results = answer_question(
                query,
                records=records,
                top_k=args.top_k,
                model=model,
                hybrid=getattr(args, "hybrid", False),
            )

            if results:
                print("\n[Evidence]")
                for score, rec in results:
                    link, term_loc = get_source_link(rec, web_url, branch, repo_path)
                    print(f"  - [{rec.get('kind', 'code')}] {rec.get('name', 'unnamed')} ({term_loc})")
                    if link:
                        print(f"    ➔ {link}")

            print("\n[Answer]")
            print(answer)
            print("-" * 50 + "\n")
        except Exception as err:
            print(f"\nError: {err}\n")


def cmd_list(args: argparse.Namespace) -> None:
    """List all cached repositories and available indexes."""
    repos = list_cached_repos()
    indexes = list_available_indexes()

    print(f"=== Cached Repositories ({len(repos)}) ===")
    for r in repos:
        print(f"  - [{r.id}] {r.name} -> {r.path} (branch: {r.branch or 'N/A'})")

    print(f"\n=== Available Indexes ({len(indexes)}) ===")
    for idx in indexes:
        print(f"  - {idx}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check system and provider status."""
    is_online, msg = check_ollama_available()
    print("=== RepoLens System Status ===")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Project root: {PROJECT_DIR}")
    print(f"Ollama URL:   {OLLAMA_URL}")
    print(f"Default model:{MODEL}")
    print(f"Ollama check: {'ONLINE' if is_online else 'OFFLINE'} - {msg}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repolens",
        description="RepoLens: Codebase intelligence and RAG assistant for GitHub repositories.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Clone/load and index a repository")
    p_ingest.add_argument("source", help="GitHub URL, owner/repo shorthand, or local directory path")
    p_ingest.add_argument("--force", action="store_true", help="Force re-clone if already cached")
    p_ingest.add_argument("--python-only", action="store_true", help="Only index Python files")
    p_ingest.set_defaults(func=cmd_ingest)

    # Search
    p_search = subparsers.add_parser("search", help="Search code and docs passages")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--repo", "-r", default=None, help="Repository ID to search")
    p_search.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--hybrid", action="store_true", help="Use hybrid BM25 + dense semantic vector search")
    p_search.set_defaults(func=cmd_search)

    # Ask
    p_ask = subparsers.add_parser("ask", help="Ask a question and get a cited answer")
    p_ask.add_argument("question", help="Question about the repository")
    p_ask.add_argument("--repo", "-r", default=None, help="Repository ID to query")
    p_ask.add_argument("--model", "-m", default=None, help="LLM model override")
    p_ask.add_argument("--top-k", "-k", type=int, default=4, help="Retrieved passages count")
    p_ask.add_argument("--hybrid", action="store_true", help="Use hybrid BM25 + dense semantic vector search")
    p_ask.set_defaults(func=cmd_ask)

    # Chat
    p_chat = subparsers.add_parser("chat", help="Start an interactive chat session")
    p_chat.add_argument("--repo", "-r", default=None, help="Repository ID to query")
    p_chat.add_argument("--model", "-m", default=None, help="LLM model override")
    p_chat.add_argument("--top-k", "-k", type=int, default=4, help="Retrieved passages count")
    p_chat.add_argument("--hybrid", action="store_true", help="Use hybrid BM25 + dense semantic vector search")
    p_chat.set_defaults(func=cmd_chat)

    # List
    p_list = subparsers.add_parser("list", help="List cached repositories and indexes")
    p_list.set_defaults(func=cmd_list)

    # Status
    p_status = subparsers.add_parser("status", help="Check system status and Ollama availability")
    p_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
