"""RepoLens CLI - Unified Command Line Interface for GitHub Codebase RAG.

Provides user-friendly commands to clone, index, search, ask questions, and chat with
any GitHub or local repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ingestion.indexer import INDEXES_DIR, get_repo_metadata, index_repository
from src.ingestion.repo_loader import DEFAULT_STORAGE_DIR, list_cached_repos, load_repository
from src.llm.answer import answer_question
from src.llm.provider import MODEL, OLLAMA_URL, check_ollama_available
from src.retrieval.search import display_results, list_available_indexes, load_index, search

ACTIVE_REPO_FILE = PROJECT_DIR / "data" / ".active_repo"


# Terminal Colors & Styling
class Style:
    ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    BOLD = "\033[1m" if ENABLED else ""
    DIM = "\033[2m" if ENABLED else ""
    GREEN = "\033[32m" if ENABLED else ""
    CYAN = "\033[36m" if ENABLED else ""
    YELLOW = "\033[33m" if ENABLED else ""
    RED = "\033[31m" if ENABLED else ""
    MAGENTA = "\033[35m" if ENABLED else ""
    RESET = "\033[0m" if ENABLED else ""


def get_active_repo() -> Optional[str]:
    """Retrieve the currently active repository ID."""
    if ACTIVE_REPO_FILE.exists():
        try:
            name = ACTIVE_REPO_FILE.read_text(encoding="utf-8").strip()
            if name:
                return name
        except Exception:
            pass
    return None


def set_active_repo(repo_id: str) -> None:
    """Save the active repository ID."""
    try:
        ACTIVE_REPO_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_REPO_FILE.write_text(repo_id.strip(), encoding="utf-8")
    except Exception:
        pass


def resolve_repo_id(requested: Optional[str]) -> tuple[str, str]:
    """Resolve repository identifier from argument, active state, or available indexes.

    Returns:
        tuple of (repo_id_or_none, display_title)
    """
    available = list_available_indexes()

    if requested:
        req_clean = requested.strip()
        # Direct match
        if req_clean in available:
            return req_clean, req_clean
        # Prefix or substring match (e.g. 'flask' -> 'pallets_flask' or 'sample_repo')
        for av in available:
            if req_clean.lower() in av.lower():
                return av, av
        return req_clean, req_clean

    # Fall back to active repo
    active = get_active_repo()
    if active and active in available:
        return active, f"{active} (active)"

    # If only one repository exists, use it automatically
    if len(available) == 1 and available[0] != "default":
        return available[0], available[0]

    # Global default
    return "default", "global index"


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
    source = args.source.strip()
    print(f"\n{Style.BOLD}🔍 Ingesting repository:{Style.RESET} {Style.CYAN}{source}{Style.RESET}")

    try:
        info = load_repository(source, force_clone=args.force)
    except Exception as err:
        print(f"\n{Style.RED}✖ Failed to load repository:{Style.RESET} {err}")
        return

    print(f"  {Style.DIM}↳ Cached at:{Style.RESET} {info.path}")
    print(f"  {Style.DIM}↳ ID:{Style.RESET} {info.id} | {Style.DIM}Branch:{Style.RESET} {info.branch or 'main'}")

    print(f"\n{Style.YELLOW}⚡ Parsing files and building index...{Style.RESET}")
    index_file, records, stats = index_repository(info, include_all_files=not args.python_only)

    # Set as active repository
    set_active_repo(info.id)

    print(f"\n{Style.GREEN}✔ Ingestion successful!{Style.RESET}")
    print(f"  • Repository ID:  {Style.BOLD}{stats['repo_id']}{Style.RESET} {Style.GREEN}(set as active){Style.RESET}")
    print(f"  • Files scanned:  {stats['scanned_files']}")
    print(f"  • Total passages: {stats['total_records']}")
    print(f"  • Index saved to: {index_file}")

    print(f"\n{Style.BOLD}File breakdown:{Style.RESET}")
    for ext, count in stats["file_types"].items():
        print(f"  • {ext:12}: {count} files")

    print(f"\n{Style.DIM}💡 Next steps:{Style.RESET}")
    print(f"  Ask a question:  {Style.CYAN}repolens ask \"Explain the project structure\"{Style.RESET}")
    print(f"  Start a chat:    {Style.CYAN}repolens chat{Style.RESET}")
    print(f"  Search symbols:  {Style.CYAN}repolens search main{Style.RESET}\n")


def cmd_search(args: argparse.Namespace) -> None:
    """Search code and documentation passages in an indexed repository."""
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    if not query.strip():
        print(f"{Style.RED}Please provide a search query.{Style.RESET}")
        return

    repo_id, repo_desc = resolve_repo_id(args.repo)
    records = load_index(repo_id=repo_id)
    meta = get_repo_metadata(repo_id)
    mode_desc = "hybrid (BM25 + LSI)" if getattr(args, "hybrid", False) else "BM25"

    print(f"\n{Style.BOLD}🔍 Searching [{repo_desc}]{Style.RESET} ({len(records)} passages | {mode_desc}) for: '{Style.CYAN}{query}{Style.RESET}'")
    results = search(query, records, top_k=args.top_k, hybrid=getattr(args, "hybrid", False))
    display_results(results, meta=meta)


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question about a repository and receive a cited response."""
    question = " ".join(args.question) if isinstance(args.question, list) else args.question
    if not question.strip():
        print(f"{Style.RED}Please provide a question.{Style.RESET}")
        return

    repo_id, repo_desc = resolve_repo_id(args.repo)
    model = args.model or MODEL

    meta = get_repo_metadata(repo_id)
    web_url = meta.get("web_url")
    branch = meta.get("branch", "main")
    repo_path = meta.get("repo_path")

    print(f"\n{Style.BOLD}🤖 Querying [{repo_desc}]{Style.RESET} using model '{Style.CYAN}{model}{Style.RESET}'...")

    try:
        answer, results = answer_question(
            question,
            repo_id=repo_id,
            top_k=args.top_k,
            model=model,
            hybrid=getattr(args, "hybrid", False),
        )
    except Exception as err:
        print(f"\n{Style.RED}✖ Error generating answer:{Style.RESET} {err}\n")
        return

    if results:
        print(f"\n{Style.BOLD}--- Retrieved Evidence ---{Style.RESET}")
        for score, rec in results:
            link, term_loc = get_source_link(rec, web_url, branch, repo_path)
            print(f"• [{rec.get('kind', 'code')}] {Style.BOLD}{rec.get('name', 'unnamed')}{Style.RESET}")
            print(f"  Location: {term_loc}")
            if link:
                print(f"  Link:     {Style.CYAN}{link}{Style.RESET}")

    print(f"\n{Style.BOLD}--- RepoLens Answer ---{Style.RESET}")
    print(answer)
    print()


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive multi-turn Q&A terminal loop."""
    repo_id, repo_desc = resolve_repo_id(args.repo)
    model = args.model or MODEL
    records = load_index(repo_id=repo_id)
    meta = get_repo_metadata(repo_id)
    web_url = meta.get("web_url")
    branch = meta.get("branch", "main")
    repo_path = meta.get("repo_path")
    hybrid = getattr(args, "hybrid", False)

    print(f"\n{Style.BOLD}=== RepoLens Interactive Chat ==={Style.RESET}")
    print(f"Target repository: {Style.CYAN}[{repo_desc}]{Style.RESET} ({len(records)} passages)")
    print(f"Active LLM:        {model}")
    print(f"{Style.DIM}Commands: /use <repo>, /list, /hybrid, /exit{Style.RESET}\n")

    while True:
        try:
            prompt_str = f"{Style.BOLD}repolens [{repo_id}]{Style.RESET} > "
            query = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if query.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if not query:
            continue

        # In-chat slash commands
        if query.startswith("/"):
            parts = query.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in {"/help"}:
                print("\nIn-chat commands:")
                print("  /use <repo>     Switch active repository")
                print("  /list           List available repositories")
                print("  /hybrid         Toggle hybrid search on/off")
                print("  /exit, /quit    Leave chat session\n")
                continue
            elif cmd in {"/use", "/repo"}:
                if not arg:
                    print(f"{Style.RED}Usage: /use <repo_name>{Style.RESET}")
                else:
                    new_id, new_desc = resolve_repo_id(arg)
                    try:
                        records = load_index(repo_id=new_id)
                        repo_id = new_id
                        repo_desc = new_desc
                        meta = get_repo_metadata(repo_id)
                        web_url = meta.get("web_url")
                        branch = meta.get("branch", "main")
                        repo_path = meta.get("repo_path")
                        set_active_repo(repo_id)
                        print(f"{Style.GREEN}✔ Switched to [{repo_id}] ({len(records)} passages){Style.RESET}\n")
                    except Exception as err:
                        print(f"{Style.RED}✖ Failed to load [{arg}]: {err}{Style.RESET}\n")
                continue
            elif cmd in {"/list", "/repos"}:
                indexes = list_available_indexes()
                print(f"\nAvailable repositories ({len(indexes)}):")
                for idx in indexes:
                    star = f"{Style.GREEN}* (active){Style.RESET}" if idx == repo_id else ""
                    print(f"  • {idx} {star}")
                print()
                continue
            elif cmd in {"/hybrid"}:
                hybrid = not hybrid
                state = f"{Style.GREEN}ON{Style.RESET}" if hybrid else f"{Style.YELLOW}OFF{Style.RESET}"
                print(f"Hybrid search is now {state}.\n")
                continue
            else:
                print(f"{Style.YELLOW}Unknown command '{cmd}'. Type /help for options.{Style.RESET}\n")
                continue

        try:
            answer, results = answer_question(
                query,
                records=records,
                top_k=args.top_k,
                model=model,
                hybrid=hybrid,
                web_url=web_url,
            )

            if results:
                print(f"\n{Style.BOLD}[Evidence]{Style.RESET}")
                for score, rec in results:
                    link, term_loc = get_source_link(rec, web_url, branch, repo_path)
                    print(f"  • [{rec.get('kind', 'code')}] {Style.BOLD}{rec.get('name', 'unnamed')}{Style.RESET} ({term_loc})")
                    if link:
                        print(f"    ➔ {Style.CYAN}{link}{Style.RESET}")

            print(f"\n{Style.BOLD}[Answer]{Style.RESET}")
            print(answer)
            print(f"{Style.DIM}" + "─" * 60 + f"{Style.RESET}\n")
        except Exception as err:
            print(f"\n{Style.RED}✖ Error:{Style.RESET} {err}\n")


def cmd_use(args: argparse.Namespace) -> None:
    """Set the active repository for future commands."""
    available = list_available_indexes()
    target = args.repo.strip()

    resolved_id, _ = resolve_repo_id(target)
    if resolved_id not in available:
        print(f"{Style.RED}✖ Repository '{target}' not found.{Style.RESET}")
        print(f"Available repositories: {', '.join(available)}")
        return

    set_active_repo(resolved_id)
    print(f"{Style.GREEN}✔ Active repository set to:{Style.RESET} {Style.BOLD}{resolved_id}{Style.RESET}")
    print(f"Future queries will target this repository by default.")


def cmd_list(args: argparse.Namespace) -> None:
    """List all cached repositories and available indexes."""
    repos = list_cached_repos()
    indexes = list_available_indexes()
    active = get_active_repo()

    print(f"\n{Style.BOLD}=== Available Repositories ==={Style.RESET}")
    if not indexes:
        print("  No repositories indexed yet. Run: repolens ingest <github-url>")
        return

    for idx in indexes:
        is_active = (idx == active)
        active_badge = f" {Style.GREEN}✔ [active]{Style.RESET}" if is_active else ""
        meta = get_repo_metadata(idx)
        total = meta.get("total_records", "N/A")
        branch = meta.get("branch", "main")
        web = meta.get("web_url") or "local"

        print(f"  • {Style.BOLD}{idx}{Style.RESET}{active_badge}")
        print(f"    {Style.DIM}↳ Passages: {total} | Branch: {branch} | URL: {web}{Style.RESET}")

    print(f"\n{Style.DIM}Switch active repo: repolens use <repo_name>{Style.RESET}\n")


def cmd_remove(args: argparse.Namespace) -> None:
    """Delete a cached repository and its index."""
    repo_id = args.repo.strip()
    available = list_available_indexes()

    if repo_id not in available and repo_id != "default":
        print(f"{Style.RED}✖ Repository '{repo_id}' not found.{Style.RESET}")
        return

    # Delete index files
    index_file = INDEXES_DIR / f"{repo_id}.json"
    meta_file = INDEXES_DIR / f"{repo_id}_meta.json"
    if index_file.exists():
        index_file.unlink(missing_ok=True)
    if meta_file.exists():
        meta_file.unlink(missing_ok=True)

    # Delete cloned repo if inside data/repos
    repo_dir = DEFAULT_STORAGE_DIR / repo_id
    if repo_dir.exists() and repo_dir.is_dir():
        shutil.rmtree(repo_dir, ignore_errors=True)

    # Clear active if it was this one
    if get_active_repo() == repo_id:
        if ACTIVE_REPO_FILE.exists():
            ACTIVE_REPO_FILE.unlink(missing_ok=True)

    print(f"{Style.GREEN}✔ Removed repository '{repo_id}' and deleted its index.{Style.RESET}\n")


def cmd_status(args: argparse.Namespace) -> None:
    """Check system and provider status."""
    is_online, msg = check_ollama_available()
    active = get_active_repo() or "None"

    print(f"\n{Style.BOLD}=== RepoLens System Status ==={Style.RESET}")
    print(f"  • Python:        {sys.version.split()[0]}")
    print(f"  • Active repo:   {Style.CYAN}{active}{Style.RESET}")
    print(f"  • Project root:  {PROJECT_DIR}")
    print(f"  • Ollama URL:    {OLLAMA_URL}")
    print(f"  • Default model: {MODEL}")
    status_str = f"{Style.GREEN}ONLINE{Style.RESET}" if is_online else f"{Style.RED}OFFLINE{Style.RESET}"
    print(f"  • Ollama status: {status_str} - {msg}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repolens",
        description="RepoLens 🔍 - Intelligent GitHub Repository RAG & Codebase Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  repolens ingest pallets/flask               Clone and index a GitHub repository
  repolens ask "How does routing work?"       Ask a question with verified citations
  repolens ask How does routing work?         Quotes are optional!
  repolens "What is this project?"            Direct question without 'ask'!
  repolens chat                               Start interactive chat with active repo
  repolens search route                       Search code and documentation
  repolens use pallets_flask                  Set default repository
  repolens list                               List all indexed repositories
  repolens status                             Check Ollama and system status
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest / Clone / Add
    p_ingest = subparsers.add_parser(
        "ingest",
        aliases=["clone", "add", "load"],
        help="Clone/load and index a repository (aliases: clone, add)",
    )
    p_ingest.add_argument("source", help="GitHub URL, owner/repo shorthand, or local directory path")
    p_ingest.add_argument("--force", action="store_true", help="Force re-clone if already cached")
    p_ingest.add_argument("--python-only", action="store_true", help="Only index Python files")
    p_ingest.set_defaults(func=cmd_ingest)

    # Ask
    p_ask = subparsers.add_parser(
        "ask",
        help="Ask a question about the repository and receive a cited response",
    )
    p_ask.add_argument("question", nargs="+", help="Question about the repository (quotes optional)")
    p_ask.add_argument("--repo", "-r", default=None, help="Repository ID (defaults to active repo)")
    p_ask.add_argument("--model", "-m", default=None, help="LLM model override")
    p_ask.add_argument("--top-k", "-k", type=int, default=4, help="Retrieved passages count")
    p_ask.add_argument("--hybrid", action="store_true", help="Use hybrid BM25 + dense semantic vector search")
    p_ask.set_defaults(func=cmd_ask)

    # Search
    p_search = subparsers.add_parser(
        "search",
        help="Search code and docs passages (quotes optional)",
    )
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--repo", "-r", default=None, help="Repository ID (defaults to active repo)")
    p_search.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--hybrid", action="store_true", help="Use hybrid BM25 + dense semantic vector search")
    p_search.set_defaults(func=cmd_search)

    # Chat / Interactive
    p_chat = subparsers.add_parser(
        "chat",
        aliases=["talk", "interactive"],
        help="Start an interactive chat session (aliases: talk, interactive)",
    )
    p_chat.add_argument("--repo", "-r", default=None, help="Repository ID (defaults to active repo)")
    p_chat.add_argument("--model", "-m", default=None, help="LLM model override")
    p_chat.add_argument("--top-k", "-k", type=int, default=4, help="Retrieved passages count")
    p_chat.add_argument("--hybrid", action="store_true", help="Use hybrid BM25 + dense semantic vector search")
    p_chat.set_defaults(func=cmd_chat)

    # Use / Select / Switch
    p_use = subparsers.add_parser(
        "use",
        aliases=["select", "switch"],
        help="Set the active repository for future queries (aliases: select, switch)",
    )
    p_use.add_argument("repo", help="Repository ID or name to set as active")
    p_use.set_defaults(func=cmd_use)

    # List / Repos
    p_list = subparsers.add_parser(
        "list",
        aliases=["ls", "repos"],
        help="List all cached and indexed repositories (aliases: ls, repos)",
    )
    p_list.set_defaults(func=cmd_list)

    # Remove / Delete / Rm
    p_remove = subparsers.add_parser(
        "remove",
        aliases=["rm", "delete"],
        help="Delete a cached repository and its index (aliases: rm, delete)",
    )
    p_remove.add_argument("repo", help="Repository ID to remove")
    p_remove.set_defaults(func=cmd_remove)

    # Status
    p_status = subparsers.add_parser(
        "status",
        help="Check system status and Ollama availability",
    )
    p_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # User convenience: Check if the first argument is a question or search query directly!
    first_arg = sys.argv[1]
    known_commands = {
        "ingest", "clone", "add", "load",
        "ask",
        "search",
        "chat", "talk", "interactive",
        "use", "select", "switch",
        "list", "ls", "repos",
        "remove", "rm", "delete",
        "status",
        "-h", "--help", "-v", "--version",
    }

    if not first_arg.startswith("-") and first_arg not in known_commands:
        # User typed: repolens "What does url_for do?" or repolens How does routing work?
        # Route directly to ask!
        modified_argv = [sys.argv[0], "ask"] + sys.argv[1:]
        args = parser.parse_args(modified_argv[1:])
        cmd_ask(args)
        return

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
