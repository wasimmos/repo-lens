"""Command-line interface for RepoLens repository intelligence and RAG Q&A.

Provides user-friendly commands:
  repolens ingest <source>        - Load and index a repository
  repolens ask <question>         - Grounded Q&A with clickable source links
  repolens chat                   - Interactive multi-turn chat session
  repolens search <query>         - Search code and documentation passages
  repolens use <repo_name>        - Set default repository
  repolens provider [name]        - Switch or configure LLM provider (aliases: agent, llm)
  repolens model [model_name]     - Switch active LLM model
  repolens list                   - List cached repositories and indexes
  repolens status                 - Check system and provider health
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
from src.llm.provider import (
    MODEL,
    OLLAMA_URL,
    SUPPORTED_PROVIDERS,
    check_ollama_available,
    check_provider_status,
    get_active_model,
    get_active_provider,
    list_providers,
    set_active_model,
    set_active_provider,
)
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
        req = requested.strip()
        if req in available:
            return req, req
        # Fuzzy / prefix matching
        matches = [idx for idx in available if req.lower() in idx.lower()]
        if len(matches) == 1:
            return matches[0], matches[0]
        elif len(matches) > 1:
            # Prefer exact match prefix
            for m in matches:
                if m.lower().startswith(req.lower()):
                    return m, m
            return matches[0], matches[0]
        return req, req

    # Fallback to active repository
    active = get_active_repo()
    if active and active in available:
        return active, f"{active} (active)"

    # Fallback to most recent or sample_repo
    if "sample_repo" in available:
        return "sample_repo", "sample_repo (default)"
    if available:
        return available[0], f"{available[0]} (latest)"

    return "default", "default"


def get_source_link(
    record: dict[str, Any],
    web_url: Optional[str] = None,
    branch: Optional[str] = "main",
    repo_path: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Build clickable terminal link and display string for a result record."""
    file_path = record.get("file", "")
    start = record.get("start_line", 1)
    end = record.get("end_line", start)
    term_loc = f"{file_path}:{start}-{end}"

    if web_url:
        clean_base = web_url.rstrip("/")
        active_branch = branch or "main"
        url = f"{clean_base}/blob/{active_branch}/{file_path}#L{start}-L{end}"
        return url, term_loc

    if repo_path:
        abs_p = Path(repo_path) / file_path
        file_url = f"file://{abs_p.resolve()}"
        return file_url, term_loc

    return None, term_loc


def cmd_ingest(args: argparse.Namespace) -> None:
    """Clone/load and index a repository."""
    source = args.source.strip()
    print(f"\n{Style.BOLD}🚀 Ingesting repository:{Style.RESET} {Style.CYAN}{source}{Style.RESET}")

    repo_info = load_repository(source, force=args.force)
    repo_id, repo_path, branch, web_url = repo_info
    print(f"✔ Repository loaded at: {repo_path}")
    if web_url:
        print(f"✔ Remote GitHub URL:    {web_url} (branch: {branch})")

    print(f"\n⚡ Indexing source code, markdown, and config files...")
    idx_file, records, stats = index_repository(
        repo_info,
        repo_id=repo_id,
        python_only=args.python_only,
        web_url=web_url,
        branch=branch,
    )
    print(f"{Style.GREEN}✔ Successfully indexed {len(records)} passages for [{repo_id}]{Style.RESET}")

    # Set as active repository automatically
    set_active_repo(repo_id)
    print(f"{Style.DIM}✔ Set as active repository for future queries.{Style.RESET}\n")


def cmd_search(args: argparse.Namespace) -> None:
    """Search code and documentation passages."""
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    if not query.strip():
        print(f"{Style.RED}Please provide a search query.{Style.RESET}")
        return

    repo_id, repo_desc = resolve_repo_id(args.repo)
    records = load_index(repo_id=repo_id)
    meta = get_repo_metadata(repo_id)

    mode = "Hybrid BM25+LSI" if getattr(args, "hybrid", False) else "BM25"
    print(f"\n🔍 Searching [{repo_desc}] ({len(records)} passages | {mode}) for: '{Style.CYAN}{query}{Style.RESET}'")

    results = search(query, records, top_k=args.top_k, hybrid=getattr(args, "hybrid", False))
    display_results(results, meta=meta)


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question about a repository and receive a cited response."""
    question = " ".join(args.question) if isinstance(args.question, list) else args.question
    if not question.strip():
        print(f"{Style.RED}Please provide a question.{Style.RESET}")
        return

    repo_id, repo_desc = resolve_repo_id(args.repo)
    provider = getattr(args, "provider", None) or get_active_provider()
    model = args.model or get_active_model(provider)

    meta = get_repo_metadata(repo_id)
    web_url = meta.get("web_url")
    branch = meta.get("branch", "main")
    repo_path = meta.get("repo_path")

    print(f"\n{Style.BOLD}🤖 Querying [{repo_desc}]{Style.RESET} using {Style.CYAN}{provider}{Style.RESET} ('{Style.CYAN}{model}{Style.RESET}')...")

    try:
        answer, results = answer_question(
            question,
            repo_id=repo_id,
            top_k=args.top_k,
            model=model,
            provider=provider,
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
    provider = getattr(args, "provider", None) or get_active_provider()
    model = args.model or get_active_model(provider)
    records = load_index(repo_id=repo_id)
    meta = get_repo_metadata(repo_id)
    web_url = meta.get("web_url")
    branch = meta.get("branch", "main")
    repo_path = meta.get("repo_path")
    hybrid = getattr(args, "hybrid", False)

    print(f"\n{Style.BOLD}=== RepoLens Interactive Chat ==={Style.RESET}")
    print(f"Target repository: {Style.CYAN}[{repo_desc}]{Style.RESET} ({len(records)} passages)")
    print(f"Active Provider:   {Style.CYAN}{provider}{Style.RESET} (Model: {Style.BOLD}{model}{Style.RESET})")
    print(f"{Style.DIM}Commands: /use <repo>, /provider [name], /model [name], /hybrid, /exit{Style.RESET}\n")

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
                print("  /use <repo>             Switch active repository")
                print("  /list, /repos           List available repositories")
                print("  /provider [name] [mod]  View or switch LLM provider")
                print("  /model [name]           View or switch active model")
                print("  /hybrid                 Toggle hybrid search on/off")
                print("  /exit, /quit            Leave chat session\n")
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
            elif cmd in {"/provider", "/agent", "/llm"}:
                if not arg:
                    # List providers
                    providers = list_providers()
                    print("\nAvailable LLM Providers:")
                    for p in providers:
                        mark = f"{Style.GREEN}* (active){Style.RESET}" if p["name"] == provider else ""
                        print(f"  • {p['name']:<12} {p['model']} {mark}")
                    print()
                else:
                    prov_parts = arg.split(maxsplit=1)
                    new_p = prov_parts[0].lower()
                    new_m = prov_parts[1].strip() if len(prov_parts) > 1 else None
                    if new_p in SUPPORTED_PROVIDERS:
                        set_active_provider(new_p, model=new_m)
                        provider = new_p
                        model = get_active_model(new_p)
                        print(f"{Style.GREEN}✔ Switched LLM provider to '{provider}' (model: {model}){Style.RESET}\n")
                    else:
                        valid = ", ".join(SUPPORTED_PROVIDERS.keys())
                        print(f"{Style.RED}Unknown provider '{new_p}'. Supported: {valid}{Style.RESET}\n")
                continue
            elif cmd in {"/model"}:
                if not arg:
                    print(f"Current model: {model} (provider: {provider})\n")
                else:
                    set_active_model(arg, provider_name=provider)
                    model = arg
                    print(f"{Style.GREEN}✔ Switched model to: {model}{Style.RESET}\n")
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
                provider=provider,
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


def cmd_provider(args: argparse.Namespace) -> None:
    """Manage and switch LLM providers/agents."""
    if args.name:
        target = args.name.lower()
        if target not in SUPPORTED_PROVIDERS:
            matches = [k for k in SUPPORTED_PROVIDERS if target in k]
            if len(matches) == 1:
                target = matches[0]
            else:
                valid = ", ".join(SUPPORTED_PROVIDERS.keys())
                print(f"{Style.RED}Unknown provider '{args.name}'. Supported: {valid}{Style.RESET}")
                return

        set_active_provider(
            target,
            model=args.model,
            api_key=args.api_key,
            url=args.url,
        )
        spec = SUPPORTED_PROVIDERS[target]
        active_m = get_active_model(target)
        print(f"{Style.GREEN}✔ Switched active provider to '{target}' ({spec.display_name}){Style.RESET}")
        print(f"  Model: {Style.CYAN}{active_m}{Style.RESET}")
        if args.url:
            print(f"  Endpoint URL: {args.url}")
        if args.api_key:
            print(f"  API key: Saved")
        print()
        return

    # No arguments given: Display all providers
    providers = list_providers()
    active_p = get_active_provider()
    active_m = get_active_model(active_p)

    print(f"\n{Style.BOLD}=== RepoLens LLM Providers & Agents ==={Style.RESET}")
    print(f"Active Provider: {Style.CYAN}{active_p}{Style.RESET} (Model: {Style.BOLD}{active_m}{Style.RESET})\n")

    for p in providers:
        name = p["name"]
        is_active = p["is_active"]
        active_badge = f"{Style.GREEN}✔ [active]{Style.RESET}" if is_active else "○         "
        type_badge = f"{Style.YELLOW}Cloud{Style.RESET}" if p["is_cloud"] else f"{Style.CYAN}Local{Style.RESET}"

        status_info = ""
        if p["is_cloud"]:
            status_info = f"Key: {Style.GREEN}Configured{Style.RESET}" if p["has_key"] else f"Key: {Style.RED}Missing ({p['env_keys'][0]}){Style.RESET}"
        else:
            if name == "ollama":
                is_on, msg = check_ollama_available()
                status_info = f"Status: {Style.GREEN}Online{Style.RESET}" if is_on else f"Status: {Style.RED}Offline{Style.RESET}"
            else:
                status_info = f"URL: {p['url']}"

        print(f"  {active_badge} {Style.BOLD}{name:<12}{Style.RESET} [{type_badge}] {p['display_name']}")
        print(f"             ↳ Model: {p['model']} | {status_info}")
        print(f"             ↳ {Style.DIM}{p['description']}{Style.RESET}")

    print(f"\n{Style.BOLD}Quick Commands:{Style.RESET}")
    print(f"  Switch provider: {Style.CYAN}repolens provider <name>{Style.RESET}")
    print(f"  Switch model:    {Style.CYAN}repolens model <model_name>{Style.RESET}")
    print(f"  Configure:       {Style.CYAN}repolens provider <name> --api-key <KEY> [--model <MODEL>]{Style.RESET}\n")


def cmd_model(args: argparse.Namespace) -> None:
    """View or change active LLM model."""
    provider = args.provider or get_active_provider()
    if args.model_name:
        set_active_model(args.model_name, provider_name=provider)
        print(f"{Style.GREEN}✔ Active model for provider '{provider}' set to: {args.model_name}{Style.RESET}\n")
    else:
        current_model = get_active_model(provider)
        print(f"\nProvider: {Style.CYAN}{provider}{Style.RESET}")
        print(f"Model:    {Style.BOLD}{current_model}{Style.RESET}\n")
        print(f"To change: {Style.DIM}repolens model <new_model_name>{Style.RESET}\n")


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
    active_p = get_active_provider()
    active_m = get_active_model(active_p)
    is_ready, p_msg = check_provider_status(active_p)
    active_repo = get_active_repo() or "None"

    print(f"\n{Style.BOLD}=== RepoLens System Status ==={Style.RESET}")
    print(f"  • Python:          {sys.version.split()[0]}")
    print(f"  • Active repo:     {Style.CYAN}{active_repo}{Style.RESET}")
    print(f"  • Active Provider: {Style.BOLD}{active_p}{Style.RESET} (model: {Style.CYAN}{active_m}{Style.RESET})")
    status_str = f"{Style.GREEN}ONLINE / READY{Style.RESET}" if is_ready else f"{Style.RED}UNAVAILABLE{Style.RESET}"
    print(f"  • LLM status:      {status_str} - {p_msg}")
    print(f"  • Project root:    {PROJECT_DIR}\n")


def cmd_web(args: argparse.Namespace) -> None:
    """Launch the mobile-friendly RepoLens Web & Phone application."""
    from src.web.server import run_server

    is_cloud = getattr(args, "cloud", False) or getattr(args, "command", "") in {"cloud", "share"}
    run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        cloud_tunnel=is_cloud,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repolens",
        description="RepoLens 🔍 - Intelligent GitHub Repository RAG & Codebase Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  repolens web                                Launch mobile & web application
  repolens web --cloud                        Launch with instant public HTTPS cloud URL
  repolens cloud                              Shorthand for cloud website
  repolens ingest pallets/flask               Clone and index a GitHub repository
  repolens ask "How does routing work?"       Ask a question with verified citations
  repolens ask How does routing work?         Quotes are optional!
  repolens "What is this project?"            Direct question without 'ask'!
  repolens chat                               Start interactive chat with active repo
  repolens search route                       Search code and documentation
  repolens use pallets_flask                  Set default repository
  repolens provider                           List all supported LLM providers
  repolens provider openai                    Switch to OpenAI provider
  repolens model llama3.2                     Switch model for current provider
  repolens list                               List all indexed repositories
  repolens status                             Check provider and system status
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Web / UI / Serve / Cloud
    p_web = subparsers.add_parser(
        "web",
        aliases=["ui", "serve", "app", "cloud", "share"],
        help="Launch Web & Phone application (aliases: ui, serve, app, cloud, share)",
    )
    p_web.add_argument("--host", "-H", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    p_web.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    p_web.add_argument("--cloud", "-c", "--tunnel", dest="cloud", action="store_true", help="Generate public HTTPS cloud URL for worldwide access")
    p_web.add_argument("--no-browser", action="store_true", help="Do not open desktop browser automatically")
    p_web.set_defaults(func=cmd_web)

    # Ingest / Clone / Add
    p_ingest = subparsers.add_parser(
        "ingest",
        aliases=["clone", "add", "load"],
        help="Clone/load and index a repository (aliases: clone, add, load)",
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
    p_ask.add_argument("--provider", "-p", default=None, help="LLM provider override (e.g. ollama, openai, anthropic, gemini)")
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
    p_chat.add_argument("--provider", "-p", default=None, help="LLM provider override")
    p_chat.add_argument("--model", "-m", default=None, help="LLM model override")
    p_chat.add_argument("--top-k", "-k", type=int, default=4, help="Retrieved passages count")
    p_chat.add_argument("--hybrid", action="store_true", help="Use hybrid BM25 + dense semantic vector search")
    p_chat.set_defaults(func=cmd_chat)

    # Provider / Agent / LLM
    p_prov = subparsers.add_parser(
        "provider",
        aliases=["agent", "llm"],
        help="Manage and switch LLM providers/agents (aliases: agent, llm)",
    )
    p_prov.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Provider name to activate (e.g. ollama, openai, anthropic, gemini, deepseek, groq, openrouter, custom)",
    )
    p_prov.add_argument("--model", "-m", default=None, help="Model name for provider")
    p_prov.add_argument("--api-key", "-k", default=None, help="API key for cloud provider")
    p_prov.add_argument("--url", "-u", default=None, help="Endpoint URL for provider")
    p_prov.set_defaults(func=cmd_provider)

    # Model
    p_mod = subparsers.add_parser(
        "model",
        help="View or change active LLM model",
    )
    p_mod.add_argument("model_name", nargs="?", default=None, help="New model name (optional)")
    p_mod.add_argument("--provider", "-p", default=None, help="Target provider (defaults to active provider)")
    p_mod.set_defaults(func=cmd_model)

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
        help="Check system and LLM provider status",
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
        "provider", "agent", "llm",
        "model",
        "web", "ui", "serve", "app", "cloud", "share",
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
