"""Fast, zero-dependency HTTP server and REST API for RepoLens Web & Mobile App."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import socket
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

PROJECT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.cli import (
    ACTIVE_REPO_FILE,
    get_active_repo,
    get_source_link,
    resolve_repo_id,
    set_active_repo,
)
from src.ingestion.indexer import INDEXES_DIR, get_repo_metadata, index_repository
from src.ingestion.repo_loader import DEFAULT_STORAGE_DIR, list_cached_repos, load_repository
from src.llm.answer import answer_question
from src.llm.provider import (
    SUPPORTED_PROVIDERS,
    check_provider_status,
    get_active_model,
    get_active_provider,
    list_providers,
    set_active_model,
    set_active_provider,
)
from src.retrieval.search import list_available_indexes, load_index, search


def get_local_ip() -> str:
    """Retrieve LAN IP address of this host for mobile connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class RepoLensHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP request handler serving REST API and responsive SPA static files."""

    server_version = "RepoLensServer/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy default request logging, keep clean console output."""
        pass

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message, "success": False}, status=status)

    def _read_json_body(self) -> dict[str, Any]:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len <= 0:
            return {}
        body = self.rfile.read(content_len).decode("utf-8")
        return json.loads(body) if body else {}

    def _serve_static(self, relative_path: str) -> None:
        target = (STATIC_DIR / relative_path).resolve()
        if not target.is_relative_to(STATIC_DIR) or not target.exists() or target.is_dir():
            target = STATIC_DIR / "index.html"

        mime, _ = mimetypes.guess_type(str(target))
        if target.name.endswith(".js"):
            mime = "application/javascript"
        elif target.name.endswith(".css"):
            mime = "text/css"
        elif target.name.endswith(".json"):
            mime = "application/json"

        mime = mime or "text/html"
        content = target.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # 1. API: System & Provider Status
        if path == "/api/status":
            active_p = get_active_provider()
            active_m = get_active_model(active_p)
            is_ready, msg = check_provider_status(active_p)
            active_repo = get_active_repo()

            self._send_json({
                "active_repo": active_repo,
                "active_provider": active_p,
                "active_model": active_m,
                "provider_ready": is_ready,
                "provider_msg": msg,
                "lan_ip": get_local_ip(),
                "public_url": CURRENT_PUBLIC_URL,
                "server_host": self.server.server_address[0],
                "server_port": self.server.server_address[1],
            })
            return

        # 2. API: List Available Repositories
        if path == "/api/repos":
            indexes = list_available_indexes()
            active = get_active_repo()
            data = []
            for idx in indexes:
                meta = get_repo_metadata(idx)
                data.append({
                    "id": idx,
                    "name": meta.get("repo_id", idx),
                    "is_active": (idx == active),
                    "passages": meta.get("total_records", 0),
                    "scanned_files": meta.get("scanned_files", 0),
                    "branch": meta.get("branch", "main"),
                    "web_url": meta.get("web_url"),
                    "created_at": meta.get("created_at"),
                })
            self._send_json({"repos": data, "active": active})
            return

        # 3. API: List LLM Providers
        if path == "/api/providers":
            providers = list_providers()
            active_p = get_active_provider()
            self._send_json({"providers": providers, "active": active_p})
            return

        # 4. Static Files / SPA Routes
        if path == "/" or path == "/index.html":
            self._serve_static("index.html")
        elif path == "/manifest.json":
            self._serve_static("manifest.json")
        elif path.startswith("/static/"):
            rel = path[len("/static/") :]
            self._serve_static(rel)
        else:
            # Fallback to SPA root for mobile client-side routing
            self._serve_static("index.html")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            body = self._read_json_body()
        except Exception as err:
            self._send_error(f"Malformed JSON body: {err}", status=400)
            return

        # 1. API: Switch Active Repo
        if path == "/api/repos/use":
            repo_id = body.get("repo", "").strip()
            if not repo_id:
                self._send_error("Parameter 'repo' is required")
                return
            resolved, _ = resolve_repo_id(repo_id)
            set_active_repo(resolved)
            self._send_json({"success": True, "active_repo": resolved})
            return

        # 2. API: Ingest New Repository
        if path == "/api/repos/ingest":
            source = body.get("source", "").strip()
            if not source:
                self._send_error("Parameter 'source' is required")
                return

            force = bool(body.get("force", False))
            python_only = bool(body.get("python_only", False))

            try:
                repo_info = load_repository(source, force=force)
                repo_id, repo_path, branch, web_url = repo_info
                idx_file, records, stats = index_repository(
                    repo_info,
                    repo_id=repo_id,
                    python_only=python_only,
                    web_url=web_url,
                    branch=branch,
                )
                set_active_repo(repo_id)
                self._send_json({
                    "success": True,
                    "repo_id": repo_id,
                    "passages": len(records),
                    "branch": branch,
                    "web_url": web_url,
                })
            except Exception as err:
                self._send_error(f"Ingestion failed: {err}", status=500)
            return

        # 3. API: Switch or Configure LLM Provider
        if path == "/api/providers/switch":
            provider = body.get("provider", "").strip().lower()
            model = body.get("model", "").strip() or None
            api_key = body.get("api_key", "").strip() or None
            url = body.get("url", "").strip() or None

            if provider not in SUPPORTED_PROVIDERS:
                self._send_error(f"Unsupported provider '{provider}'")
                return

            try:
                set_active_provider(provider, model=model, api_key=api_key, url=url)
                active_model = get_active_model(provider)
                self._send_json({
                    "success": True,
                    "active_provider": provider,
                    "active_model": active_model,
                })
            except Exception as err:
                self._send_error(str(err), status=500)
            return

        # 4. API: Code & Docs Search
        if path == "/api/search":
            query = body.get("query", "").strip()
            if not query:
                self._send_error("Query string is required")
                return

            repo_id, _ = resolve_repo_id(body.get("repo"))
            top_k = int(body.get("top_k", 5))
            hybrid = bool(body.get("hybrid", False))

            try:
                records = load_index(repo_id=repo_id)
                meta = get_repo_metadata(repo_id)
                web_url = meta.get("web_url")
                branch = meta.get("branch", "main")
                repo_path = meta.get("repo_path")

                results = search(query, records, top_k=top_k, hybrid=hybrid)

                formatted = []
                for score, rec in results:
                    link, term_loc = get_source_link(rec, web_url, branch, repo_path)
                    formatted.append({
                        "score": round(score, 3),
                        "name": rec.get("name", "unnamed"),
                        "kind": rec.get("kind", "code"),
                        "file": rec.get("file", ""),
                        "start_line": rec.get("start_line", 1),
                        "end_line": rec.get("end_line", 1),
                        "location": term_loc,
                        "link": link,
                        "docstring": rec.get("docstring", ""),
                        "code": rec.get("code", ""),
                    })

                self._send_json({
                    "query": query,
                    "repo_id": repo_id,
                    "results": formatted,
                    "count": len(formatted),
                })
            except Exception as err:
                self._send_error(f"Search failed: {err}", status=500)
            return

        # 5. API: Grounded Q&A Asking
        if path == "/api/ask":
            question = body.get("question", "").strip()
            if not question:
                self._send_error("Question string is required")
                return

            repo_id, _ = resolve_repo_id(body.get("repo"))
            provider = body.get("provider") or get_active_provider()
            model = body.get("model") or get_active_model(provider)
            top_k = int(body.get("top_k", 4))
            hybrid = bool(body.get("hybrid", False))

            try:
                records = load_index(repo_id=repo_id)
                meta = get_repo_metadata(repo_id)
                web_url = meta.get("web_url")
                branch = meta.get("branch", "main")
                repo_path = meta.get("repo_path")

                answer, results = answer_question(
                    question=question,
                    repo_id=repo_id,
                    records=records,
                    top_k=top_k,
                    model=model,
                    provider=provider,
                    hybrid=hybrid,
                    web_url=web_url,
                )

                evidence = []
                for score, rec in results:
                    link, term_loc = get_source_link(rec, web_url, branch, repo_path)
                    evidence.append({
                        "score": round(score, 3),
                        "name": rec.get("name", "unnamed"),
                        "kind": rec.get("kind", "code"),
                        "file": rec.get("file", ""),
                        "start_line": rec.get("start_line", 1),
                        "end_line": rec.get("end_line", 1),
                        "location": term_loc,
                        "link": link,
                        "docstring": rec.get("docstring", ""),
                        "code": rec.get("code", ""),
                    })

                self._send_json({
                    "success": True,
                    "question": question,
                    "answer": answer,
                    "repo_id": repo_id,
                    "provider": provider,
                    "model": model,
                    "evidence": evidence,
                })
            except Exception as err:
                self._send_error(f"Answer generation error: {err}", status=500)
            return

        self._send_error(f"Endpoint '{path}' not found", status=404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/api/repos/"):
            repo_id = path[len("/api/repos/") :]
            if not repo_id:
                self._send_error("Repository ID required", status=400)
                return

            # Delete index files
            idx_file = INDEXES_DIR / f"{repo_id}.json"
            meta_file = INDEXES_DIR / f"{repo_id}_meta.json"
            if idx_file.exists():
                idx_file.unlink(missing_ok=True)
            if meta_file.exists():
                meta_file.unlink(missing_ok=True)

            # Delete cached repo dir
            repo_dir = DEFAULT_STORAGE_DIR / repo_id
            if repo_dir.exists() and repo_dir.is_dir():
                shutil.rmtree(repo_dir, ignore_errors=True)

            if get_active_repo() == repo_id and ACTIVE_REPO_FILE.exists():
                ACTIVE_REPO_FILE.unlink(missing_ok=True)

            self._send_json({"success": True, "deleted": repo_id})
            return

        self._send_error(f"Endpoint '{path}' not found", status=404)


CURRENT_PUBLIC_URL: Optional[str] = None


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    open_browser: bool = False,
    cloud_tunnel: bool = False,
) -> None:
    """Start the multi-threaded RepoLens Web and Mobile application server."""
    global CURRENT_PUBLIC_URL

    # Environment variable overrides (essential for cloud platforms like Render / Railway / Heroku / Fly.io)
    if "PORT" in os.environ:
        try:
            port = int(os.environ["PORT"])
        except ValueError:
            pass
    if "HOST" in os.environ:
        host = os.environ["HOST"]

    lan_ip = get_local_ip()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Find open port if requested port is taken
    active_port = port
    while active_port < port + 20:
        try:
            server = ThreadingHTTPServer((host, active_port), RepoLensHandler)
            break
        except OSError:
            active_port += 1
    else:
        raise RuntimeError(f"Could not bind to any port between {port} and {port + 20}")

    desktop_url = f"http://localhost:{active_port}"
    mobile_url = f"http://{lan_ip}:{active_port}"

    # Initialize cloud tunnel if requested
    tunnel = None
    if cloud_tunnel or os.environ.get("REPOLENS_CLOUD") == "1":
        from src.web.tunnel import CloudTunnel

        print("\n  ⏳ Initializing secure public cloud tunnel...")
        tunnel = CloudTunnel(active_port)
        pub_url = tunnel.start(timeout=12)
        if pub_url:
            CURRENT_PUBLIC_URL = pub_url

    print("\n" + "=" * 68)
    print("  🚀 RepoLens 🔍 Web & Mobile Application")
    print("=" * 68)
    if CURRENT_PUBLIC_URL:
        print(f"  🌐 Public Cloud URL:  \033[1;32m{CURRENT_PUBLIC_URL}\033[0m")
        print(f"     ↳ Accessible from ANY phone or browser worldwide (4G/5G/Wi-Fi)")
    print(f"  💻 Localhost URL:     \033[1;36m{desktop_url}\033[0m")
    print(f"  📱 Phone / LAN URL:   \033[1;36m{mobile_url}\033[0m")
    print("=" * 68)
    if not CURRENT_PUBLIC_URL:
        print("  • To access from outside your Wi-Fi, run: repolens web --cloud")
    print("  • Tap 'Add to Home Screen' on your phone to install as a native app.")
    print("  Press Ctrl+C to stop the server.\n")

    if open_browser and not CURRENT_PUBLIC_URL:
        try:
            import webbrowser

            webbrowser.open(desktop_url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RepoLens Web server...")
    finally:
        if tunnel:
            tunnel.stop()
        server.server_close()

