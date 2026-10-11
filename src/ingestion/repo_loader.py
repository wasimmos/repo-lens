"""Repository loader for RepoLens.

Handles loading local directories or cloning remote GitHub repositories.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_DIR = PROJECT_DIR / "data" / "repos"


class RepositoryError(Exception):
    """Base exception for repository operations."""


class CloneError(RepositoryError):
    """Raised when git clone fails."""


class NotFoundError(RepositoryError):
    """Raised when repository path or URL cannot be resolved."""


@dataclass
class RepoInfo:
    id: str
    name: str
    source: str
    path: Path
    is_git: bool
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    web_url: Optional[str] = None

    def __iter__(self):
        """Allow tuple unpacking: repo_id, repo_path, branch, web_url = repo_info"""
        return iter((self.id, self.path, self.branch, self.web_url))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def extract_web_url(source: str) -> Optional[str]:
    """Extract a canonical GitHub web URL from a URL, SSH string, or shorthand."""
    clean = source.strip()
    if not clean:
        return None

    # git@github.com:owner/repo.git
    ssh_match = re.match(r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", clean)
    if ssh_match:
        return f"https://github.com/{ssh_match.group(1)}/{ssh_match.group(2)}"

    # Shorthand: owner/repo
    shorthand_match = re.match(r"^([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-\.]+)$", clean)
    if shorthand_match and not clean.startswith("http"):
        repo = shorthand_match.group(2).removesuffix(".git")
        return f"https://github.com/{shorthand_match.group(1)}/{repo}"

    # HTTP/HTTPS URL
    if "github.com" in clean:
        target = clean if clean.startswith("http") else f"https://{clean}"
        parsed = urlparse(target)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1].removesuffix(".git")
            return f"https://github.com/{owner}/{repo}"

    return None


def sanitize_identifier(text: str) -> str:
    """Sanitize a string to be safely used as a directory or file name."""
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text).strip("_")
    return clean or "unnamed_repo"


def normalize_repo_source(source: str) -> tuple[str, str, str, bool]:
    """Parse and normalize repo source into (repo_id, repo_name, resolved_source, is_remote).

    Supports:
    - https://github.com/owner/repo(.git)
    - http://...
    - git@github.com:owner/repo(.git)
    - github.com/owner/repo
    - owner/repo (GitHub shorthand)
    - local paths (relative or absolute)
    """
    clean_source = source.strip()
    if not clean_source:
        raise NotFoundError("Repository source cannot be empty.")

    # Check local path first if it exists
    local_path = Path(clean_source).expanduser()
    if not local_path.is_absolute():
        cwd_local = (Path.cwd() / clean_source).resolve()
        if cwd_local.exists():
            repo_name = cwd_local.name
            repo_id = sanitize_identifier(repo_name)
            return repo_id, repo_name, str(cwd_local), False
        resolved_local = (PROJECT_DIR / clean_source).resolve()
        if resolved_local.exists():
            repo_name = resolved_local.name
            repo_id = sanitize_identifier(repo_name)
            return repo_id, repo_name, str(resolved_local), False
    elif local_path.exists():
        repo_name = local_path.name
        repo_id = sanitize_identifier(repo_name)
        return repo_id, repo_name, str(local_path.resolve()), False

    # Git SSH: git@github.com:owner/repo.git
    ssh_match = re.match(r"^git@([^:]+):([^/]+)/([^/]+?)(?:\.git)?$", clean_source)
    if ssh_match:
        _, owner, repo = ssh_match.groups()
        repo_id = sanitize_identifier(f"{owner}_{repo}")
        return repo_id, repo, clean_source, True

    # Shorthand: owner/repo (e.g. pallets/flask)
    shorthand_match = re.match(r"^([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-\.]+)$", clean_source)
    if shorthand_match:
        owner, repo = shorthand_match.groups()
        if repo.endswith(".git"):
            repo = repo[:-4]
        repo_id = sanitize_identifier(f"{owner}_{repo}")
        resolved_url = f"https://github.com/{owner}/{repo}.git"
        return repo_id, repo, resolved_url, True

    # URL: https:// or http:// or github.com/...
    url_target = clean_source
    if url_target.startswith("github.com/"):
        url_target = "https://" + url_target

    parsed = urlparse(url_target)
    if parsed.scheme in ("http", "https"):
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) >= 2:
            owner = path_parts[-2]
            repo = path_parts[-1]
            if repo.endswith(".git"):
                repo = repo[:-4]
            repo_id = sanitize_identifier(f"{owner}_{repo}")
            resolved_url = url_target if url_target.endswith(".git") else f"{url_target}.git"
            return repo_id, repo, resolved_url, True
        elif len(path_parts) == 1:
            repo = path_parts[0].removesuffix(".git")
            repo_id = sanitize_identifier(repo)
            return repo_id, repo, url_target, True

    # If it reached here and doesn't look like a URL, assume local path that doesn't exist yet
    repo_name = local_path.name or sanitize_identifier(clean_source)
    repo_id = sanitize_identifier(repo_name)
    return repo_id, repo_name, str(local_path), False


def get_git_metadata(repo_path: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Retrieve the current commit SHA, branch name, and remote web URL if git is present."""
    if not (repo_path / ".git").exists():
        return None, None, None

    commit_sha = None
    branch = None
    web_url = None

    try:
        sha_proc = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit_sha = sha_proc.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        branch_proc = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        branch = branch_proc.stdout.strip()
        if branch == "HEAD":
            branch = None
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        remote_proc = subprocess.run(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        remote_raw = remote_proc.stdout.strip()
        if remote_raw:
            web_url = extract_web_url(remote_raw)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return commit_sha, branch, web_url


def clone_repository(
    remote_url: str,
    target_dir: Path,
    depth: int = 1,
    branch: Optional[str] = None,
) -> None:
    """Clone a git repository to target directory."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone"]
    if depth and depth > 0:
        cmd.extend(["--depth", str(depth)])
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([remote_url, str(target_dir)])

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            env=env,
        )
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            raise CloneError(f"Failed to clone {remote_url}: {err_msg}")
    except FileNotFoundError as err:
        raise CloneError("git executable not found on system PATH.") from err
    except subprocess.TimeoutExpired as err:
        raise CloneError(f"Cloning {remote_url} timed out after 300s.") from err


def load_repository(
    source: str,
    storage_dir: Optional[Path] = None,
    force_clone: bool = False,
    depth: int = 1,
    branch: Optional[str] = None,
    force: bool = False,
) -> RepoInfo:
    """Load or clone a repository from a local path or remote URL."""
    storage = storage_dir or DEFAULT_STORAGE_DIR
    repo_id, repo_name, resolved_source, is_remote = normalize_repo_source(source)
    should_force = force or force_clone

    if not is_remote:
        local_dir = Path(resolved_source)
        if not local_dir.exists():
            raise NotFoundError(f"Local repository path does not exist: {local_dir}")
        if not local_dir.is_dir():
            raise RepositoryError(f"Specified path is not a directory: {local_dir}")

        commit_sha, git_branch, web_url = get_git_metadata(local_dir)
        if not web_url:
            web_url = extract_web_url(source)

        return RepoInfo(
            id=repo_id,
            name=repo_name,
            source=source,
            path=local_dir.resolve(),
            is_git=(local_dir / ".git").exists(),
            commit_sha=commit_sha,
            branch=git_branch,
            web_url=web_url,
        )

    # Remote repository handling
    target_path = storage / repo_id

    if target_path.exists():
        if should_force:
            shutil.rmtree(target_path, ignore_errors=True)
            clone_repository(resolved_source, target_path, depth=depth, branch=branch)
        else:
            if not any(target_path.iterdir()):
                shutil.rmtree(target_path, ignore_errors=True)
                clone_repository(resolved_source, target_path, depth=depth, branch=branch)
    else:
        clone_repository(resolved_source, target_path, depth=depth, branch=branch)

    commit_sha, git_branch, web_url = get_git_metadata(target_path)
    if not web_url:
        web_url = extract_web_url(resolved_source) or extract_web_url(source)

    return RepoInfo(
        id=repo_id,
        name=repo_name,
        source=source,
        path=target_path.resolve(),
        is_git=True,
        commit_sha=commit_sha,
        branch=branch or git_branch,
        web_url=web_url,
    )


def list_cached_repos(storage_dir: Optional[Path] = None) -> list[RepoInfo]:
    """Scan and list all locally available repositories."""
    storage = storage_dir or DEFAULT_STORAGE_DIR
    results: list[RepoInfo] = []
    seen_paths: set[Path] = set()

    # Check sample_repo if it exists
    sample_dir = PROJECT_DIR / "data" / "sample_repo"
    if sample_dir.exists() and sample_dir.is_dir():
        sha, branch, web_url = get_git_metadata(sample_dir)
        info = RepoInfo(
            id="sample_repo",
            name="sample_repo",
            source=str(sample_dir),
            path=sample_dir.resolve(),
            is_git=(sample_dir / ".git").exists(),
            commit_sha=sha,
            branch=branch,
            web_url=web_url,
        )
        results.append(info)
        seen_paths.add(sample_dir.resolve())

    # Check cached repos directory
    if storage.exists() and storage.is_dir():
        for item in sorted(storage.iterdir()):
            if item.is_dir() and item.resolve() not in seen_paths:
                sha, branch, web_url = get_git_metadata(item)
                repo_id = item.name
                results.append(
                    RepoInfo(
                        id=repo_id,
                        name=repo_id,
                        source=str(item),
                        path=item.resolve(),
                        is_git=(item / ".git").exists(),
                        commit_sha=sha,
                        branch=branch,
                        web_url=web_url,
                    )
                )
                seen_paths.add(item.resolve())

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RepoLens Repository Loader")
    parser.add_argument(
        "source",
        nargs="?",
        help="GitHub URL, user/repo shorthand, or local directory path",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all cached/local repositories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-clone if repository already exists locally",
    )
    args = parser.parse_args()

    if args.list:
        repos = list_cached_repos()
        print(f"Found {len(repos)} repositories:")
        for r in repos:
            print(f"- [{r.id}] {r.name} -> {r.path} (git: {r.is_git}, branch: {r.branch or 'N/A'})")
        return

    if not args.source:
        parser.print_help()
        return

    print(f"Loading repository from: {args.source}")
    info = load_repository(args.source, force_clone=args.force)
    print("\nRepository successfully loaded:")
    print(f"  ID:         {info.id}")
    print(f"  Name:       {info.name}")
    print(f"  Local Path: {info.path}")
    print(f"  Is Git:     {info.is_git}")
    if info.branch:
        print(f"  Branch:     {info.branch}")
    if info.commit_sha:
        print(f"  Commit:     {info.commit_sha}")


if __name__ == "__main__":
    main()
