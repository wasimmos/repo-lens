"""Search engine for RepoLens codebase RAG.

Provides code-aware BM25 lexical search with camelCase/snake_case tokenization,
exact symbol boosting, path matching, and multi-repository index support.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

INDEX_FILE = PROJECT_DIR / "data" / "index.json"
INDEXES_DIR = PROJECT_DIR / "data" / "indexes"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "how", "what",
    "where", "when", "why", "does", "do", "did", "in", "on", "to",
    "of", "for", "and", "or", "it", "this", "that", "with", "from",
    "me", "show", "find", "explain", "can", "i", "we", "be", "by",
    "at", "as", "if", "about", "into", "through", "using", "use",
}


def split_identifier(token: str) -> list[str]:
    """Split programming identifiers like snake_case, camelCase, PascalCase.

    Examples:
        'handle_request' -> ['handle_request', 'handle', 'request']
        'FlaskClient'    -> ['flaskclient', 'flask', 'client']
        'HTTPResponse200' -> ['httpresponse200', 'http', 'response', '200']
    """
    if not token:
        return []

    # Insert separator between lowercase and uppercase
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", token)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    # Insert separator between letters and digits
    s3 = re.sub(r"([a-zA-Z])([0-9])", r"\1_\2", s2)
    s4 = re.sub(r"([0-9])([a-zA-Z])", r"\1_\2", s3)

    parts = [p.lower() for p in re.split(r"[_.\-]+", s4) if p]

    tokens = [token.lower()]
    for part in parts:
        if part not in tokens and len(part) > 1:
            tokens.append(part)

    return tokens


SYNONYM_EXPANSIONS: dict[str, list[str]] = {
    "deployment": ["deploy", "run", "serve", "app", "streamlit", "docker", "pipeline", "workflow", "main", "start"],
    "deploy": ["deployment", "run", "serve", "app", "streamlit", "docker", "main"],
    "pipeline": ["workflow", "train", "model", "process", "step", "notebook", "scaler", "fit"],
    "pipelines": ["workflow", "train", "model", "process", "step", "notebook"],
    "locate": ["find", "path", "file", "location", "where"],
    "location": ["path", "file", "where", "directory"],
    "where": ["file", "path", "location"],
    "entrypoint": ["main", "app", "run", "cli", "server", "start"],
    "entry": ["main", "app", "run", "cli", "server", "start"],
    "start": ["run", "app", "main", "server", "init"],
    "database": ["db", "model", "schema", "sql", "data", "table"],
    "api": ["route", "endpoint", "url", "handler", "view", "request"],
    "endpoints": ["route", "endpoint", "url", "handler", "view"],
    "endpoint": ["route", "endpoint", "url", "handler", "view"],
    "auth": ["login", "token", "jwt", "password", "user", "authenticate"],
    "train": ["training", "model", "fit", "dataset", "classifier", "notebook"],
    "training": ["train", "model", "fit", "dataset", "classifier", "notebook"],
    "model": ["classifier", "weights", "predict", "train", "model", "pkl"],
    "predict": ["prediction", "inference", "score", "classifier"],
    "prediction": ["predict", "inference", "score", "classifier"],
    "frontend": ["ui", "streamlit", "app", "html", "css", "view", "interface"],
    "interface": ["ui", "streamlit", "app", "view", "frontend"],
    "config": ["settings", "env", "configuration", "requirements"],
}


def tokenize(text: str, code_aware: bool = True) -> list[str]:
    """Tokenize a text string, optionally expanding code identifiers."""
    raw_tokens = re.findall(r"[a-zA-Z0-9_\-\.]+", text)
    if not code_aware:
        return [t.lower() for t in raw_tokens]

    all_tokens: list[str] = []
    for raw in raw_tokens:
        sub_tokens = split_identifier(raw)
        all_tokens.extend(sub_tokens)

    return all_tokens


def list_available_indexes() -> list[str]:
    """Return all available repository index names."""
    names: list[str] = []
    if INDEXES_DIR.exists():
        for p in sorted(INDEXES_DIR.glob("*.json")):
            if not p.stem.endswith("_meta") and not p.stem.endswith("_embeddings"):
                names.append(p.stem)

    if INDEX_FILE.exists() and "default" not in names:
        names.append("default")

    return names


def load_index(
    repo_id: Optional[str] = None,
    index_path: Optional[Path | str] = None,
) -> list[dict[str, Any]]:
    """Load indexed repository records from disk.

    Supports:
    - Direct index file path
    - Specific repository ID (looks in data/indexes/<repo_id>.json)
    - Default global index (data/index.json)
    """
    target: Path

    if index_path:
        target = Path(index_path)
    elif repo_id and repo_id != "default":
        target = INDEXES_DIR / f"{repo_id}.json"
        if not target.exists():
            # Try sanitizing or matching without extension
            clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", repo_id)
            alt = INDEXES_DIR / f"{clean_id}.json"
            if alt.exists():
                target = alt
            else:
                available = list_available_indexes()
                raise FileNotFoundError(
                    f"Index for repository '{repo_id}' not found at {target}.\n"
                    f"Available indexes: {', '.join(available) or 'None'}"
                )
    else:
        target = INDEX_FILE

    if not target.exists():
        raise FileNotFoundError(
            f"Index not found: {target}\n"
            "Run 'python src/ingestion/indexer.py' first."
        )

    with target.open(encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return []


def document_text(record: dict[str, Any]) -> str:
    """Combine record fields into a searchable text passage."""
    return " ".join([
        record.get("name", ""),
        record.get("kind", ""),
        record.get("file", ""),
        record.get("docstring", ""),
        record.get("code", ""),
    ])


def get_record_tokens(record: dict[str, Any]) -> list[str]:
    """Retrieve pre-tokenized terms for record or compute and cache them."""
    tokens = record.get("_tokens")
    if tokens is None:
        tokens = tokenize(document_text(record), code_aware=True)
        record["_tokens"] = tokens
    return tokens


def search(
    query: str,
    records: Optional[list[dict[str, Any]]] = None,
    repo_id: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.5,
    hybrid: bool = False,
) -> list[tuple[float, dict[str, Any]]]:
    """Perform BM25 (or hybrid RRF) search over code passages.

    Args:
        query: User search query.
        records: Pre-loaded records. If None, loaded via repo_id or default index.
        repo_id: Repository ID to search if records is None.
        top_k: Number of top results to return.
        min_score: Minimum relevance score cutoff.
        hybrid: If True, combines BM25 with dense semantic vector search via RRF.

    Returns:
        List of (score, record) tuples sorted in descending order of relevance.
    """
    if hybrid:
        return hybrid_search(query, records=records, repo_id=repo_id, top_k=top_k)

    if records is None:
        records = load_index(repo_id=repo_id)

    if not records:
        return []

    # Tokenize query
    all_query_tokens = tokenize(query, code_aware=True)
    query_terms = [t for t in all_query_tokens if t not in STOP_WORDS]

    if not query_terms:
        query_terms = [t for t in all_query_tokens if len(t) > 1]

    if not query_terms:
        return []

    query_lower = query.lower()

    # Pre-tokenize documents (cached per record for instant multi-query searches)
    documents = [get_record_tokens(record) for record in records]

    document_count = len(documents)
    average_length = (
        sum(len(document) for document in documents)
        / max(document_count, 1)
    )

    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))

    k1 = 1.5
    b = 0.75
    # Synonyms mapping
    query_synonyms: dict[str, list[str]] = {}
    for qt in query_terms:
        if qt in SYNONYM_EXPANSIONS:
            query_synonyms[qt] = [s for s in SYNONYM_EXPANSIONS[qt] if s not in query_terms]

    results: list[tuple[float, dict[str, Any]]] = []
    for record, terms in zip(records, documents):
        frequencies = Counter(terms)
        length = len(terms)
        score = 0.0

        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue

            df = document_frequency[term]
            idf = math.log(
                1 + (document_count - df + 0.5) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * length / max(average_length, 1)
            )

            score += idf * frequency * (k1 + 1) / denominator

        # Synonyms matching
        for qt, syns in query_synonyms.items():
            for syn in syns:
                syn_freq = frequencies.get(syn, 0)
                if syn_freq > 0:
                    df = document_frequency.get(syn, 0)
                    if df > 0:
                        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                        denom = syn_freq + k1 * (1 - b + b * length / max(average_length, 1))
                        score += 0.45 * (idf * syn_freq * (k1 + 1) / denom)

        # --- Relevance Boosts ---
        name = record.get("name", "").lower()
        file_path = record.get("file", "").lower()
        docstring = record.get("docstring", "").lower()
        kind = record.get("kind", "")

        # 1. Exact or partial symbol-name match
        if name:
            if name in query_lower:
                score += 6.0
            elif any(qt == name for qt in query_terms):
                score += 5.0
            elif any(qt in name.split(".") for qt in query_terms):
                score += 3.0

        # 2. File path match (e.g. query mentions "auth" or "views.py")
        if file_path:
            file_name = Path(file_path).name.lower()
            file_stem = Path(file_path).stem.lower()
            if file_name in query_lower:
                score += 4.0
            elif file_stem in query_terms:
                score += 2.5

        # 3. Docstring relevance
        if docstring:
            doc_hits = sum(1 for qt in query_terms if qt in docstring)
            if doc_hits > 0:
                score += min(doc_hits * 0.75, 3.0)

        # 4. Documentation boost for general usage / overview queries
        doc_seeking_phrases = ["how to", "install", "getting started", "setup", "tutorial", "overview", "what is"]
        if kind == "documentation" and any(p in query_lower for p in doc_seeking_phrases):
            score += 3.5

        if score >= min_score:
            results.append((score, record))

    results.sort(key=lambda item: item[0], reverse=True)

    # Automatic fallback 1: If BM25 returns 0 results and not already hybrid, try hybrid semantic search
    if not results and not hybrid:
        try:
            sem_res = hybrid_search(query, records=records, repo_id=repo_id, top_k=top_k)
            if sem_res:
                return sem_res
        except Exception:
            pass

    # Automatic fallback 2: If still 0 results, retrieve top representative codebase files
    if not results and records:
        candidates = []
        for r in records:
            f_lower = r.get("file", "").lower()
            base_score = 0.5
            if any(ep in f_lower for ep in ("app.py", "main.py", "index.", "server.py", "pipeline", "train", "streamlit")):
                base_score += 2.5
            if "readme" in f_lower:
                base_score += 1.5
            if "requirements" in f_lower or "docker" in f_lower or "setup" in f_lower:
                base_score += 1.0
            candidates.append((base_score, r))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:top_k]

    return results[:top_k]


def hybrid_search(
    query: str,
    records: Optional[list[dict[str, Any]]] = None,
    repo_id: Optional[str] = None,
    top_k: int = 5,
    bm25_weight: float = 0.6,
    semantic_weight: float = 0.4,
) -> list[tuple[float, dict[str, Any]]]:
    """Combine lexical BM25 search with dense semantic search using Reciprocal Rank Fusion."""
    if records is None:
        records = load_index(repo_id=repo_id)

    if not records:
        return []

    from src.retrieval.semantic import SemanticSearchEngine

    # 1. Lexical BM25 Ranking
    bm25_results = search(query, records, top_k=len(records), min_score=0.1)

    # 2. Dense Semantic Latent Vector Ranking
    semantic_engine = SemanticSearchEngine(records)
    semantic_results = semantic_engine.search(query, top_k=len(records), min_similarity=0.0)

    # 3. Reciprocal Rank Fusion (RRF)
    record_to_idx = {id(rec): idx for idx, rec in enumerate(records)}
    rrf_scores: dict[int, float] = Counter()

    for rank, (_, rec) in enumerate(bm25_results, start=1):
        rec_idx = record_to_idx[id(rec)]
        rrf_scores[rec_idx] += bm25_weight / (60.0 + rank)

    for rank, (_, rec) in enumerate(semantic_results, start=1):
        rec_idx = record_to_idx[id(rec)]
        rrf_scores[rec_idx] += semantic_weight / (60.0 + rank)

    if not rrf_scores:
        return []

    sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
    return [(rrf_scores[i], records[i]) for i in sorted_indices[:top_k]]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RepoLens Codebase Search")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--repo",
        "-r",
        default=None,
        help="Repository ID to search (defaults to global index)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid BM25 + dense semantic vector search with RRF",
    )
    parser.add_argument(
        "--list-repos",
        action="store_true",
        help="List all available indexed repositories",
    )
    args = parser.parse_args()

    if args.list_repos:
        repos = list_available_indexes()
        print(f"Available indexes ({len(repos)}):")
        for r in repos:
            print(f"- {r}")
        return

    from src.ingestion.indexer import get_repo_metadata
    records = load_index(repo_id=args.repo)
    meta = get_repo_metadata(args.repo)
    repo_desc = args.repo or "global index"

    if not args.query:
        print(f"RepoLens search loaded {len(records)} passages from {repo_desc}.")
        print("Enter a query, or type 'exit' to quit.\n")

        while True:
            query = input("Search code: ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            results = search(query, records, top_k=args.top_k, hybrid=args.hybrid)
            display_results(results, meta=meta)
        return

    results = search(args.query, records, top_k=args.top_k, hybrid=args.hybrid)
    display_results(results, meta=meta)


def display_results(
    results: list[tuple[float, dict[str, Any]]],
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Format and display search results with clickable links."""
    if not results:
        print("\nNo matching symbols or passages found.\n")
        return

    web_url = meta.get("web_url") if meta else None
    branch = meta.get("branch", "main") if meta else "main"
    repo_path = meta.get("repo_path") if meta else None

    for rank, (score, record) in enumerate(results, start=1):
        file_path = record.get("file", "")
        start_line = record.get("start_line", 1)
        end_line = record.get("end_line", 1)
        location = f"{file_path}:{start_line}-{end_line}"

        # Construct link
        link = ""
        if web_url:
            clean_base = web_url.rstrip("/")
            link = f"{clean_base}/blob/{branch}/{file_path}#L{start_line}-L{end_line}"
        elif repo_path:
            abs_p = (Path(repo_path) / file_path).resolve()
            link = f"file://{abs_p}#L{start_line}"

        terminal_loc = f"\033]8;;{link}\033\\{location}\033]8;;\033\\" if link else location

        print(f"\n--- Result {rank} | Score: {score:.3f} ---")
        print(f"Symbol:   [{record.get('kind', 'code')}] {record.get('name', 'unnamed')}")
        print(f"Location: {terminal_loc}")
        if link:
            print(f"Link:     {link}")

        if record.get("docstring"):
            print(f"Docstring: {record['docstring']}")

        code = record.get("code", "")
        if len(code) > 800:
            code = code[:800] + "\n... [truncated]"
        print("\n" + code)

    print()


if __name__ == "__main__":
    main()
