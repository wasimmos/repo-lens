# RepoLens 🔍

**RepoLens** is an intelligent repository parsing and Retrieval-Augmented Generation (RAG) assistant for GitHub and local codebases. It ingests repositories, extracts structured symbols (classes, functions, methods) and documentation with exact line tracking, performs code-aware BM25 search, and generates grounded, cited answers using local LLMs (via Ollama).

---

## Features

- **GitHub & Local Ingestion**: Clone any GitHub repository with shallow clones (`git clone --depth 1`) or load local directories directly.
- **Multi-Format Indexing**:
  - **Python AST**: Extracts classes, methods (with qualified names like `Flask.route`), functions, and module docstrings.
  - **Documentation**: Parses Markdown (`.md`), reStructuredText (`.rst`), and text into section chunks based on headers.
  - **Configurations**: Indexes `pyproject.toml`, `requirements.txt`, `package.json`, `Dockerfile`, `.yaml`, etc.
  - **General Code**: Sliding-window chunker with line number preservation for JavaScript, TypeScript, Go, Rust, HTML, CSS, SQL, Shell, etc.
- **Code-Aware Retrieval**:
  - Sub-identifier splitting for `snake_case`, `camelCase`, and digits (e.g., `handle_request` $\rightarrow$ `['handle_request', 'handle', 'request']`).
  - Relevance boosting for exact symbol names, file paths, and documentation queries.
  - Per-repository index isolation (`data/indexes/<repo_id>.json`).
- **Grounded Answering & Citation Checking**:
  - Enforces strict factual grounding on retrieved evidence.
  - Automatically verifies machine-readable citations (`[file:start-end]` and markdown lists) against retrieved source boundaries.
- **Local & Private**: Powered by local Ollama (`qwen2.5-coder:3b` by default) with zero telemetry.

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) running locally:
  ```bash
  ollama serve
  ollama pull qwen2.5-coder:3b
  ```

### 2. Ingest a Repository
Clone and index any GitHub repository:
```bash
# Using GitHub shorthand
python3 -m src ingest pallets/flask

# Or full URL
python3 -m src ingest https://github.com/psf/requests.git

# Or a local folder
python3 -m src ingest data/sample_repo
```

### 3. Ask Questions
Ask a question about the repository and receive answers with verified source citations:
```bash
python3 -m src ask "How does route registration work?" --repo pallets_flask
```

### 4. Interactive Chat
Start an interactive Q&A session in your terminal:
```bash
python3 -m src chat --repo pallets_flask
```

### 5. Search Code & Docs
Run fast lexical code search across the indexed repository:
```bash
python3 -m src search "url_for" --repo pallets_flask --top-k 5
```

### 6. List Repositories & System Status
```bash
# List all cached repositories and indexes
python3 -m src list

# Check system status and Ollama availability
python3 -m src status
```

---

## CLI Reference

| Command | Arguments | Description |
|---|---|---|
| `python3 -m src ingest <source>` | `[--force] [--python-only]` | Clone/load and index a repository |
| `python3 -m src ask "<question>"` | `[--repo <id>] [--model <m>] [--top-k <k>]` | Single-shot Q&A with citations |
| `python3 -m src chat` | `[--repo <id>] [--model <m>] [--top-k <k>]` | Interactive continuous terminal chat |
| `python3 -m src search "<query>"` | `[--repo <id>] [--top-k <k>]` | BM25 search over code and docs |
| `python3 -m src list` | | List cached repositories and indexes |
| `python3 -m src status` | | Check Ollama and system status |

---

## Configuration

You can configure the LLM provider using environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5-coder:3b` | Target Ollama model |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama API chat endpoint |

Example:
```bash
OLLAMA_MODEL="qwen2.5-coder:7b" python3 -m src ask "Explain the middleware architecture"
```

---

## Running Tests

Run the complete test suite:
```bash
pytest
```
All 31 unit tests cover repository loading, multi-format indexing, code tokenization, BM25 retrieval, citation verification, and CLI parsers.
