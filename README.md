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
### 1. Prerequisites
- Python 3.10+ with `numpy`
- [Ollama](https://ollama.com/) running locally:
  ```bash
  ollama serve
  ollama pull qwen2.5-coder:3b
  ```

### 2. Ingest a Repository
Clone and index any GitHub repository:
```bash
# Using GitHub shorthand
repolens ingest pallets/flask

# Or full URL
repolens ingest https://github.com/psf/requests.git

# Or a local folder
repolens ingest data/sample_repo
```

### 3. Active Repository Memory
RepoLens automatically remembers the active repository, so you don't need to pass `--repo` every time:
```bash
# Switch active repo (supports partial name matching)
repolens use flask

# List cached repositories and see which one is active
repolens list    # alias: repolens ls
```

### 4. Direct Q&A (No Quotes or Subcommands Needed)
You can ask questions directly from the command line:
```bash
# Direct question
repolens "How does route registration work?"

# Or using ask without quotes
repolens ask How does route registration work

# Target a specific repository or enable hybrid search
repolens ask "How are cookies signed?" --hybrid
```

### 5. Hybrid Semantic Search
Search code and documentation passages without quotes:
```bash
repolens search credentials and session tokens --hybrid
```

### 6. Interactive Chat with Slash Commands
Start an interactive terminal Q&A session:
```bash
repolens chat    # alias: repolens talk
```
Inside chat, you can switch repositories, toggle hybrid search, or view loaded indexes on the fly using `/use <repo>`, `/hybrid`, `/list`, or `/exit`.

### 7. System Status
```bash
repolens status
```

---

## CLI Reference

| Command | Aliases | Arguments | Description |
|---|---|---|---|
| `repolens <question>` | | | Ask directly without typing `ask` |
| `repolens ingest <source>` | `clone`, `add` | `[--force] [--python-only]` | Clone/load, index, and activate repository |
| `repolens use <repo>` | `select`, `switch` | `<repo_name>` | Set active repository (supports fuzzy match) |
| `repolens ask <question>` | | `[--repo <id>] [--model <m>] [--top-k <k>] [--hybrid]` | Single-shot Q&A with clickable citations (quotes optional) |
| `repolens chat` | `talk`, `interactive` | `[--repo <id>] [--model <m>] [--top-k <k>] [--hybrid]` | Interactive terminal chat |
| `repolens search <query>` | | `[--repo <id>] [--top-k <k>] [--hybrid]` | Lexical or Hybrid BM25+LSI search (quotes optional) |
| `repolens list` | `ls`, `repos` | | List cached repositories with active indicator |
| `repolens remove <repo>` | `rm`, `delete` | `<repo_id>` | Remove cached repository and index files |
| `repolens status` | | | Check Ollama and system status |

---

## Configuration

You can configure the LLM provider using environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5-coder:3b` | Target Ollama model |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama API chat endpoint |

---

## Running Tests

Run the complete test suite:
```bash
pytest
```
Includes 42 automated tests covering repository loading, multi-format indexing, code tokenization, dense semantic vectors, hybrid RRF retrieval, citation verification with GitHub links, and CLI UX commands.
