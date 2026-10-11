"""Q&A and citation verification engine for RepoLens.

Constructs grounded prompts with retrieved repository passages, queries the LLM,
and validates citations against actual retrieved source lines.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.llm.provider import MODEL, generate_answer
from src.retrieval.search import load_index, search


from src.ingestion.indexer import get_repo_metadata


def format_citation_link(
    path: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    web_url: Optional[str] = None,
    branch: Optional[str] = "main",
) -> str:
    """Format citation with clickable GitHub markdown link if web_url is available."""
    active_branch = branch or "main"
    if start is not None and end is not None:
        text = f"[{path}:{start}-{end}]"
        if not web_url:
            return text
        clean_base = web_url.rstrip("/")
        url = f"{clean_base}/blob/{active_branch}/{path}#L{start}-L{end}"
        return f"[{path}:{start}-{end}]({url})"
    else:
        text = f"[{path}]"
        if not web_url:
            return text
        clean_base = web_url.rstrip("/")
        url = f"{clean_base}/blob/{active_branch}/{path}"
        return f"[{path}]({url})"


def validate_citations(
    answer: str,
    results: list[tuple[float, dict[str, Any]]],
    web_url: Optional[str] = None,
    branch: Optional[str] = "main",
    records: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Find citations in answer, verify against retrieved results, and format links."""
    active_branch = branch or "main"

    # Match bracketed citations [path:start-end] or list items - path:start-end
    bracketed = re.findall(
        r"\[([a-zA-Z0-9_\-/\.]+):(\d+)-(\d+)\]", answer
    )
    markdown_list = re.findall(
        r"[-*]\s+([a-zA-Z0-9_\-/\.]+):(\d+)-(\d+)", answer
    )
    citations = bracketed if bracketed else markdown_list

    allowed = [
        (
            record["file"],
            int(record["start_line"]),
            int(record["end_line"]),
        )
        for _, record in results
    ]

    verified = []
    unverified = []

    for path, start, end in citations:
        start_i, end_i = int(start), int(end)

        is_valid = any(
            source_path == path
            and source_start <= start_i
            and end_i <= source_end
            for source_path, source_start, source_end in allowed
        )

        citation = format_citation_link(
            path, start_i, end_i, web_url=web_url, branch=active_branch
        )
        raw_label = f"[{path}:{start_i}-{end_i}]"

        if is_valid:
            verified.append(citation)
        else:
            unverified.append(raw_label)

    # If web_url is available, make inline citations clickable GitHub links
    if web_url:
        known_files = {record["file"] for _, record in results}
        if records:
            known_files.update(r["file"] for r in records)

        def replace_bracketed_lines(match: re.Match) -> str:
            p = match.group(1)
            s = int(match.group(2))
            e = int(match.group(3))
            return format_citation_link(p, s, e, web_url=web_url, branch=active_branch)

        def replace_bracketed_file(match: re.Match) -> str:
            p = match.group(1)
            if p in known_files:
                return format_citation_link(p, web_url=web_url, branch=active_branch)
            return match.group(0)

        # Replace [path:start-end] if not already part of a link [path:start-end](...)
        answer = re.sub(
            r"\[([a-zA-Z0-9_\-/\.]+):(\d+)-(\d+)\](?!\()",
            replace_bracketed_lines,
            answer,
        )
        # Replace [path.ext] if in known files and not already part of a link
        answer = re.sub(
            r"\[([a-zA-Z0-9_\-/\.]+\.[a-zA-Z0-9]+)\](?!\()",
            replace_bracketed_file,
            answer,
        )

    if not citations:
        return answer + (
            "\n\nCitation check: No machine-readable citations were found. "
            "The Sources section has not been verified."
        )

    if verified:
        answer += "\n\nVerified citations: " + ", ".join(
            dict.fromkeys(verified)
        )

    if unverified:
        answer += "\n\nUnverified citations: " + ", ".join(
            dict.fromkeys(unverified)
        )

    return answer


def build_context(
    results: list[tuple[float, dict[str, Any]]],
    records: Optional[list[dict[str, Any]]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> str:
    """Format retrieved records into a structured context block for the LLM."""
    sections = []

    # 1. Repository Structure & Manifest
    manifest_parts = []
    if meta and meta.get("repo_id"):
        manifest_parts.append(f"Repository: {meta['repo_id']}")
    if meta and meta.get("web_url"):
        manifest_parts.append(f"GitHub URL: {meta['web_url']} (branch: {meta.get('branch', 'main')})")

    unique_files = sorted({r["file"] for r in (records or [rec for _, rec in results])})
    if unique_files:
        manifest_parts.append(f"Repository Files: {', '.join(unique_files)}")

    if manifest_parts:
        sections.append("=== Repository Structure & Files ===\n" + "\n".join(manifest_parts))

    # 2. Retrieved Passages
    passages = []
    for score, record in results:
        passages.append(
            f"""Source: {record['file']}:{record['start_line']}-{record['end_line']}
Symbol: {record.get('kind', 'code')} {record.get('name', 'unnamed')}
Relevance score: {score:.3f}
Docstring: {record.get('docstring', '')}
Code:
{record.get('code', '')}"""
        )

    if passages:
        sections.append("=== Retrieved Code & Document Evidence ===\n" + "\n---\n".join(passages))

    return "\n\n".join(sections)


def ask_llm(
    question: str,
    context: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Format prompt with strict groundedness, code-first synthesis, and citation rules."""
    prompt = f"""You are RepoLens, an expert codebase intelligence assistant.

Answer the user's question accurately using the repository evidence and repository structure provided below.

STRICT INSTRUCTIONS:
1. Direct Answer: Answer the user's question directly in the opening paragraph.
2. Code-First Synthesis: Prioritize actual source code implementation (functions, logic, entry points, scripts, pipelines) rather than merely quoting the README. Explain how the code works.
3. Explicit Handling of Missing Components: If the user asks about a component that does NOT exist in the repository (such as a CI/CD pipeline, Dockerfile, or database), explicitly confirm that it does not exist, and explain what the repository uses instead (e.g. local execution via app.py, Streamlit).
4. Accurate Citations: Cite every referenced code file and line range using [path:start_line-end_line] or [path] (e.g. [app.py:1-74], [README.md:50-58]).
5. Answer Design: Structure your response cleanly with Markdown:
   - Use bold section titles (e.g. **Direct Answer**, **Implementation & Architecture**, **Execution & Deployment**)
   - Use bullet points and concise code snippets where helpful
6. Sources: End with a "### 📚 Sources & Location" section listing each referenced file and its purpose.

Repository evidence & files:
{context}

User question:
{question}
"""
    return generate_answer(prompt, model=model, provider=provider)


# Backward-compatibility alias for tests and existing callers
ask_ollama = ask_llm


def answer_question(
    question: str,
    repo_id: Optional[str] = None,
    records: Optional[list[dict[str, Any]]] = None,
    top_k: int = 4,
    model: Optional[str] = None,
    web_url: Optional[str] = None,
    hybrid: bool = False,
    provider: Optional[str] = None,
) -> tuple[str, list[tuple[float, dict[str, Any]]]]:
    """Complete pipeline: search index -> build context -> query LLM -> validate citations.

    Returns:
        tuple of (verified_answer, retrieved_results)
    """
    if records is None:
        records = load_index(repo_id=repo_id)

    meta = get_repo_metadata(repo_id)
    active_url = web_url or meta.get("web_url")
    active_branch = meta.get("branch", "main")

    results = search(question, records, top_k=top_k, hybrid=hybrid)

    if not results and records:
        results = search(question, records, top_k=top_k, min_score=0.0)

    if not results:
        return (
            "No matching code or documentation was found in the repository for your question.",
            [],
        )

    context = build_context(results, records=records, meta=meta)
    raw_answer = ask_ollama(question, context, model=model, provider=provider)
    final_answer = validate_citations(
        raw_answer, results, web_url=active_url, branch=active_branch, records=records
    )

    return final_answer, results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RepoLens Interactive Q&A")
    parser.add_argument(
        "--repo",
        "-r",
        default=None,
        help="Repository ID to query (default: global index)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Ollama model to use",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=4,
        help="Number of retrieved passages (default: 4)",
    )
    args = parser.parse_args()

    records = load_index(repo_id=args.repo)
    repo_title = args.repo or "global index"
    active_model = args.model or MODEL

    print(f"RepoLens RAG ready: {len(records)} indexed passages from [{repo_title}]")
    print(f"LLM: {active_model}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask RepoLens: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            answer, results = answer_question(
                question,
                records=records,
                top_k=args.top_k,
                model=active_model,
            )

            print("\n--- Retrieved evidence ---")
            for score, record in results:
                print(
                    f"- {record['file']}:{record['start_line']}-"
                    f"{record['end_line']} ({record.get('name', 'unnamed')})"
                )

            print("\n--- RepoLens answer ---")
            print(answer)
        except Exception as error:
            print(f"Error: {error}")

        print()


if __name__ == "__main__":
    main()
