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
    start: int,
    end: int,
    web_url: Optional[str] = None,
    branch: Optional[str] = "main",
) -> str:
    """Format citation with clickable GitHub markdown link if web_url is available."""
    text = f"[{path}:{start}-{end}]"
    if not web_url:
        return text
    clean_base = web_url.rstrip("/")
    active_branch = branch or "main"
    url = f"{clean_base}/blob/{active_branch}/{path}#L{start}-L{end}"
    return f"[{path}:{start}-{end}]({url})"


def validate_citations(
    answer: str,
    results: list[tuple[float, dict[str, Any]]],
    web_url: Optional[str] = None,
    branch: Optional[str] = "main",
) -> str:
    """Check whether citations in the answer refer to retrieved source ranges."""
    bracketed = re.findall(r"\[([^:\[\]\n]+):(\d+)-(\d+)\]", answer)
    unbracketed = re.findall(
        r"(?:^|\n)\s*[-*]?\s*`?([a-zA-Z0-9_\-/\.]+):(\d+)-(\d+)`?",
        answer,
    )
    # Combine unique citations preserving order
    citations: list[tuple[str, str, str]] = []
    for c in bracketed + unbracketed:
        if c not in citations:
            citations.append(c)

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
        start, end = int(start), int(end)

        is_valid = any(
            source_path == path
            and source_start <= start
            and end <= source_end
            for source_path, source_start, source_end in allowed
        )

        citation = format_citation_link(
            path, start, end, web_url=web_url, branch=branch
        )
        raw_label = f"[{path}:{start}-{end}]"

        if is_valid:
            verified.append(citation)
        else:
            unverified.append(raw_label)

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


def build_context(results: list[tuple[float, dict[str, Any]]]) -> str:
    """Format retrieved records into a structured context block for the LLM."""
    passages = []

    for score, record in results:
        passages.append(
            f"""Source: {record['file']}:{record['start_line']}-{record['end_line']}
Symbol: {record.get('kind', 'code')} {record.get('name', 'unnamed')}
Relevance score: {score:.3f}
Docstring: {record.get('docstring', '')}
Code:
{record.get('code', '')}
"""
        )

    return "\n---\n".join(passages)


def ask_llm(
    question: str,
    context: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Format prompt with strict groundedness rules and query LLM."""
    prompt = f"""You are RepoLens, a codebase intelligence assistant.

Answer the user's question using ONLY the repository evidence provided below.

STRICT RULES:
1. Explain only behavior supported by the supplied source code.
2. Never invent functions, files, line numbers, examples, or test results.
3. Do not generate code examples unless explicitly requested.
4. Cite important claims using [path:start_line-end_line].
5. Use only paths and line ranges shown in the evidence.
6. If evidence is insufficient, say what cannot be verified.
7. Prefer concise, structured answers.
8. Distinguish explicit behavior from inference.
9. Do not attribute behavior from another function without evidence.
10. For exceptions, list only conditions supported by the supplied code.
11. End with a Sources section containing exact file paths and line ranges.

Repository evidence:
{context}

User question:
{question}

Write a concise answer with a Sources section at the end.
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

    if not results:
        return (
            "No matching code or documentation was found in the repository for your question.",
            [],
        )

    context = build_context(results)
    raw_answer = ask_ollama(question, context, model=model, provider=provider)
    final_answer = validate_citations(
        raw_answer, results, web_url=active_url, branch=active_branch
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
