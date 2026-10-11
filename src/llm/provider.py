"""LLM provider integration for RepoLens using local Ollama.

Supports custom models, URLs, and environment variable overrides.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")

OLLAMA_URL = DEFAULT_OLLAMA_URL
MODEL = DEFAULT_MODEL


def check_ollama_available(url: Optional[str] = None) -> tuple[bool, str]:
    """Check whether the Ollama server is reachable."""
    target_url = url or OLLAMA_URL
    # Base URL for checking tags
    base_url = target_url.replace("/api/chat", "").rstrip("/") + "/api/tags"

    try:
        req = urllib.request.Request(base_url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return True, f"Ollama is online. Models: {', '.join(models) or 'none'}"
    except Exception as err:
        return False, f"Could not reach Ollama at {target_url}: {err}"


def generate_answer(
    prompt: str,
    model: Optional[str] = None,
    url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    """Generate a response using the configured local Ollama model."""
    active_model = model or MODEL
    active_url = url or OLLAMA_URL

    sys_content = system_prompt or (
        "You are RepoLens, a codebase intelligence assistant. "
        "Explain source code using supplied evidence and citations."
    )

    payload = {
        "model": active_model,
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    request = urllib.request.Request(
        active_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["message"]["content"]

    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not reach Ollama at {active_url}. Check: systemctl status ollama or start ollama serve"
        ) from error
