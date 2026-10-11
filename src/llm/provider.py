"""Multi-provider LLM integration for RepoLens.

Supports switching seamlessly between local LLMs (Ollama, LM Studio, vLLM,
LocalAI) and cloud LLMs (OpenAI, Anthropic Claude, Google Gemini, DeepSeek,
Groq, OpenRouter).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
CONFIG_FILE = DATA_DIR / "config.json"


@dataclass
class ProviderSpec:
    name: str
    display_name: str
    default_model: str
    default_url: str
    env_keys: list[str] = field(default_factory=list)
    is_cloud: bool = False
    format_type: str = "openai"  # "ollama", "openai", "anthropic", "gemini"
    description: str = ""


SUPPORTED_PROVIDERS: dict[str, ProviderSpec] = {
    "ollama": ProviderSpec(
        name="ollama",
        display_name="Ollama (Local)",
        default_model="qwen2.5-coder:3b",
        default_url="http://localhost:11434/api/chat",
        env_keys=["OLLAMA_MODEL", "OLLAMA_URL"],
        is_cloud=False,
        format_type="ollama",
        description="Local private inference via Ollama daemon",
    ),
    "openai": ProviderSpec(
        name="openai",
        display_name="OpenAI (Cloud)",
        default_model="gpt-4o-mini",
        default_url="https://api.openai.com/v1/chat/completions",
        env_keys=["OPENAI_API_KEY"],
        is_cloud=True,
        format_type="openai",
        description="OpenAI API (GPT-4o, GPT-4o-mini, o1, o3-mini)",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        display_name="Anthropic (Cloud)",
        default_model="claude-3-5-sonnet-latest",
        default_url="https://api.anthropic.com/v1/messages",
        env_keys=["ANTHROPIC_API_KEY"],
        is_cloud=True,
        format_type="anthropic",
        description="Anthropic Claude API (Sonnet 3.5, Haiku 3.5)",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        display_name="Google Gemini (Cloud)",
        default_model="gemini-2.0-flash",
        default_url="https://generativelanguage.googleapis.com/v1beta",
        env_keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        is_cloud=True,
        format_type="gemini",
        description="Google Gemini API (2.0 Flash, 1.5 Pro, 2.5 Flash)",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        display_name="DeepSeek (Cloud)",
        default_model="deepseek-chat",
        default_url="https://api.deepseek.com/chat/completions",
        env_keys=["DEEPSEEK_API_KEY"],
        is_cloud=True,
        format_type="openai",
        description="DeepSeek AI API (DeepSeek-V3, DeepSeek-R1)",
    ),
    "groq": ProviderSpec(
        name="groq",
        display_name="Groq (Cloud)",
        default_model="llama-3.3-70b-versatile",
        default_url="https://api.groq.com/openai/v1/chat/completions",
        env_keys=["GROQ_API_KEY"],
        is_cloud=True,
        format_type="openai",
        description="Ultra-fast cloud inference on Groq LPUs",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        display_name="OpenRouter (Cloud)",
        default_model="openai/gpt-4o-mini",
        default_url="https://openrouter.ai/api/v1/chat/completions",
        env_keys=["OPENROUTER_API_KEY"],
        is_cloud=True,
        format_type="openai",
        description="Unified router accessing 100+ AI models",
    ),
    "custom": ProviderSpec(
        name="custom",
        display_name="Custom / Local OpenAI (LM Studio / vLLM / Jan)",
        default_model="local-model",
        default_url="http://localhost:1234/v1/chat/completions",
        env_keys=["CUSTOM_LLM_API_KEY"],
        is_cloud=False,
        format_type="openai",
        description="Any local OpenAI-compatible endpoint (LM Studio, vLLM, etc.)",
    ),
}

# Backward compatibility defaults
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_URL = DEFAULT_OLLAMA_URL
MODEL = DEFAULT_MODEL


def load_config() -> dict[str, Any]:
    """Load RepoLens configuration from disk."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(config: dict[str, Any]) -> None:
    """Save RepoLens configuration to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def get_active_provider() -> str:
    """Get active provider name.

    Priority:
    1. REPOLENS_PROVIDER / LLM_PROVIDER env var
    2. config.json 'provider' setting
    3. Default 'ollama'
    """
    for env_var in ("REPOLENS_PROVIDER", "LLM_PROVIDER"):
        val = os.environ.get(env_var)
        if val and val.lower() in SUPPORTED_PROVIDERS:
            return val.lower()

    cfg = load_config()
    provider = cfg.get("provider", "ollama")
    if provider in SUPPORTED_PROVIDERS:
        return provider
    return "ollama"


def get_active_model(provider_name: Optional[str] = None) -> str:
    """Get active model for given provider.

    Priority:
    1. REPOLENS_MODEL / LLM_MODEL env var
    2. Provider-specific env var (e.g. OLLAMA_MODEL)
    3. config.json model setting for provider
    4. Provider default model
    """
    p_name = (provider_name or get_active_provider()).lower()
    spec = SUPPORTED_PROVIDERS.get(p_name, SUPPORTED_PROVIDERS["ollama"])

    for env_var in ("REPOLENS_MODEL", "LLM_MODEL"):
        val = os.environ.get(env_var)
        if val:
            return val

    if p_name == "ollama" and "OLLAMA_MODEL" in os.environ:
        return os.environ["OLLAMA_MODEL"]

    cfg = load_config()
    prov_cfg = cfg.get("providers", {}).get(p_name, {})
    if prov_cfg.get("model"):
        return prov_cfg["model"]

    if cfg.get("provider") == p_name and cfg.get("model"):
        return cfg["model"]

    return spec.default_model


def get_api_key(provider_name: str) -> Optional[str]:
    """Retrieve API key from environment variables or saved config."""
    p_name = provider_name.lower()
    spec = SUPPORTED_PROVIDERS.get(p_name)
    if not spec:
        return None

    # Check environment variables
    for key in spec.env_keys:
        val = os.environ.get(key)
        if val:
            return val

    # Check config file
    cfg = load_config()
    return cfg.get("providers", {}).get(p_name, {}).get("api_key") or None


def get_provider_url(provider_name: str) -> str:
    """Retrieve endpoint URL for provider."""
    p_name = provider_name.lower()
    spec = SUPPORTED_PROVIDERS.get(p_name, SUPPORTED_PROVIDERS["ollama"])

    if p_name == "ollama" and "OLLAMA_URL" in os.environ:
        return os.environ["OLLAMA_URL"]

    cfg = load_config()
    return cfg.get("providers", {}).get(p_name, {}).get("url") or spec.default_url


def set_active_provider(
    provider_name: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    """Set active provider and optionally update its credentials/model."""
    p_name = provider_name.lower()
    if p_name not in SUPPORTED_PROVIDERS:
        valid = ", ".join(SUPPORTED_PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Supported providers: {valid}")

    cfg = load_config()
    cfg["provider"] = p_name

    providers_cfg = cfg.setdefault("providers", {})
    p_cfg = providers_cfg.setdefault(p_name, {})

    if model:
        p_cfg["model"] = model
        cfg["model"] = model
    if api_key:
        p_cfg["api_key"] = api_key
    if url:
        p_cfg["url"] = url

    save_config(cfg)


def set_active_model(model: str, provider_name: Optional[str] = None) -> None:
    """Set active model for current or specified provider."""
    p_name = (provider_name or get_active_provider()).lower()
    cfg = load_config()
    providers_cfg = cfg.setdefault("providers", {})
    p_cfg = providers_cfg.setdefault(p_name, {})
    p_cfg["model"] = model

    if cfg.get("provider", "ollama") == p_name:
        cfg["model"] = model

    save_config(cfg)


def check_ollama_available(url: Optional[str] = None) -> tuple[bool, str]:
    """Check whether the Ollama server is reachable."""
    target_url = url or get_provider_url("ollama")
    base_url = target_url.replace("/api/chat", "").rstrip("/") + "/api/tags"

    try:
        req = urllib.request.Request(base_url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return True, f"Ollama is online. Models: {', '.join(models) or 'none'}"
    except Exception as err:
        return False, f"Could not reach Ollama at {target_url}: {err}"


def check_provider_status(provider_name: Optional[str] = None) -> tuple[bool, str]:
    """Check health and configuration status of given provider."""
    p_name = (provider_name or get_active_provider()).lower()
    spec = SUPPORTED_PROVIDERS.get(p_name)
    if not spec:
        return False, f"Unknown provider '{p_name}'"

    if p_name == "ollama":
        return check_ollama_available()

    if spec.is_cloud:
        key = get_api_key(p_name)
        active_model = get_active_model(p_name)
        if key:
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            return True, f"Ready ({spec.display_name} | model: {active_model} | key: {masked})"
        else:
            env_hint = " or ".join(spec.env_keys)
            return False, f"Missing API key. Set {env_hint} or configure via 'repolens provider set {p_name} --api-key <KEY>'"

    # Custom local endpoint
    url = get_provider_url("custom")
    active_model = get_active_model("custom")
    return True, f"Configured ({spec.display_name} | URL: {url} | model: {active_model})"


def list_providers() -> list[dict[str, Any]]:
    """Return summary information for all supported providers."""
    active_p = get_active_provider()
    result = []
    for name, spec in SUPPORTED_PROVIDERS.items():
        is_active = (name == active_p)
        model = get_active_model(name)
        has_key = bool(get_api_key(name)) if spec.is_cloud else True
        result.append({
            "name": name,
            "display_name": spec.display_name,
            "is_active": is_active,
            "is_cloud": spec.is_cloud,
            "model": model,
            "has_key": has_key,
            "description": spec.description,
            "env_keys": spec.env_keys,
            "url": get_provider_url(name),
        })
    return result


# ---------------------------------------------------------------------------
# Provider Call Implementations
# ---------------------------------------------------------------------------

DEFAULT_USER_AGENT = "RepoLens/1.0 (https://github.com/wasimmos/repo-lens)"


def _call_ollama(
    prompt: str,
    model: str,
    url: str,
    system_prompt: str,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": "15m",
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    return res["message"]["content"]


def _call_openai_compatible(
    prompt: str,
    model: str,
    url: str,
    system_prompt: str,
    temperature: float,
    timeout: int,
    api_key: Optional[str] = None,
    extra_headers: Optional[dict[str, str]] = None,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    choices = res.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenAI-compatible server returned no choices: {res}")
    return choices[0]["message"]["content"]


def _call_anthropic(
    prompt: str,
    model: str,
    url: str,
    system_prompt: str,
    temperature: float,
    timeout: int,
    api_key: str,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    content_list = res.get("content", [])
    text_pieces = [c.get("text", "") for c in content_list if c.get("type") == "text"]
    return "".join(text_pieces)


def _call_gemini(
    prompt: str,
    model: str,
    base_url: str,
    system_prompt: str,
    temperature: float,
    timeout: int,
    api_key: str,
) -> str:
    clean_base = base_url.rstrip("/")
    clean_model = model.removeprefix("models/").strip()
    url = f"{clean_base}/models/{clean_model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 1024,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    candidates = res.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {res}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def generate_answer(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    """Generate a response using the active or requested LLM provider."""
    p_name = (provider or get_active_provider()).lower()
    if p_name not in SUPPORTED_PROVIDERS:
        valid = ", ".join(SUPPORTED_PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{p_name}'. Supported providers: {valid}")

    spec = SUPPORTED_PROVIDERS[p_name]
    active_model = model or get_active_model(p_name)
    active_url = url or get_provider_url(p_name)
    active_key = api_key or get_api_key(p_name)

    # Validate cloud credentials
    if spec.is_cloud and not active_key:
        env_hint = " or ".join(spec.env_keys)
        raise RuntimeError(
            f"API key missing for provider '{p_name}'.\n"
            f"Please set environment variable {env_hint}, or run:\n"
            f"  repolens provider set {p_name} --api-key <YOUR_KEY>"
        )

    sys_content = system_prompt or (
        "You are RepoLens, a codebase intelligence assistant. "
        "Explain source code using supplied evidence and citations."
    )

    try:
        if spec.format_type == "ollama":
            return _call_ollama(
                prompt=prompt,
                model=active_model,
                url=active_url,
                system_prompt=sys_content,
                temperature=temperature,
                timeout=timeout,
            )
        elif spec.format_type == "anthropic":
            return _call_anthropic(
                prompt=prompt,
                model=active_model,
                url=active_url,
                system_prompt=sys_content,
                temperature=temperature,
                timeout=timeout,
                api_key=active_key or "",
            )
        elif spec.format_type == "gemini":
            return _call_gemini(
                prompt=prompt,
                model=active_model,
                base_url=active_url,
                system_prompt=sys_content,
                temperature=temperature,
                timeout=timeout,
                api_key=active_key or "",
            )
        else:
            # OpenAI and OpenAI-compatible
            extra_headers = None
            if p_name == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/wasimmos/repo-lens",
                    "X-Title": "RepoLens",
                }
            return _call_openai_compatible(
                prompt=prompt,
                model=active_model,
                url=active_url,
                system_prompt=sys_content,
                temperature=temperature,
                timeout=timeout,
                api_key=active_key,
                extra_headers=extra_headers,
            )

    except urllib.error.HTTPError as err:
        err_msg = ""
        try:
            body = err.read().decode("utf-8")
            data = json.loads(body)
            if "error" in data:
                if isinstance(data["error"], dict) and "message" in data["error"]:
                    err_msg = data["error"]["message"]
                else:
                    err_msg = str(data["error"])
            elif "message" in data:
                err_msg = str(data["message"])
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {err.code} from {spec.display_name}: {err_msg or err.reason}"
        ) from err
    except urllib.error.URLError as err:
        if p_name == "ollama":
            raise RuntimeError(
                f"Could not reach Ollama at {active_url}. "
                f"Check: systemctl status ollama or start ollama serve"
            ) from err
        raise RuntimeError(
            f"Could not reach {spec.display_name} at {active_url}: {err.reason}"
        ) from err
