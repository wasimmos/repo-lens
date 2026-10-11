"""Tests for multi-provider LLM integration."""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from src.llm.provider import (
    SUPPORTED_PROVIDERS,
    check_provider_status,
    generate_answer,
    get_active_model,
    get_active_provider,
    get_api_key,
    get_provider_url,
    list_providers,
    set_active_model,
    set_active_provider,
)


def _mock_http_response(json_data: dict, status: int = 200):
    content = json.dumps(json_data).encode("utf-8")
    mock = MagicMock()
    mock.read.return_value = content
    mock.status = status
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = None
    return mock


def test_supported_providers_registered():
    expected = {"ollama", "openai", "anthropic", "gemini", "deepseek", "groq", "openrouter", "custom"}
    assert expected.issubset(SUPPORTED_PROVIDERS.keys())


def test_provider_and_model_switching(tmp_path: Path):
    fake_config = tmp_path / "config.json"
    with patch("src.llm.provider.CONFIG_FILE", fake_config), \
         patch("src.llm.provider.DATA_DIR", tmp_path):

        # Initial default is ollama
        assert get_active_provider() == "ollama"
        assert get_active_model("ollama") == "qwen2.5-coder:3b"

        # Switch to openai
        set_active_provider("openai", model="gpt-4o", api_key="sk-test1234")
        assert get_active_provider() == "openai"
        assert get_active_model("openai") == "gpt-4o"
        assert get_api_key("openai") == "sk-test1234"

        # Switch model for active provider
        set_active_model("gpt-4o-mini")
        assert get_active_model("openai") == "gpt-4o-mini"

        # Switch to custom endpoint
        set_active_provider("custom", url="http://localhost:8000/v1/chat/completions", model="my-model")
        assert get_active_provider() == "custom"
        assert get_provider_url("custom") == "http://localhost:8000/v1/chat/completions"
        assert get_active_model("custom") == "my-model"


def test_generate_answer_ollama():
    mock_resp = _mock_http_response({
        "message": {"role": "assistant", "content": "Ollama generated answer."}
    })
    with patch("urllib.request.urlopen", return_value=mock_resp):
        ans = generate_answer("Hello", provider="ollama", model="qwen2.5-coder:3b")
        assert ans == "Ollama generated answer."


def test_generate_answer_openai_compatible():
    mock_resp = _mock_http_response({
        "choices": [{"message": {"role": "assistant", "content": "OpenAI generated answer."}}]
    })
    with patch("urllib.request.urlopen", return_value=mock_resp):
        ans = generate_answer("Hello", provider="openai", api_key="sk-fake", model="gpt-4o-mini")
        assert ans == "OpenAI generated answer."


def test_generate_answer_anthropic():
    mock_resp = _mock_http_response({
        "content": [{"type": "text", "text": "Anthropic Claude answer."}]
    })
    with patch("urllib.request.urlopen", return_value=mock_resp):
        ans = generate_answer("Hello", provider="anthropic", api_key="ant-fake", model="claude-3-5-sonnet-latest")
        assert ans == "Anthropic Claude answer."


def test_generate_answer_gemini():
    mock_resp = _mock_http_response({
        "candidates": [{
            "content": {"parts": [{"text": "Gemini generated answer."}], "role": "model"}
        }]
    })
    with patch("urllib.request.urlopen", return_value=mock_resp):
        ans = generate_answer("Hello", provider="gemini", api_key="gem-fake", model="gemini-1.5-flash")
        assert ans == "Gemini generated answer."


def test_generate_answer_missing_key_raises_error():
    with patch.dict("os.environ", {}, clear=True), \
         patch("src.llm.provider.get_api_key", return_value=None):
        with pytest.raises(RuntimeError, match="API key missing"):
            generate_answer("Hello", provider="openai", api_key=None)


def test_generate_answer_http_error_handling():
    err_body = json.dumps({"error": {"message": "Invalid API key provided"}}).encode("utf-8")
    http_error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=BytesIO(err_body),
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(RuntimeError, match="HTTP 401 from OpenAI"):
            generate_answer("Hello", provider="openai", api_key="sk-invalid")


def test_check_provider_status():
    # Ollama online
    with patch("src.llm.provider.check_ollama_available", return_value=(True, "Online")):
        is_ok, msg = check_provider_status("ollama")
        assert is_ok is True

    # Cloud provider without key
    with patch("src.llm.provider.get_api_key", return_value=None):
        is_ok, msg = check_provider_status("anthropic")
        assert is_ok is False
        assert "Missing API key" in msg

    # Cloud provider with key
    with patch("src.llm.provider.get_api_key", return_value="sk-12345678"):
        is_ok, msg = check_provider_status("anthropic")
        assert is_ok is True
        assert "Ready" in msg


def test_list_providers():
    provs = list_providers()
    assert len(provs) >= 8
    names = [p["name"] for p in provs]
    assert "ollama" in names
    assert "openai" in names
    assert "anthropic" in names
    assert "gemini" in names


def test_user_agent_header_and_gemini_prefix(monkeypatch):
    captured_reqs = []

    def fake_urlopen(req, timeout=180):
        captured_reqs.append(req)
        return _mock_http_response({
            "candidates": [{
                "content": {"parts": [{"text": "Gemini 2.0 response"}], "role": "model"}
            }]
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    # Calling with 'models/gemini-2.0-flash' prefix
    ans = generate_answer("Hello", provider="gemini", api_key="gem-key", model="models/gemini-2.0-flash")
    assert ans == "Gemini 2.0 response"
    assert len(captured_reqs) == 1

    req = captured_reqs[0]
    # Check User-Agent header
    assert "RepoLens" in req.get_header("User-agent")
    # Check that 'models/models/' was avoided
    assert "/models/gemini-2.0-flash:generateContent" in req.full_url
    assert "models/models" not in req.full_url

