"""Unit tests for the provider-agnostic LLM client (mocked, no real API calls)."""

from unittest.mock import MagicMock

import pytest

from src.llm.client import Completion, LLMClient, Message


def _mk_openai_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 7
    return resp


@pytest.fixture
def env_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    # Clear the config singleton
    from src.utils.config import get_env

    get_env.cache_clear()


def test_openai_compatible_path(env_groq, monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient()
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = _mk_openai_response("hello world")
    client._client = fake_openai  # bypass lazy init
    client._init_client = lambda: None  # type: ignore[method-assign]

    out = client.chat([Message(role="user", content="hi")])
    assert isinstance(out, Completion)
    assert out.text == "hello world"
    assert out.provider == "groq"
    assert out.prompt_tokens == 10
    assert out.completion_tokens == 7
    fake_openai.chat.completions.create.assert_called_once()


def test_message_to_openai() -> None:
    m = Message(role="user", content="hi there")
    d = m.to_openai()
    assert d == {"role": "user", "content": "hi there"}
