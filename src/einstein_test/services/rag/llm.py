"""LLM para o RAG: Protocol, fakes e adapters (OpenAI-compatível + Ollama).

Contém `ChatLLM` (Protocol), `FakeChatLLM` (determinístico p/ testes),
`ChatOpenAIAdapter` (client real com retry transiente no primeiro token) e
`ChatOllamaAdapter` (modelos locais via Ollama, endpoint OpenAI-compatível),
além de `build_chat_llm` (factory que roteia pelo prefixo `ollama/`) e das
constantes de configuração e do helper `allowed_models()`.
"""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from einstein_test.services.rag.retrieval import _is_transient
from einstein_test.settings import get_settings

DEFAULT_CHAT_MODEL = "azure/gpt-5-mini"
DEFAULT_BASE_URL = "https://spillway.ensinoeinstein.com/openai/v1"

# Modelos locais via Ollama (endpoint OpenAI-compatível).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_PREFIX = "ollama/"

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1"))

DEFAULT_ALLOWED_MODELS = "azure/gpt-5-mini,azure/gpt-6-luna"


def is_ollama_model(model: str | None) -> bool:
    """True se o identificador de modelo é um modelo local do Ollama."""
    return bool(model) and model.startswith(OLLAMA_PREFIX)


def allowed_models() -> list[str]:
    """Modelos de LLM permitidos, lidos do env `ALLOWED_MODELS`.

    Lido por request (call-time) para refletir mudanças do operador; usa o
    default quando ausente/vazio.
    """
    from dotenv import load_dotenv

    load_dotenv()
    raw = os.getenv("ALLOWED_MODELS") or DEFAULT_ALLOWED_MODELS
    return [m.strip() for m in raw.split(",") if m.strip()]


class ChatLLM(Protocol):
    def stream(self, messages) -> Iterator[str]: ...
    def astream(self, messages) -> AsyncIterator[str]: ...
    def bind_tools(self, tools): ...


class FakeChatLLM:
    """LLM determinístico para testes (emite tokens fixos)."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens or ["token1", "token2"]

    def stream(self, messages) -> Iterator[str]:
        yield from self.tokens

    async def astream(self, messages) -> AsyncIterator[str]:
        for token in self.tokens:
            yield token

    def bind_tools(self, tools):
        """Suporte a tool-calling para o agente: responde direto sem tool."""

        class _Bound:
            def invoke(self, messages):
                from langchain_core.messages import AIMessage

                return AIMessage(content=" ".join(self_outer.tokens))

        self_outer = self
        return _Bound()


def _retrying() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=LLM_RETRY_BACKOFF, min=0.05, max=5),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )


def _retrying_sync() -> Retrying:
    return Retrying(
        stop=stop_after_attempt(LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=LLM_RETRY_BACKOFF, min=0.05, max=5),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )


def _as_prompt(messages) -> str:
    """Normaliza a entrada do LLM (str ou lista de BaseMessage) em prompt str."""
    if isinstance(messages, str):
        return messages
    return "\n".join(getattr(m, "content", str(m)) for m in messages)


class ChatOpenAIAdapter:
    """ChatLLM real via langchain-openai ChatOpenAI (azure/gpt-5-mini)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        cfg = get_settings()
        self.api_key = api_key or cfg.openai_api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY não configurada (defina no .env)")
        from langchain_openai import ChatOpenAI

        self.model = model or cfg.openai_chat_model
        self.base_url = base_url or cfg.openai_base_url
        self._llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            streaming=True,
        )

    def stream(self, messages) -> Iterator[str]:
        prompt = _as_prompt(messages)
        # retry transiente na aquisição do 1º token (não depois)
        it = iter(self._llm.stream(prompt))
        while True:
            try:
                first = next(it)
                break
            except StopIteration:
                return
            except Exception as exc:  # noqa: BLE001
                if not _is_transient(exc):
                    raise
                for attempt in _retrying_sync():
                    with attempt:
                        it = iter(self._llm.stream(prompt))
                        first = next(it)
                break
        yield (first.content or "")
        for c in it:
            yield (c.content or "")

    async def astream(self, messages) -> AsyncIterator[str]:
        prompt = _as_prompt(messages)
        # retry transiente apenas ANTES do primeiro token
        it: Any = None
        first: Any = None
        try:
            async for attempt in _retrying():
                with attempt:
                    it = self._llm.astream(prompt).__aiter__()
                    first = await it.__anext__()
        except StopAsyncIteration:
            return
        yield (getattr(first, "content", None) or "")
        async for chunk in it:
            yield (getattr(chunk, "content", None) or "")

    def bind_tools(self, tools):
        """Delega ao ChatOpenAI do langchain para tool-calling (agente)."""
        return self._llm.bind_tools(tools)


class ChatOllamaAdapter(ChatOpenAIAdapter):
    """ChatLLM local via Ollama (endpoint OpenAI-compatível em localhost).

    Reusa o mesmo client/streaming/retry/tool-calling do `ChatOpenAIAdapter`
    contra o endpoint OpenAI-compatível do Ollama (`OLLAMA_BASE_URL`, default
    `http://localhost:11434/v1`). O Ollama não exige chave — usa `ollama` como
    api_key fictícia para satisfazer o client OpenAI.
    """

    def __init__(self, model: str | None = None) -> None:
        if model and model.startswith(OLLAMA_PREFIX):
            model = model.removeprefix(OLLAMA_PREFIX)
        super().__init__(
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            model=model or OLLAMA_DEFAULT_MODEL,
        )


def build_chat_llm(model: str | None = None) -> ChatLLM:
    """Factory de ChatLLM: roteia pelo prefixo do modelo.

    `ollama/<nome>` -> `ChatOllamaAdapter` (local); qualquer outro -> o adapter
    OpenAI-compatível padrão (`ChatOpenAIAdapter`).
    """
    if is_ollama_model(model):
        return ChatOllamaAdapter(model=model)
    return ChatOpenAIAdapter(model=model)


__all__ = [
    "ChatLLM",
    "FakeChatLLM",
    "ChatOpenAIAdapter",
    "ChatOllamaAdapter",
    "build_chat_llm",
    "is_ollama_model",
    "allowed_models",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_BASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_DEFAULT_MODEL",
    "OLLAMA_PREFIX",
    "LLM_MAX_RETRIES",
    "LLM_RETRY_BACKOFF",
]