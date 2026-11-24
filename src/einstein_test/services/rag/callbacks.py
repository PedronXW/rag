"""Observabilidade Langfuse do RAG (G-005).

`get_langfuse_handler` retorna o callback de trace do Langfuse quando
habilitado (`LANGFUSE_ENABLED`); `graph_config` injeta `callbacks` no config do
graph. Desabilitado por padrão (testes determinísticos intactos).
"""

import os
from functools import lru_cache

from langfuse import Langfuse


def langfuse_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"


def _base_url() -> str:
    # SDK v5 lê LANGFUSE_BASE_URL; mantém compat com LANGFUSE_HOST
    return os.getenv(
        "LANGFUSE_BASE_URL",
        os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )


@lru_cache(maxsize=1)
def _get_client() -> Langfuse:
    """Client global do Langfuse inicializado com as credenciais do ambiente."""
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-local"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-local"),
        base_url=_base_url(),
        tracing_enabled=True,
    )


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """Handler de callback Langfuse para traces por execução do graph."""
    if not langfuse_enabled():
        return None
    try:
        _get_client()  # garante o client inicializado (habilitado)
        from langfuse.langchain import CallbackHandler

        # CallbackHandler() sem public_key reusa o client global habilitado
        return CallbackHandler()
    except Exception:
        return None


def graph_config(chat_id: int | None = None) -> dict:
    """Config do graph com callbacks do Langfuse quando habilitado."""
    config: dict = {}
    if langfuse_enabled():
        handler = get_langfuse_handler()
        if handler is not None:
            config["callbacks"] = [handler]
    if chat_id is not None:
        config["configurable"] = {"thread_id": str(chat_id)}
    return config


__all__ = ["get_langfuse_handler", "graph_config", "langfuse_enabled"]