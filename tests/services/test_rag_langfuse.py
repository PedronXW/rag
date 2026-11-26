"""G-005 — Testes da observabilidade Langfuse (AC-001..AC-004)."""

import yaml
from pathlib import Path


def test_langfuse_no_docker_compose():
    """AC-001: o docker-compose tem serviço langfuse (self-host)."""
    compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    services = data.get("services", {})
    assert "langfuse" in services


def test_handler_none_quando_desabilitado(monkeypatch):
    """AC-003: com LANGFUSE_ENABLED desabilitado, o handler é None."""
    from einstein_test.services.rag.callbacks import (
        get_langfuse_handler,
        langfuse_enabled,
    )

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_langfuse_handler.cache_clear()
    assert langfuse_enabled() is False
    assert get_langfuse_handler() is None


def test_handler_criado_quando_habilitado(monkeypatch):
    """AC-002: com LANGFUSE_ENABLED habilitado, o handler é criado com client ativo."""
    from einstein_test.services.rag.callbacks import _get_client, get_langfuse_handler

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-local")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-local")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    _get_client.cache_clear()
    get_langfuse_handler.cache_clear()
    try:
        handler = get_langfuse_handler()
        assert handler is not None
        from langfuse.langchain import CallbackHandler

        assert isinstance(handler, CallbackHandler)
        # o client está habilitado (não é um NoOp desabilitado)
        client = _get_client()
        assert getattr(client, "_tracing_enabled", False) is True
    finally:
        _get_client.cache_clear()
        get_langfuse_handler.cache_clear()


def test_graph_config_injeta_callbacks_quando_habilitado(monkeypatch):
    """AC-002: graph_config injeta callbacks quando habilitado."""
    from einstein_test.services.rag.callbacks import (
        get_langfuse_handler,
        graph_config,
    )

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-local")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-local")
    get_langfuse_handler.cache_clear()
    cfg = graph_config(chat_id=1)
    assert "callbacks" in cfg
    assert cfg["configurable"]["thread_id"] == "1"