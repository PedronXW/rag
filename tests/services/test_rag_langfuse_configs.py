"""J-001 — Testes do tracing Langfuse nos graphs novos."""

import pytest


@pytest.fixture
def enable_langfuse(monkeypatch):
    from einstein_test.services.rag.callbacks import _get_client, get_langfuse_handler

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-local")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-local")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    _get_client.cache_clear()
    get_langfuse_handler.cache_clear()
    yield
    _get_client.cache_clear()
    get_langfuse_handler.cache_clear()


def test_agent_config_injeta_callbacks(enable_langfuse):
    """AC-001: agent_config injeta callbacks quando habilitado, mantendo thread_id."""
    from einstein_test.services.rag.agent import agent_config

    cfg = agent_config(chat_id=7)
    assert "callbacks" in cfg
    assert cfg["configurable"]["thread_id"] == "7"


def test_supervisor_config_injeta_callbacks(enable_langfuse):
    """AC-002: supervisor_config injeta callbacks quando habilitado."""
    from einstein_test.services.rag.supervisor import supervisor_config

    cfg = supervisor_config(chat_id=7)
    assert "callbacks" in cfg
    assert cfg["configurable"]["thread_id"] == "supervisor:7"


def test_multi_config_injeta_callbacks(enable_langfuse):
    """AC-003: multi_config injeta callbacks, mantendo recursion_limit."""
    from einstein_test.services.rag.multiquery import multi_config

    cfg = multi_config(chat_id=7)
    assert "callbacks" in cfg
    assert cfg["configurable"]["thread_id"] == "multi:7"
    assert "recursion_limit" in cfg


def test_hitl_config_injeta_callbacks(enable_langfuse):
    """AC-004: hitl_config injeta callbacks quando habilitado."""
    from einstein_test.services.rag.hitl import hitl_config

    cfg = hitl_config(chat_id=7)
    assert "callbacks" in cfg
    assert cfg["configurable"]["thread_id"] == "hitl:7"


def test_sem_langfuse_nao_injeta_callbacks(monkeypatch):
    """AC-005: com Langfuse desabilitado, os configs não têm callbacks."""
    from einstein_test.services.rag.callbacks import get_langfuse_handler

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_langfuse_handler.cache_clear()
    try:
        from einstein_test.services.rag.agent import agent_config

        cfg = agent_config(chat_id=7)
        assert "callbacks" not in cfg
        assert cfg["configurable"]["thread_id"] == "7"
    finally:
        get_langfuse_handler.cache_clear()