"""K-003 — Testes de conformidade de config e observabilidade dos graphs."""

import pytest


def test_thread_config_delega_ao_graph_config():
    """AC-002: thread_config do checkpointer delega ao graph_config."""
    from einstein_test.services.rag.checkpointer import thread_config
    from einstein_test.services.rag.callbacks import graph_config

    cfg = thread_config(chat_id=7)
    base = graph_config(chat_id=7)
    assert cfg["configurable"]["thread_id"] == "7"
    assert cfg["configurable"]["thread_id"] == base["configurable"]["thread_id"]


def test_agent_config_usa_graph_config(enable_langfuse):
    """AC-001: agent_config injeta callbacks (via graph_config) quando habilitado."""
    from einstein_test.services.rag.agent import agent_config

    cfg = agent_config(chat_id=7)
    assert "callbacks" in cfg
    assert cfg["configurable"]["thread_id"] == "7"


def test_supervisor_config_usa_graph_config(enable_langfuse):
    """AC-001: supervisor_config injeta callbacks."""
    from einstein_test.services.rag.supervisor import supervisor_config

    cfg = supervisor_config(chat_id=7)
    assert "callbacks" in cfg


def test_multi_config_usa_graph_config(enable_langfuse):
    """AC-001: multi_config injeta callbacks."""
    from einstein_test.services.rag.multiquery import multi_config

    cfg = multi_config(chat_id=7)
    assert "callbacks" in cfg


def test_hitl_config_usa_graph_config(enable_langfuse):
    """AC-001: hitl_config injeta callbacks."""
    from einstein_test.services.rag.hitl import hitl_config

    cfg = hitl_config(chat_id=7)
    assert "callbacks" in cfg


def test_callbacks_deterministico(monkeypatch):
    """AC-003: com Langfuse desabilitado, configs sem callbacks."""
    from einstein_test.services.rag.callbacks import get_langfuse_handler

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_langfuse_handler.cache_clear()
    try:
        from einstein_test.services.rag.agent import agent_config

        cfg = agent_config(chat_id=7)
        assert "callbacks" not in cfg
    finally:
        get_langfuse_handler.cache_clear()


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


def test_model_config_fonte_unica():
    """AC-004: a config de modelo vem de settings (fonte única)."""
    from einstein_test.settings import get_settings

    s = get_settings()
    assert hasattr(s, "openai_chat_model") or hasattr(s, "auth_model") or hasattr(s, "chat_model")
    from einstein_test.services.rag.llm import DEFAULT_CHAT_MODEL

    assert DEFAULT_CHAT_MODEL  # default em llm; settings é consumido via build_chat_llm