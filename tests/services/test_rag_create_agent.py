"""I-004 — Testes da migração create_react_agent → create_agent."""

import pytest


def test_langchain_dependencia():
    """AC-001: langchain é dependência do projeto."""
    import langchain

    assert langchain is not None


def test_build_agent_usa_create_agent():
    """AC-002: build_agent usa langchain.agents.create_agent."""
    import inspect

    from einstein_test.services.rag.react import build_agent

    src = inspect.getsource(build_agent)
    assert "langchain.agents.create_agent" in src or "from langchain.agents" in src
    assert "create_react_agent" not in src


def test_build_agent_compila():
    """AC-003: build_agent existe e é chamável (create_agent)."""
    from einstein_test.services.rag.react import build_agent, react_config

    assert callable(build_agent)
    cfg = react_config(chat_id=1)
    assert cfg["configurable"]["thread_id"] == "react:1"


def test_react_multi_tenant():
    """AC-004: o react_config é por chat (multi-tenant)."""
    from einstein_test.services.rag.react import react_config

    cfg1 = react_config(chat_id=1)
    cfg2 = react_config(chat_id=2)
    assert cfg1["configurable"]["thread_id"] != cfg2["configurable"]["thread_id"]