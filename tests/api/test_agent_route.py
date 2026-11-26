"""G-002 — Testes da rota /agent (SSE) e regressão do /messages (AC-005/AC-006)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from einstein_test.db.base import Base
from einstein_test.main import create_app
from einstein_test.services.rag import FakeChatLLM, RagService, set_rag_factory
from tests.conftest import register_and_login, seed_ready_file


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    from einstein_test.api.dependencies import configure_session_factory

    configure_session_factory(engine)

    from einstein_test.vector import FakeEmbedder

    def _factory(*args, **kwargs):
        return RagService(
            llm=FakeChatLLM(),
            retriever=kwargs["retriever"],
            embedder=FakeEmbedder(),
        )

    set_rag_factory(_factory)

    app = create_app()
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        yield c, engine

    from einstein_test.services.rag import reset_rag_factory

    reset_rag_factory()
    engine.dispose()


def _auth(client):
    return register_and_login(client, email="user-g002@b.com")


def _chat(client, headers):
    return client.post("/chats", json={"name": "Chat"}, headers=headers).json()


def test_ac006_messages_classico_continua(client):
    """AC-006: POST /chats/{chat_id}/messages (RAG clássico) continua funcionando."""
    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)
    seed_ready_file(engine, chat["id"])
    resp = c.post(
        f"/chats/{chat['id']}/messages",
        json={"content": "pergunta"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "[DONE]" in resp.text


def test_ac005_agent_rota_responde_sse(client):
    """AC-005: POST /chats/{chat_id}/agent responde SSE com answer + [DONE]."""
    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)
    seed_ready_file(engine, chat["id"])
    resp = c.post(
        f"/chats/{chat['id']}/agent",
        json={"content": "pergunta"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "[DONE]" in resp.text
    assert "event: error" not in resp.text
    assert "answer" in resp.text


def test_rotas_rag_usam_modelo_do_usuario(client, monkeypatch):
    """As rotas RAG propagam o modelo escolhido pelo usuário (F-022).

    Regressão: /agent, /multi-query e /agent/supervisor agora passam o modelo do
    usuário ao `_rag_service` (antes usavam o default do ambiente).
    """
    c, engine = client
    h = _auth(c)
    r = c.patch("/users/me", json={"model": "azure/gpt-6-luna"}, headers=h)
    assert r.status_code == 200

    import einstein_test.api.messages as messages_module

    original = messages_module._rag_service
    calls: list[str | None] = []

    def _spy(session, model=None):
        calls.append(model)
        return original(session, model)

    monkeypatch.setattr(messages_module, "_rag_service", _spy)

    chat = _chat(c, h)
    seed_ready_file(engine, chat["id"])
    for path in ("/messages", "/agent", "/multi-query", "/agent/supervisor"):
        resp = c.post(
            f"/chats/{chat['id']}{path}",
            json={"content": "pergunta"},
            headers={**h, "Accept": "text/event-stream"},
        )
        assert resp.status_code == 200, path
        assert "event: error" not in resp.text, path

    assert len(calls) >= 4, calls
    assert all(m == "azure/gpt-6-luna" for m in calls), calls