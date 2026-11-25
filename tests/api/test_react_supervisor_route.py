"""H-004 — Testes da rota /agent/supervisor (AC-004) e regressão (AC-005)."""

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
    return register_and_login(client, email="user-sup@b.com")


def test_ac005_messages_continua(client):
    """AC-005: o /messages (RAG clássico) continua funcionando."""
    c, engine = client
    h = _auth(c)
    chat = c.post("/chats", json={"name": "Chat"}, headers=h).json()
    seed_ready_file(engine, chat["id"])
    resp = c.post(
        f"/chats/{chat['id']}/messages",
        json={"content": "pergunta"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "[DONE]" in resp.text


def test_ac004_supervisor_rota_sse(client):
    """AC-004: POST /chats/{chat_id}/agent/supervisor responde SSE."""
    c, engine = client
    h = _auth(c)
    chat = c.post("/chats", json={"name": "Chat"}, headers=h).json()
    seed_ready_file(engine, chat["id"])
    resp = c.post(
        f"/chats/{chat['id']}/agent/supervisor",
        json={"content": "pergunta"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "[DONE]" in resp.text