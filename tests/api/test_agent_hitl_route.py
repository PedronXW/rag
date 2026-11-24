"""G-003 — Testes das rotas /agent/hitl e /agent/hitl/resume (AC-006), herméticos."""

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
    return register_and_login(client, email="user-g003@b.com")


def _chat(client, headers):
    chat = client.post("/chats", json={"name": "Chat"}, headers=headers).json()
    return chat


def test_ac006_hitl_devolve_pending(client):
    """AC-006: POST /chats/{chat_id}/agent/hitl devolve estado pending com draft."""
    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)
    seed_ready_file(engine, chat["id"])
    resp = c.post(
        f"/chats/{chat['id']}/agent/hitl",
        json={"content": "pergunta"},
        headers=h,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["draft"]  # draft preenchido (não vazio)

def test_ac006_resume_aprovado_entrega(client):
    """AC-006: /agent/hitl/resume approved=True entrega a resposta e persiste."""
    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)
    seed_ready_file(engine, chat["id"])
    c.post(f"/chats/{chat['id']}/agent/hitl", json={"content": "pergunta"}, headers=h)
    resp = c.post(
        f"/chats/{chat['id']}/agent/hitl/resume",
        json={"approved": True},
        headers=h,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "approved"
    assert body["answer"]
