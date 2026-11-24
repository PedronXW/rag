"""G-001 — Testes do SSE com eventos por nó e sources no [DONE] (AC-004/AC-005).

O endpoint POST /chats/{chat_id}/messages, com o graph em MessagesState +
checkpointer, emite:
  - event: node (retrieve/grade/generate)
  - data: token (resposta em streaming)
  - data: [DONE] contendo message_id e sources
"""

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
    h = register_and_login(client, email="user-g001@b.com")
    return h


def _chat(client, headers):
    chat = client.post("/chats", json={"name": "Chat"}, headers=headers).json()
    return chat


def test_ac004_sse_emite_event_node(client):
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
    assert "event: node" in resp.text, "SSE deve emitir eventos por nó"


def test_ac005_done_inclui_sources(client):
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
    body = resp.text
    assert "[DONE]" in body
    assert "sources" in body, "DONE deve incluir as sources recuperadas"