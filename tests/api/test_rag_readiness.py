"""I-005 — Testes do gate de readiness nas rotas RAG novas."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from einstein_test.db.base import Base
from einstein_test.main import create_app
from einstein_test.services.rag import FakeChatLLM, RagService, set_rag_factory
from tests.conftest import register_and_login


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
    return register_and_login(client, email="user-ready@b.com")


def _chat(client, headers):
    return client.post("/chats", json={"name": "Chat"}, headers=headers).json()


@pytest.mark.parametrize("route", ["agent", "multi-query", "agent/supervisor"])
def test_sem_arquivos_emite_no_ready_files(client, route):
    """AC-001: rotas sem arquivos prontos emitem notice no_ready_files (sem LLM)."""
    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)  # sem seed de arquivo
    resp = c.post(
        f"/chats/{chat['id']}/{route}",
        json={"content": "pergunta"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "no_ready_files" in resp.text or "notice" in resp.text


@pytest.mark.parametrize("route", ["agent", "multi-query", "agent/supervisor"])
def test_com_arquivo_roda_normal(client, route):
    """AC-003: com arquivos prontos, a rota roda (resposta gerada)."""
    from tests.conftest import seed_ready_file

    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)
    seed_ready_file(engine, chat["id"])
    resp = c.post(
        f"/chats/{chat['id']}/{route}",
        json={"content": "pergunta"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "[DONE]" in resp.text


def test_messages_regressao(client):
    """AC-005: o /messages continua funcionando (com e sem arquivo)."""
    from tests.conftest import seed_ready_file

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

def test_partial_files_emite_notice(client):
    """AC-002: arquivos parciais (pending) emitem notice partial_files."""
    import sqlalchemy as sa

    from einstein_test.models.file import File
    from einstein_test.models.file_chunk import FileChunk

    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)
    with engine.begin() as conn:
        conn.execute(
            sa.text("INSERT INTO files (chat_id, name, url, size, status) VALUES (:c, 'a.txt', 's3://b/a', 5, 'pending')"),
            {"c": chat["id"]},
        )
    resp = c.post(
        f"/chats/{chat['id']}/multi-query",
        json={"content": "pergunta"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "partial_files" in resp.text or "notice" in resp.text


def test_rate_limit_429(client, monkeypatch):
    """AC-004: a rota /agent é rate-limitada (429 quando excede)."""
    from einstein_test.middleware.rate_limit import RateLimiter

    monkeypatch.setenv("RATE_LIMIT_AGENT", "3")
    RateLimiter.reset()
    c, engine = client
    h = _auth(c)
    chat = _chat(c, h)
    codes = []
    for _ in range(5):
        resp = c.post(
            f"/chats/{chat['id']}/agent",
            json={"content": "pergunta"},
            headers=h,
        )
        codes.append(resp.status_code)
    assert 429 in codes, f"esperava 429, obtive {codes}"
