"""H-002 — Testes da memória de longo prazo (PostgresStore + endpoints)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from einstein_test.db.base import Base
from einstein_test.main import create_app
from tests.conftest import register_and_login


@pytest.fixture
def memory_store():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    yield store
    from einstein_test.services.rag.longterm import reset_memory_store

    reset_memory_store()


@pytest.fixture
def client(memory_store, monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    from einstein_test.api.dependencies import configure_session_factory

    configure_session_factory(engine)
    from einstein_test.services.rag.longterm import set_memory_store

    set_memory_store(memory_store)
    app = create_app()
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        yield c, engine
    engine.dispose()


def _auth(client):
    return register_and_login(client, email="user-mem@b.com")


def test_store_put_get(memory_store):
    """AC-002/003: put_memory grava e get_memory recupera."""
    from einstein_test.services.rag.longterm import put_memory, get_memory

    put_memory(7, "preferencia", {"valor": "respostas curtas"})
    got = get_memory(7, "preferencia")
    assert got is not None
    assert got.value["valor"] == "respostas curtas"


def test_store_namespace_isolado(memory_store):
    """AC-006: dois usuários distintos não compartilham memória."""
    from einstein_test.services.rag.longterm import put_memory, get_memory

    put_memory(1, "k", {"v": "do usuário 1"})
    assert get_memory(2, "k") is None


def test_store_nao_encontra_chave_ausente(memory_store):
    """AC-003: get de chave inexistente retorna None."""
    from einstein_test.services.rag.longterm import get_memory

    assert get_memory(99, "inexistente") is None


def test_postgres_store_factory(monkeypatch):
    """AC-001: existe um PostgresStore criado com setup()."""
    from langgraph.store.postgres import PostgresStore

    from einstein_test.services.rag.longterm import _pool, build_postgres_store

    class _FakePool:
        def __init__(self):
            self.closed = False

    monkeypatch.setattr("einstein_test.services.rag.longterm._pool", lambda: _FakePool())
    monkeypatch.setattr(
        "einstein_test.services.rag.longterm.PostgresStore.setup",
        lambda self: None,
    )
    _pool.cache_clear()
    build_postgres_store.cache_clear()
    try:
        store = build_postgres_store()
        assert isinstance(store, PostgresStore)
    finally:
        _pool.cache_clear()
        build_postgres_store.cache_clear()


def test_api_put_get_memory(client):
    """AC-004: PUT e GET /users/{id}/memory funcionam."""
    c, engine = client
    h = _auth(c)
    user = c.get("/users/me", headers=h).json()
    user_id = user["id"] if isinstance(user, dict) and "id" in user else 1

    put = c.put(
        f"/users/{user_id}/memory",
        json={"key": "preferencia", "value": {"estilo": "curto"}},
        headers=h,
    )
    assert put.status_code == 200

    get = c.get(f"/users/{user_id}/memory", headers=h)
    assert get.status_code == 200
    body = get.json()
    assert "items" in body
    assert any(i.get("key") == "preferencia" for i in body["items"])


def test_graph_inclui_memoria_no_prompt(memory_store):
    """AC-005: o graph recupera a memória e a inclui no prompt."""
    from langgraph.checkpoint.memory import MemorySaver

    from einstein_test.services.rag.longterm import put_memory
    from einstein_test.services.rag.service import RagService

    put_memory(5, "preferencia", {"valor": "responder em pt-BR"})

    class _LLM:
        def stream(self, messages):
            # ecoa o conteúdo do prompt (deve conter a memória)
            yield " ".join(getattr(m, "content", str(m)) for m in messages)

        async def astream(self, messages):
            for t in self.stream(messages):
                yield t

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    svc = RagService(llm=_LLM(), retriever=_Ret(), embedder=_Emb(), checkpointer=MemorySaver())
    tokens = list(svc.stream(question="oi", chat_id=5, history=[], user_id=5))
    prompt = "".join(tokens)
    assert "responder em pt-BR" in prompt