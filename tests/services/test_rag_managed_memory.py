"""I-002 — Testes da memória longa como estado gerenciado no graph."""

import pytest


@pytest.fixture
def memory_store():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    from einstein_test.services.rag.longterm import set_memory_store

    set_memory_store(store)
    yield store
    from einstein_test.services.rag.longterm import reset_memory_store

    reset_memory_store()


def test_graph_aceita_store_gerenciado(memory_store):
    """AC-001: o graph aceita o store como estado gerenciado."""
    from einstein_test.services.rag.managed import build_graph_with_memory_store

    g = build_graph_with_memory_store(memory_store)
    assert g is not None
    assert hasattr(g, "invoke")


def test_no_memoria_recupera_do_store(memory_store):
    """AC-002: o nó de memória recupera a memória do usuário e inclui no estado."""
    from langchain_core.messages import HumanMessage

    from einstein_test.services.rag.longterm import put_memory
    from einstein_test.services.rag.managed import memory_node

    put_memory(5, "preferencia", {"valor": "responder em pt-BR"})

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    state = {"question": "oi", "chat_id": 5, "user_id": 5, "messages": [HumanMessage(content="oi")]}
    out = memory_node(state, store=memory_store)
    assert out.get("memory"), "a memória deve ser recuperada"
    assert "responder em pt-BR" in out["memory"]


def test_memoria_entra_no_prompt(memory_store):
    """AC-003: a memória recuperada entra no prompt de geração."""
    from langgraph.checkpoint.memory import MemorySaver

    from einstein_test.services.rag.longterm import put_memory
    from einstein_test.services.rag.managed import build_graph_with_memory_store

    put_memory(7, "preferencia", {"valor": "curto"})

    class _LLM:
        def stream(self, messages):
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

    g = build_graph_with_memory_store(
        memory_store,
        llm=_LLM(),
        retriever=_Ret(),
        embedder=_Emb(),
        checkpointer=MemorySaver(),
    )
    out = g.invoke(
        {"question": "oi", "chat_id": 7, "user_id": 7},
        config={"configurable": {"thread_id": "7"}},
    )
    assert out.get("answer")
    assert "curto" in out["answer"]


def test_memoria_isolada_por_usuario(memory_store):
    """AC-004: a memória é isolada por usuário."""
    from einstein_test.services.rag.longterm import put_memory
    from einstein_test.services.rag.managed import memory_node

    put_memory(1, "k", {"valor": "do usuário 1"})

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    from langchain_core.messages import HumanMessage

    out = memory_node(
        {"question": "q", "chat_id": 2, "user_id": 2, "messages": [HumanMessage(content="q")]},
        store=memory_store,
    )
    assert not out.get("memory")  # usuário 2 não tem memória


def test_endpoints_memoria_continuam(memory_store):
    """AC-005: os endpoints GET/PUT /users/{id}/memory continuam (regressão)."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from einstein_test.db.base import Base
    from einstein_test.main import create_app
    from tests.conftest import register_and_login

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    from einstein_test.api.dependencies import configure_session_factory

    configure_session_factory(engine)
    from einstein_test.services.rag.longterm import set_memory_store

    set_memory_store(memory_store)
    app = create_app()
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        h = register_and_login(c, email="managed@b.com")
        user = c.get("/users/me", headers=h).json()
        user_id = user.get("id", 1)
        r = c.put(f"/users/{user_id}/memory", json={"key": "p", "value": {"v": 1}}, headers=h)
        assert r.status_code == 200
        g = c.get(f"/users/{user_id}/memory", headers=h)
        assert g.status_code == 200
        assert "items" in g.json()
    engine.dispose()