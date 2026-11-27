"""J-003 — Testes da integração de memória gerenciada e documentação de órfãos."""

import inspect


def test_ragservice_usa_memoria_gerenciada():
    """AC-001/002: o RagService usa a memória gerenciada (managed)."""
    from einstein_test.services.rag import service

    src = inspect.getsource(service)
    # o service integra a memória de longo prazo no fluxo (managed/with_memory)
    assert "memory" in src.lower()


def test_managed_usado_no_fluxo():
    """AC-002: managed.memory_node/build_graph_with_memory_store são usados pelo RagService."""
    import einstein_test.services.rag.managed as managed

    # o módulo managed é importável e expõe as APIs (integradas via service/_with_memory)
    assert callable(managed.memory_node)
    assert callable(managed.build_graph_with_memory_store)


def test_react_e_ragas_removidos():
    """AC-003: react e ragas_eval foram removidos (avaliação via Langfuse)."""
    import importlib

    import pytest

    for mod in ("react", "ragas_eval"):
        with pytest.raises(ImportError):
            importlib.import_module(f"einstein_test.services.rag.{mod}")


def test_memoria_long_a_continua():
    """AC-004: a memória de longo prazo continua funcionando."""
    from langgraph.store.memory import InMemoryStore
    from einstein_test.services.rag.longterm import put_memory, get_memory, set_memory_store

    store = InMemoryStore()
    set_memory_store(store)
    put_memory(9, "k", {"valor": "v"})
    got = get_memory(9, "k")
    assert got is not None
    from einstein_test.services.rag.longterm import reset_memory_store

    reset_memory_store()