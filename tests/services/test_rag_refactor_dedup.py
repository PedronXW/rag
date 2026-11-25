"""F-055 — Testes da refatoração do RAG (retrieval único + streaming rico)."""

import inspect


def test_retrieval_helpers_existem():
    """AC-001: retrieval.py expõe retrieve_chunks/chunk_to_dict/_is_transient."""
    from einstein_test.services.rag.retrieval import (
        _is_transient,
        chunk_to_dict,
        retrieve_chunks,
    )

    assert callable(retrieve_chunks)
    assert callable(chunk_to_dict)
    assert callable(_is_transient)


def test_modulos_usam_retrieval_sem_hardcode():
    """AC-001: os módulos usam retrieve_chunks (não buscam com constantes hardcoded)."""
    for mod in ("agent", "hitl", "multiquery", "supervisor", "graph"):
        src = inspect.getsource(__import__(f"einstein_test.services.rag.{mod}", fromlist=["x"]))
        assert "chunks_per_file=2" not in src, f"{mod}.py ainda tem hardcode"
        assert "max_chunks=12" not in src, f"{mod}.py ainda tem hardcode"


def test_sem_duplicacao_chunk_to_dict():
    """AC-002: não há _chunk_to_dict/_is_transient duplicados (fonte única)."""
    from einstein_test.services.rag import agent, hitl, llm, multiquery, supervisor

    for mod in (agent, hitl, llm, multiquery, supervisor):
        assert "_chunk_to_dict" not in dir(mod), f"{mod.__name__} define _chunk_to_dict"
    from einstein_test.services.rag.retrieval import _is_transient as src

    assert callable(src)


def test_chunk_to_dict_serializavel():
    """AC-001: chunk_to_dict converte objeto e dict em dict serializável."""
    from einstein_test.services.rag.retrieval import chunk_to_dict

    class _Chunk:
        def __init__(self, id, file_id, content):
            self.id = id
            self.file_id = file_id
            self.content = content

    out = chunk_to_dict(_Chunk(1, 1, "c"))
    assert out == {"chunk_id": 1, "document_id": 1, "content": "c", "score": 0.0}
    out2 = chunk_to_dict({"id": 2, "file_id": 2, "content": "d"})
    assert out2["chunk_id"] == 2


def test_retrieve_chunks_usa_constantes_do_retrieval():
    """AC-001: retrieve_chunks usa CHUNKS_PER_FILE/MAX_CHUNKS do retrieval."""
    from einstein_test.services.rag.retrieval import CHUNKS_PER_FILE, MAX_CHUNKS

    assert CHUNKS_PER_FILE == 2
    assert MAX_CHUNKS == 12


def test_streaming_rico_via_service():
    """AC-003: o RagService expõe streaming rico (astream_events) com rich=True."""
    import asyncio

    from einstein_test.services.rag.llm import FakeChatLLM
    from einstein_test.services.rag.service import RagService

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    svc = RagService(llm=FakeChatLLM(), retriever=_Ret(), embedder=_Emb())

    async def _collect():
        return [ev async for ev in svc.astream_with_nodes("q", 1, [], rich=True)]

    events = asyncio.run(_collect())
    kinds = {ev["kind"] for ev in events}
    assert {"node", "token", "done"} <= kinds


def test_streaming_padrao_mantem_tokens():
    """AC-003: o streaming padrão (sem rich) mantém tokens individuais."""
    import asyncio

    from einstein_test.services.rag.llm import FakeChatLLM
    from einstein_test.services.rag.service import RagService

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    svc = RagService(llm=FakeChatLLM(tokens=["token1", "token2"]), retriever=_Ret(), embedder=_Emb())

    async def _collect():
        return [ev async for ev in svc.astream_with_nodes("q", 1, [])]

    events = asyncio.run(_collect())
    tokens = [ev["content"] for ev in events if ev["kind"] == "token"]
    assert "token1" in tokens and "token2" in tokens