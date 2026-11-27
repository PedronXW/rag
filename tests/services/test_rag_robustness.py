"""H-005 — Testes do retry de nós e avaliação do RAG."""

import pytest


def test_retry_node_reexecuta_em_falha_transiente():
    """AC-001/002: retry_node re-executa o nó e absorve falha transiente."""
    from einstein_test.services.rag.retry import retry_node

    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("falha transiente")
        return {"ok": True}

    result = retry_node(_flaky, retries=2, backoff=0.0)
    assert result == {"ok": True}
    assert calls["n"] == 2


def test_retry_node_esgota_apos_retries():
    """AC-001: após esgotar os retries, a falha é relançada."""
    from einstein_test.services.rag.retry import retry_node

    def _always_fail():
        raise ValueError("falha permanente")

    with pytest.raises(ValueError):
        retry_node(_always_fail, retries=2, backoff=0.0)


class _FakeLLM:
    def stream(self, messages):
        yield "resposta"

    async def astream(self, messages):
        yield "resposta"


class _Ret:
    def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
        return []


class _Emb:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


def test_retry_integrado_ao_graph():
    """AC-001/002: o retrieve do graph absorve falha transiente via retry."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from einstein_test.services.rag.graph import build_graph_with_state

    class _FlakyRet:
        def __init__(self):
            self.calls = 0

        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("falha transiente")
            return []

    class _FEmb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    retriever = _FlakyRet()
    g = build_graph_with_state(
        llm=_FakeLLM(),
        retriever=retriever,
        embedder=_FEmb(),
        async_=False,
        checkpointer=MemorySaver(),
    )
    out = g.invoke(
        {"question": "q", "chat_id": 1, "messages": [HumanMessage(content="q")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert retriever.calls == 2
    assert out["answer"]
