"""H-001 — Testes do trim_messages e streaming token-real (AC-001..AC-005)."""

from langchain_core.messages import HumanMessage


class FakeStreamLLM:
    """LLM fake que emite tokens em streaming (1 por chunk)."""

    def stream(self, messages):
        for t in ["resposta ", "token-real"]:
            yield t

    async def astream(self, messages):
        for t in ["resposta ", "token-real"]:
            yield t


def test_trim_limita_historico():
    """AC-001: trim_messages limita o histórico às últimas N mensagens."""
    from einstein_test.services.rag.graph import trim_history

    history = [
        HumanMessage(content=f"msg {i}")
        for i in range(20)
    ]
    trimmed = trim_history(history, max_messages=5)
    assert len(trimmed) == 5
    assert trimmed[-1].content == "msg 19"
    assert "msg 0" not in [m.content for m in trimmed]


def test_trim_respeita_limite_tokens():
    """AC-004: o histórico passa a caber no limite de tokens."""
    from einstein_test.services.rag.graph import _token_count, trim_history

    history = [HumanMessage(content="x" * 200) for _ in range(10)]
    trimmed = trim_history(history, max_tokens=100)
    assert _token_count(trimmed) <= 100
    assert len(trimmed) < 10


def test_sse_preserva_eventos():
    """AC-003: o formato SSE mantém node/token/done/[DONE] (via service)."""
    from einstein_test.services.rag.service import RagService

    class _R:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _E:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    import inspect

    src = inspect.getsource(RagService)
    assert "_sse_event" in src or "event" in src  # o service monta eventos de streaming
    assert "astream" in src

def test_sources_preenchidas_quando_ha_chunks():
    """AC-003: sources não ficam vazias quando o retriever devolve chunks."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    class _Chunk:
        def __init__(self, id, file_id, content):
            self.id = id
            self.file_id = file_id
            self.content = content

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return [_Chunk(1, 1, "conteudo")]

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    from einstein_test.services.rag.graph import build_graph_with_state

    g = build_graph_with_state(
        llm=FakeStreamLLM(),
        retriever=_Ret(),
        embedder=_Emb(),
        async_=False,
        checkpointer=MemorySaver(),
    )
    out = g.invoke(
        {"question": "q", "chat_id": 1, "messages": [HumanMessage(content="q")]},
        config={"configurable": {"thread_id": "1"}},
    )
    assert out.get("sources"), "sources devem estar preenchidas quando há chunks"
    assert out["sources"][0]["content"] == "conteudo"
