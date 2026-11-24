"""G-001 — Testes do state avançado (MessagesState + reducers) e memória de conversa.

O checkpointer é injetável: nos testes determinísticos (SQLite) usa-se um
`MemorySaver`; em produção/integração usa-se `PostgresSaver` (thread_id=chat_id).
"""

from typing import get_type_hints

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver


class FakeLLM:
    """LLM fake determinístico que aceita mensagens (novo Protocol)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def stream(self, messages: list) -> list[str]:
        self.calls.append(messages)
        # devolve um token por chamada, ecoando a última pergunta
        last = messages[-1].content if messages else "?"
        return [f"resposta-{last}"]

    async def astream(self, messages: list) -> list[str]:
        for t in self.stream(messages):
            yield t


def _fakeretriever(*, chunk_id: int = 1, content: str = "conteudo do arquivo"):
    class _R:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return [
                type("Chunk", (), {"id": chunk_id, "content": content, "chat_id": chat_id})()
            ]

    return _R()


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


def _build_graph(checkpointer=None, retriever=None, llm=None):
    from einstein_test.services.rag.graph import build_graph_with_state

    return build_graph_with_state(
        llm=llm or FakeLLM(),
        retriever=retriever or _fakeretriever(),
        embedder=_FakeEmbedder(),
        checkpointer=checkpointer,
    )


def test_estado_usa_messagesstate_com_add_messages():
    """AC-001: RagState usa MessagesState com canal messages via add_messages."""
    from langgraph.graph import add_messages

    from einstein_test.services.rag.graph import RagState

    hints = get_type_hints(RagState, include_extras=True)
    assert "messages" in hints
    ann = hints["messages"]
    assert hasattr(ann, "__metadata__")
    assert add_messages in ann.__metadata__


def test_estado_usa_reducer_de_substituicao_para_chunks():
    """AC-001: chunks/sources usam reducer de substituição (não acumulam entre turnos)."""
    from einstein_test.services.rag.graph import RagState

    hints = get_type_hints(RagState, include_extras=True)
    assert "chunks" in hints
    ann = hints["chunks"]
    assert hasattr(ann, "__metadata__")
    reducer = ann.__metadata__[0]
    assert reducer(["a"], ["b"]) == ["b"]


def test_checkpointer_postgres_factory(monkeypatch):
    """AC-002: o factory de checkpointer produz um PostgresSaver."""
    from langgraph.checkpoint.postgres import PostgresSaver

    from einstein_test.services.rag.checkpointer import build_postgres_checkpointer

    # o setup() exige conexão real; mockamos o pool para não conectar no unitário
    class _FakePool:
        def __init__(self):
            self.closed = False

    monkeypatch.setattr(
        "einstein_test.services.rag.checkpointer._pool",
        lambda: _FakePool(),
    )
    monkeypatch.setattr(
        "einstein_test.services.rag.checkpointer.PostgresSaver.setup",
        lambda self: None,
    )
    build_postgres_checkpointer.cache_clear()
    try:
        cp = build_postgres_checkpointer()
        assert isinstance(cp, PostgresSaver)
    finally:
        build_postgres_checkpointer.cache_clear()


def test_memoria_recupera_historico_no_mesmo_thread():
    """AC-003: invocações no mesmo thread_id acumulam o histórico de mensagens.

    O histórico contém a pergunta (Human) e a resposta (AI), sem duplicar a
    pergunta por turno.
    """
    cp = MemorySaver()
    g = _build_graph(checkpointer=cp)
    cfg = {"configurable": {"thread_id": 10}}

    g.invoke(
        {
            "question": "primeira pergunta",
            "chat_id": 10,
            "messages": [HumanMessage(content="primeira pergunta")],
        },
        config=cfg,
    )
    out = g.invoke(
        {
            "question": "segunda pergunta",
            "chat_id": 10,
            "messages": [HumanMessage(content="segunda pergunta")],
        },
        config=cfg,
    )
    contents = [m.content for m in out["messages"]]
    assert "primeira pergunta" in contents
    assert "segunda pergunta" in contents
    # sem duplicação da pergunta (cada pergunta entra uma vez no canal messages)
    assert contents.count("primeira pergunta") == 1
    assert contents.count("segunda pergunta") == 1
    # a resposta do LLM é gravada como AIMessage
    assert any(isinstance(m, AIMessage) for m in out["messages"])


def test_memoria_nao_compartilha_entre_threads():
    """AC-006: dois thread_id distintos não compartilham memória no checkpoint."""
    cp = MemorySaver()
    g = _build_graph(checkpointer=cp)

    g.invoke(
        {"question": "estado do chat A", "chat_id": 1, "messages": [HumanMessage(content="estado do chat A")]},
        config={"configurable": {"thread_id": 1}},
    )
    out = g.invoke(
        {"question": "estado do chat B", "chat_id": 2, "messages": [HumanMessage(content="estado do chat B")]},
        config={"configurable": {"thread_id": 2}},
    )
    contents = [m.content for m in out["messages"]]
    assert "estado do chat A" not in contents
    assert "estado do chat B" in contents


def test_llm_recebe_mensagens():
    """AC-007: o LLM recebe mensagens (não só prompt string)."""
    llm = FakeLLM()
    cp = MemorySaver()
    g = _build_graph(checkpointer=cp, llm=llm)
    g.invoke(
        {"question": "oi", "chat_id": 1, "messages": [HumanMessage(content="oi")]},
        config={"configurable": {"thread_id": 1}},
    )
    assert llm.calls, "LLM não foi chamado"
    msgs = llm.calls[0]
    assert isinstance(msgs, list)
    assert all(isinstance(m, HumanMessage) for m in msgs)
    # o prompt (que contém a pergunta) chega como HumanMessage
    assert any("oi" in m.content for m in msgs)