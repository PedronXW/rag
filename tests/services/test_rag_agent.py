"""G-002 — Testes do agente react com tools (ToolNode + Command)."""

import pytest
from langchain_core.messages import AIMessage, ToolMessage


class _Bound:
    def __init__(self, script):
        self.script = script

    def invoke(self, messages):
        return self.script.pop(0) if self.script else AIMessage(content="fim")


class FakeToolLLM:
    """LLM fake com bind_tools().invoke() determinístico (emite tool_calls)."""

    def __init__(self, script=None):
        self.script = script or [
            AIMessage(content="", tool_calls=[{"name": "search_chunks", "args": {"question": "q"}, "id": "t1"}]),
            AIMessage(content="resposta do agente"),
        ]
        self.calls: list[list] = []

    def bind_tools(self, tools):
        self.calls.append(tools)
        return _Bound(self.script)


class _Chunk:
    def __init__(self, id, file_id, content):
        self.id = id
        self.file_id = file_id
        self.content = content


def _fakeretriever(chunks=None, file_ids=None):
    chunks = chunks or [_Chunk(1, 1, "conteudo do arquivo 1")]
    file_ids = file_ids or [1]

    class _R:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return chunks if chat_id in (1, 10) else []

        def get_chunks(self, file_id):
            return chunks if file_id in file_ids else []

        def get_chunks_for_chat(self, chat_id, file_id):
            return chunks if file_id in file_ids else []

    return _R()


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


def _build_agent(llm=None, retriever=None, chat_id=1):
    from einstein_test.services.rag.agent import build_agent_graph

    return build_agent_graph(
        llm=llm or FakeToolLLM(),
        retriever=retriever or _fakeretriever(),
        embedder=_FakeEmbedder(),
        chat_id=chat_id,
    )


def test_tools_registradas_com_atool():
    """AC-001: existem tools search_chunks e fetch_document via @tool."""
    from einstein_test.services.rag.agent import build_tools

    tools = build_tools(_fakeretriever(), _FakeEmbedder(), chat_id=1)
    names = {t.name for t in tools}
    assert "search_chunks" in names
    assert "fetch_document" in names
    search = next(t for t in tools if t.name == "search_chunks")
    fetch = next(t for t in tools if t.name == "fetch_document")
    assert "question" in search.args
    assert "file_id" in fetch.args


def test_search_chunks_filtra_por_chat():
    """AC-002: search_chunks busca chunks do chat via retriever."""
    from einstein_test.services.rag.agent import build_tools

    tools = build_tools(_fakeretriever(), _FakeEmbedder(), chat_id=1)
    search = next(t for t in tools if t.name == "search_chunks")
    out = search.invoke({"question": "q"})
    assert "conteudo do arquivo 1" in out


def test_fetch_document_multi_tenant():
    """AC-002/007: fetch_document retorna conteúdo apenas se o arquivo é do chat."""
    from einstein_test.services.rag.agent import build_tools

    retriever = _fakeretriever(file_ids=[1])
    tools = build_tools(retriever, _FakeEmbedder(), chat_id=1)
    fetch = next(t for t in tools if t.name == "fetch_document")
    out = fetch.invoke({"file_id": 1})
    assert "conteudo" in out
    # arquivo de outro chat (file_id=999) não é do chat
    out2 = fetch.invoke({"file_id": 999})
    assert "não encontrado" in out2 or "não pertence" in out2


def test_agent_loop_de_tools():
    """AC-003: o agente roda loop de tools (ToolMessage volta) e finaliza."""
    from einstein_test.services.rag.agent import build_agent_graph

    g = build_agent_graph(
        llm=FakeToolLLM(),
        retriever=_fakeretriever(),
        embedder=_FakeEmbedder(),
        chat_id=1,
    )
    out = g.invoke({"question": "q"})
    msgs = out["messages"]
    assert any(isinstance(m, ToolMessage) for m in msgs), "ToolMessage presente"
    assert out["answer"]


def test_agent_finaliza_com_answer():
    """AC-004: o agente finaliza com answer."""
    from einstein_test.services.rag.agent import build_agent_graph

    g = build_agent_graph(
        llm=FakeToolLLM(),
        retriever=_fakeretriever(),
        embedder=_FakeEmbedder(),
        chat_id=1,
    )
    out = g.invoke({"question": "q"})
    assert out["answer"]