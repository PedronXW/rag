"""H-004 — Testes do create_react_agent e supervisor multi-agente."""

import pytest
from langchain_core.messages import AIMessage


class _ToolLLM:
    """LLM fake com bind_tools().invoke() e stream() determinísticos."""

    def __init__(self, script=None):
        self.script = script or [
            AIMessage(content="", tool_calls=[{"name": "search_chunks", "args": {"question": "q"}, "id": "t1"}]),
            AIMessage(content="resposta do react agent"),
        ]

    def stream(self, messages):
        text = " ".join(getattr(m, "content", str(m)) for m in messages)
        yield "resposta consolidada" if "contexto" in text.lower() else "retrieval"

    async def astream(self, messages):
        for t in self.stream(messages):
            yield t

    def bind_tools(self, tools):
        class _Bound:
            def invoke(self, messages):
                return self_outer.script.pop(0) if self_outer.script else AIMessage(content="fim")

        self_outer = self
        return _Bound()


class _Chunk:
    def __init__(self, id, file_id, content):
        self.id = id
        self.file_id = file_id
        self.content = content


class _Ret:
    def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
        return [_Chunk(1, 1, "conteudo")]

    def get_chunks_for_chat(self, chat_id, file_id):
        return [_Chunk(1, 1, "conteudo")]


class _Emb:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


def test_react_agent_compila():
    """AC-001: build_react_agent existe e usa create_react_agent (pre-built)."""
    from einstein_test.services.rag.react import build_react_agent, react_config

    # a factory é construível e expõe a API (Runnable real em produção via ChatOpenAIAdapter)
    assert callable(build_react_agent)
    cfg = react_config(chat_id=1)
    assert cfg["configurable"]["thread_id"] == "1"
    # o módulo referencia create_react_agent do langgraph-prebuilt
    import inspect

    src = inspect.getsource(build_react_agent)
    assert "create_react_agent" in src


def test_supervisor_roda():
    """AC-002/003: o supervisor roteia para sub-agentes e consolida."""
    from einstein_test.services.rag.supervisor import build_supervisor

    sup = build_supervisor(
        llm=_ToolLLM(),
        retriever=_Ret(),
        embedder=_Emb(),
        chat_id=1,
    )
    out = sup.invoke({"question": "pergunta", "chat_id": 1})
    assert out.get("answer")
    assert out.get("agents_used")


def test_supervisor_resposta_consolidada():
    """AC-003: o supervisor consolida a resposta final."""
    from einstein_test.services.rag.supervisor import build_supervisor

    sup = build_supervisor(
        llm=_ToolLLM(),
        retriever=_Ret(),
        embedder=_Emb(),
        chat_id=1,
    )
    out = sup.invoke({"question": "pergunta", "chat_id": 1})
    assert out["answer"]
    assert isinstance(out["agents_used"], list)

def test_supervisor_rota_general_sem_retrieval():
    """AC-002: pergunta 'general' pula o retrieval (rota direta a finalize)."""
    from einstein_test.services.rag.supervisor import build_supervisor

    class _GeneralLLM(_ToolLLM):
        def stream(self, messages):
            text = " ".join(getattr(m, "content", str(m)) for m in messages)
            yield "resposta geral" if "contexto" in text.lower() else "general"

    sup = build_supervisor(
        llm=_GeneralLLM(),
        retriever=_Ret(),
        embedder=_Emb(),
        chat_id=1,
    )
    out = sup.invoke({"question": "pergunta", "chat_id": 1})
    assert out["agents_used"] == ["general"]
    assert out["answer"]
