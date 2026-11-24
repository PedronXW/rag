"""G-003 — Testes do human-in-the-loop (interrupt + resume) no graph RAG."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver


class _Chunk:
    def __init__(self, id, file_id, content):
        self.id = id
        self.file_id = file_id
        self.content = content


class _FakeRetriever:
    def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
        return [_Chunk(1, 1, "conteudo do arquivo")]

    def get_chunks_for_chat(self, chat_id, file_id):
        return [_Chunk(1, 1, "conteudo do arquivo")]


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


class _HitlLLM:
    """LLM fake (Protocol ChatLLM): 1ª = original, 2ª = ecoa feedback do prompt."""

    def __init__(self):
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        texts = [getattr(m, "content", str(m)) for m in messages]
        joined = " ".join(texts)
        if self.calls == 1:
            yield "resposta original"
        else:
            # ecoa o feedback presente no SystemMessage do re-prompt
            if "ficou vago" in joined:
                yield "resposta revisada (ficou vago)"
            else:
                yield "resposta revisada"

    async def astream(self, messages):
        for t in self.stream(messages):
            yield t


def _build_hitl(llm=None, checkpointer=None):
    from einstein_test.services.rag.hitl import build_hitl_graph

    return build_hitl_graph(
        llm=llm or _HitlLLM(),
        retriever=_FakeRetriever(),
        embedder=_FakeEmbedder(),
        checkpointer=checkpointer or MemorySaver(),
    )


def test_graph_hitl_tem_no_review():
    """AC-001: o graph HITL tem nó review (interrupt)."""
    g = _build_hitl()
    nodes = g.get_graph().nodes.keys()
    assert "review" in nodes
    assert "finalize" in nodes


def test_invoke_sem_resume_pausa():
    """AC-002: invocar com thread_id sem resume pausa e não entrega resposta final."""
    g = _build_hitl()
    cfg = {"configurable": {"thread_id": "h1"}}
    g.invoke({"question": "q", "chat_id": 1, "messages": [HumanMessage(content="q")]}, config=cfg)
    snap = g.get_state(cfg)
    assert snap.next
    assert snap.interrupts
    assert snap.interrupts[0].value["draft"] == "resposta original"


def test_resume_aprovado_entrega_resposta():
    """AC-003: resume approved=True entrega a resposta gerada."""
    from langgraph.types import Command

    g = _build_hitl()
    cfg = {"configurable": {"thread_id": "h2"}}
    g.invoke({"question": "q", "chat_id": 1, "messages": [HumanMessage(content="q")]}, config=cfg)
    out = g.invoke(Command(resume={"approved": True}), config=cfg)
    assert out["answer"] == "resposta original"


def test_resume_rejeitado_regenera_com_feedback():
    """AC-004: resume approved=False com feedback refaz a geração."""
    from langgraph.types import Command

    g = _build_hitl()
    cfg = {"configurable": {"thread_id": "h3"}}
    g.invoke({"question": "q", "chat_id": 1, "messages": [HumanMessage(content="q")]}, config=cfg)
    out = g.invoke(Command(resume={"approved": False, "feedback": "ficou vago"}), config=cfg)
    assert "ficou vago" in out["answer"]


def test_estado_pendente_persiste():
    """AC-005: o estado pendente persiste entre pausa e resume (mesmo thread)."""
    from langgraph.types import Command

    g = _build_hitl()
    cfg = {"configurable": {"thread_id": "h4"}}
    g.invoke({"question": "q", "chat_id": 1, "messages": [HumanMessage(content="q")]}, config=cfg)
    snap = g.get_state(cfg)
    assert snap.next
    out = g.invoke(Command(resume={"approved": True}), config=cfg)
    assert out["answer"] == "resposta original"