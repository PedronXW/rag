"""G-004 — Testes do multi-query com subgraphs (Send) e fusão."""

import pytest
from langgraph.types import Send


class _Chunk:
    def __init__(self, id, file_id, content, score=0.5):
        self.id = id
        self.file_id = file_id
        self.content = content
        self.score = score


class _RewriteLLM:
    """LLM fake: rewrite devolve N variações; generate responde."""

    def __init__(self, variants=None):
        self.variants = variants or ["variação 1", "variação 2", "variação 3"]
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        # rewrite recebe [SystemMessage, HumanMessage(pergunta)]; generate [System, prompt]
        if len(messages) >= 2 and "variação" not in " ".join(getattr(m, "content", "") for m in messages):
            yield "\n".join(self.variants)
        else:
            yield "resposta multi-query"

    async def astream(self, messages):
        for t in self.stream(messages):
            yield t


def _fakeretriever(chunks_by_query=None):
    chunks_by_query = chunks_by_query or {
        "q1": [_Chunk(1, 1, "chunk A", 0.9)],
        "q2": [_Chunk(2, 1, "chunk B", 0.7)],
        "q3": [_Chunk(1, 1, "chunk A", 0.9), _Chunk(3, 1, "chunk C", 0.5)],
    }

    class _R:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            # usa o primeiro elemento do embedding como chave de query
            key = f"q{int(round(embedding[0])) + 1}"
            return chunks_by_query.get(key, [])

    return _R()


class _FakeEmbedder:
    def embed(self, texts):
        return [[float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(len(texts))]


def test_subgraph_retrieve_existe():
    """AC-001: subgraph de retrieval compilado."""
    from einstein_test.services.rag.multiquery import build_retrieval_subgraph

    sub = build_retrieval_subgraph()
    assert sub is not None
    assert "retrieve" in sub.get_graph().nodes


def test_rewrite_gera_variacoes():
    """AC-002: rewrite gera N variações quando queries > 1."""
    from einstein_test.services.rag.multiquery import _make_rewrite

    rewrite = _make_rewrite(_RewriteLLM())
    out = rewrite({"question": "pergunta", "queries": 3})
    assert len(out["query_variants"]) == 3
    assert all("variação" in v for v in out["query_variants"])


def test_send_gera_retrieves():
    """AC-003: Send dispara um retrieve por variação."""
    from einstein_test.services.rag.multiquery import _send_retrievals

    sends = _send_retrievals({"query_variants": ["v1", "v2", "v3"]})
    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)


def test_fusao_deduplica_e_ranqueia():
    """AC-005: fusão deduplica por chunk_id e ranqueia por score máximo."""
    from einstein_test.services.rag.multiquery import fuse_chunks

    chunks = [
        {"chunk_id": 1, "content": "a", "score": 0.5},
        {"chunk_id": 1, "content": "a", "score": 0.9},
        {"chunk_id": 2, "content": "b", "score": 0.7},
    ]
    fused = fuse_chunks(chunks)
    assert len(fused) == 2
    assert fused[0]["chunk_id"] == 1
    assert fused[0]["score"] == 0.9
    assert fused[1]["chunk_id"] == 2


def test_graph_acumula_chunks_paralelos():
    """AC-004: o graph acumula chunks das variações via reducer."""
    from einstein_test.services.rag.multiquery import build_multiquery_graph

    g = build_multiquery_graph(
        llm=_RewriteLLM(),
        retriever=_fakeretriever(),
        embedder=_FakeEmbedder(),
        chat_id=1,
    )
    out = g.invoke({"question": "pergunta", "queries": 3})
    assert "sources" in out or "chunks" in out
    assert out.get("answer")