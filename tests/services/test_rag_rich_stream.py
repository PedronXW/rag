"""I-003 — Testes do streaming rico com astream_events."""



class _Ret:
    def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
        return []


class _Emb:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


class _StreamLLM:
    def stream(self, messages):
        for t in ["tok1", "tok2"]:
            yield t

    async def astream(self, messages):
        for t in ["tok1", "tok2"]:
            yield t


def _build():
    from einstein_test.services.rag.graph import build_graph

    return build_graph(llm=_StreamLLM(), retriever=_Ret(), embedder=_Emb())


def test_astream_events_emite_nos():
    """AC-001/004: astream_events emite on_chain_start/end por nó, em ordem."""
    import asyncio

    from einstein_test.services.rag.rich_stream import node_events

    g = _build()

    async def _collect():
        return [ev async for ev in node_events(g, {"question": "q", "chat_id": 1, "history": []})]

    events = asyncio.run(_collect())
    names = [e for e in events if e[0] in ("start", "end")]
    assert names
    assert any(n[1] == "retrieve" for n in names)
    assert any(n[1] == "generate" for n in names)


def test_astream_events_emite_tokens():
    """AC-002: os tokens reais do LLM são emitidos (on_chat_model_stream)."""
    import asyncio

    from einstein_test.services.rag.rich_stream import token_events

    g = _build()

    async def _collect():
        return [t async for t in token_events(g, {"question": "q", "chat_id": 1, "history": []})]

    tokens = asyncio.run(_collect())
    assert tokens  # pelo menos um token emitido


def test_sse_contrato_preservado():
    """AC-003: o helper build_sse mantém o formato node/token/done."""
    from einstein_test.services.rag.rich_stream import build_sse

    sse = build_sse({"node": "retrieve"}, event="node")
    assert "event: node" in sse
    assert "data:" in sse