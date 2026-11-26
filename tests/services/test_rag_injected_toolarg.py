"""I-001 — Testes do InjectedToolArg nas tools do RAG."""



class _Chunk:
    def __init__(self, id, file_id, content):
        self.id = id
        self.file_id = file_id
        self.content = content


class _Ret:
    def __init__(self, chat_ok=(1,)):
        self.chat_ok = chat_ok

    def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
        if chat_id in self.chat_ok:
            return [_Chunk(1, 1, "conteudo do chat")]
        return []

    def get_chunks_for_chat(self, chat_id, file_id):
        if chat_id in self.chat_ok:
            return [_Chunk(1, 1, "conteudo do chat")]
        return []


class _Emb:
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


def test_tools_usam_injected_tool_arg():
    """AC-001: as tools declaram retriever/embedder/chat_id com InjectedToolArg."""
    import inspect

    from einstein_test.services.rag.agent import build_tools

    tools = build_tools(_Ret(), _Emb(), chat_id=1)
    search = next(t for t in tools if t.name == "search_chunks")
    # o retriever/embedder/chat_id devem estar injetados (não na assinatura pública)
    src = inspect.getsource(search.func)
    assert "InjectedToolArg" in src


def test_search_chunks_injeta_retriever():
    """AC-002: o runtime injeta retriever/embedder/chat_id automaticamente."""
    from einstein_test.services.rag.agent import build_tools

    retriever = _Ret()
    tools = build_tools(retriever, _Emb(), chat_id=1)
    search = next(t for t in tools if t.name == "search_chunks")
    # InjectedToolArg fica fora dos args do modelo; invoke com kwargs injetados
    out = search.invoke({"question": "q", "retriever": retriever, "embedder": _Emb(), "chat_id": 1})
    assert "conteudo do chat" in out


def test_agente_multi_tenant_preservado():
    """AC-003: o agente respeita o multi-tenant (só dados do chat)."""
    from einstein_test.services.rag.agent import build_tools

    retriever = _Ret(chat_ok=(1,))
    tools = build_tools(retriever, _Emb(), chat_id=2)  # agente do chat 2
    search = next(t for t in tools if t.name == "search_chunks")
    out = search.invoke({"question": "q", "retriever": retriever, "embedder": _Emb(), "chat_id": 2})
    assert "Nenhum chunk" in out or out == ""  # chat 2 não tem chunks


def test_tools_tem_arg_de_retriever_oculto():
    """AC-001: retriever/embedder/chat_id não aparecem no schema público."""
    from einstein_test.services.rag.agent import build_tools

    tools = build_tools(_Ret(), _Emb(), chat_id=1)
    search = next(t for t in tools if t.name == "search_chunks")
    # args públicos só têm 'question' (InjectedToolArg fica oculto)
    assert "question" in search.args
    assert "retriever" not in search.args