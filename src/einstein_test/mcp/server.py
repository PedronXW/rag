"""Servidor MCP (Model Context Protocol) expondo as tools do RAG.

Usa o SDK Python `mcp` (MCPServer) para expor as capacidades do app como tools
MCP: busca vetorial (`rag_search`), leitura de documento (`fetch_document`) e
memória de longo prazo (`get_memory`/`put_memory`). Multi-tenant por
`chat_id`/`user_id` passados como argumento (cliente de confiança).

Transporte: `stdio` por padrão; `--transport http` para streamable HTTP.

Rode com:
    uv run python -m einstein_test.mcp
"""

from __future__ import annotations

import argparse

from sqlalchemy.orm import Session

from einstein_test.db.engine import create_engine_from_config

_engine = None
_override_engine = None


def set_engine(engine) -> None:
    """Injeta engine (testes); chama com None para voltar ao configurado."""
    global _override_engine
    _override_engine = engine


def _session() -> Session:
    global _engine
    if _override_engine is not None:
        return Session(_override_engine)
    if _engine is None:
        _engine = create_engine_from_config()
    return Session(_engine)


def _retriever(s):
    from einstein_test.vector import VectorRepositorySQLAlchemy

    return VectorRepositorySQLAlchemy(s)


def _embedder():
    from einstein_test.vector import get_embedder

    return get_embedder()


def build_server():
    """Constrói o MCPServer com as tools do RAG."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("einstein-rag")

    @server.tool()
    def rag_search(chat_id: int, question: str) -> str:
        """Busca os chunks mais relevantes do chat para a pergunta (RAG).

        Retorna o conteúdo dos chunks com score de similaridade.
        """
        from einstein_test.services.rag.retrieval import retrieve_chunks

        with _session() as s:
            chunks = retrieve_chunks(question, _retriever(s), _embedder(), chat_id)
        if not chunks:
            return "Nenhum chunk relevante encontrado no chat."
        return "\n\n".join(
            f"[chunk {c['chunk_id']}] (doc {c['document_id']}) {c['content']}"
            for c in chunks
        )

    @server.tool()
    def fetch_document(chat_id: int, file_id: int) -> str:
        """Retorna o conteúdo de um arquivo do chat (multi-tenant por chat_id)."""
        from einstein_test.services.rag.retrieval import chunk_to_dict  # noqa: F401

        with _session() as s:
            chunks = _retriever(s).get_chunks_for_chat(chat_id, file_id) or []
        if not chunks:
            return f"Arquivo {file_id} não encontrado neste chat."
        return "\n".join(getattr(c, "content", str(c)) for c in chunks)

    @server.tool()
    def get_memory(user_id: int, key: str) -> str:
        """Retorna um valor de memória de longo prazo do usuário."""
        from einstein_test.services.rag.longterm import get_memory as _get

        value = _get(user_id, key)
        if value is None:
            return "Sem memória para esta chave."
        import json

        return json.dumps(value, ensure_ascii=False)

    @server.tool()
    def put_memory(user_id: int, key: str, value: dict) -> str:
        """Grava um valor de memória de longo prazo do usuário."""
        from einstein_test.services.rag.longterm import put_memory as _put

        _put(user_id, key, value)
        return "ok"

    @server.tool()
    def list_chats(user_id: int) -> str:
        """Lista os chats do usuário."""
        import json

        from einstein_test.repositories.chat import ChatRepositorySQLAlchemy
        from einstein_test.services.chat import ChatService

        with _session() as s:
            chats = ChatService(ChatRepositorySQLAlchemy(s)).list_chats(user_id)
        return json.dumps(
            [{"id": c.id, "name": c.name} for c in chats], ensure_ascii=False
        )

    @server.tool()
    def create_chat(user_id: int, name: str) -> str:
        """Cria um novo chat para o usuário."""
        import json

        from einstein_test.repositories.chat import ChatRepositorySQLAlchemy
        from einstein_test.services.chat import ChatService

        with _session() as s:
            chat = ChatService(ChatRepositorySQLAlchemy(s)).create_chat(
                user_id=user_id, name=name
            )
        return json.dumps({"id": chat.id, "name": chat.name})

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="einstein-mcp")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    args = parser.parse_args(argv)
    server = build_server()
    if args.transport == "http":
        import uvicorn

        app = server.streamable_http_app()
        uvicorn.run(app, host="127.0.0.1", port=8001)
        return 0
    import asyncio

    asyncio.run(server.run_stdio_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())