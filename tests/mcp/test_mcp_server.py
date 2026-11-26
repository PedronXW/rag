"""F-062 — Servidor MCP: tools do RAG registradas e funcionais."""

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from einstein_test.db.base import Base
from einstein_test.models import File, FileChunk


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    from einstein_test.vector import FakeEmbedder, set_embedder

    set_embedder(FakeEmbedder())
    yield eng
    eng.dispose()


def _seed(engine):
    from sqlalchemy.orm import Session

    from einstein_test.repositories.chat import ChatRepositorySQLAlchemy
    from einstein_test.repositories.user import UserRepositorySQLAlchemy
    from einstein_test.services.auth import AuthService
    from einstein_test.services.chat import ChatService
    from einstein_test.vector import FakeEmbedder

    with Session(engine) as s:
        user = AuthService(UserRepositorySQLAlchemy(s)).register("mcp@b.com", "senha123")
        chat = ChatService(ChatRepositorySQLAlchemy(s)).create_chat(
            user_id=user.id, name="mcp-chat"
        )
        file_ = File(
            chat_id=chat.id, name="doc.txt", url="s3://b/k", type="text/plain",
            size=3, status="ready", chunk_ids=[1],
        )
        s.add(file_)
        s.flush()
        emb = FakeEmbedder().embed(["O Einstein foi fundado em 1997"])[0]
        s.add(
            FileChunk(
                file_id=file_.id, chunk_index=0,
                content="O Einstein foi fundado em 1997.", embedding=emb,
            )
        )
        s.commit()
        return user.id, chat.id, file_.id


def test_server_lista_tools():
    from einstein_test.mcp.server import build_server

    server = build_server()
    names = [t.name for t in asyncio.run(server.list_tools())]
    for expected in ("rag_search", "fetch_document", "get_memory", "put_memory", "list_chats", "create_chat"):
        assert expected in names, f"tool {expected} ausente"


def test_rag_search_retorna_chunks(engine):
    from einstein_test.mcp import server as mcp_server

    mcp_server.set_engine(engine)
    try:
        srv = mcp_server.build_server()
        _, chat_id, _ = _seed(engine)
        result = asyncio.run(
            srv.call_tool("rag_search", {"chat_id": chat_id, "question": "quando foi fundado?"})
        )
        assert "1997" in str(result.content)
    finally:
        mcp_server.set_engine(None)


class _FakeMemoryStore:
    def __init__(self):
        self.data = {}

    def put(self, ns, key, value):
        self.data[(tuple(ns), key)] = value

    def get(self, ns, key):
        return self.data.get((tuple(ns), key))

    def search(self, ns, limit=100):
        out = []
        for (n, k), v in self.data.items():
            if tuple(n) == tuple(ns):
                out.append(type("I", (), {"key": k, "value": v}))
        return out[:limit]


def test_memory_e_chats(engine):
    from einstein_test.mcp import server as mcp_server
    from einstein_test.services.rag.longterm import set_memory_store

    mcp_server.set_engine(engine)
    set_memory_store(_FakeMemoryStore())
    try:
        srv = mcp_server.build_server()
        user_id, _, _ = _seed(engine)
        asyncio.run(srv.call_tool("put_memory", {"user_id": user_id, "key": "ctx", "value": {"a": 1}}))
        got = asyncio.run(srv.call_tool("get_memory", {"user_id": user_id, "key": "ctx"}))
        assert "1" in str(got.content)
        chats = asyncio.run(srv.call_tool("list_chats", {"user_id": user_id}))
        assert "mcp-chat" in str(chats.content)
        created = asyncio.run(srv.call_tool("create_chat", {"user_id": user_id, "name": "novo"}))
        assert "novo" in str(created.content)
    finally:
        mcp_server.set_engine(None)