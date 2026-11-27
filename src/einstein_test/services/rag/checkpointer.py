"""Checkpointer de memória de conversa do RAG.

`build_postgres_checkpointer` constrói um `PostgresSaver` (memória persistente
de conversa) conectado ao Postgres da app, com `thread_id = chat_id`. O
`AsyncPostgresSaver` (variante async, usada pelo endpoint SSE) é criado via
`from_conn_string` dentro de um contexto async. O `MemorySaver` (em memória) é
usado nos testes determinísticos.
"""

from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import ConnectionPool
from sqlalchemy.engine import make_url

from einstein_test.settings import get_settings


def _conninfo() -> str:
    url = make_url(get_settings().database_url)
    if url.drivername not in ("postgresql", "postgresql+psycopg"):
        raise ValueError(f"checkpointer exige Postgres; driver atual: {url.drivername}")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def _pool() -> ConnectionPool:
    return ConnectionPool(
        conninfo=_conninfo(),
        open=True,
        min_size=0,
        max_size=5,
        kwargs={"autocommit": True},
    )


@lru_cache(maxsize=1)
def build_postgres_checkpointer() -> PostgresSaver:
    """PostgresSaver síncrono pronto (setup das tabelas de checkpoint)."""
    cp = PostgresSaver(_pool())
    cp.setup()
    return cp


def build_async_postgres_checkpointer() -> AsyncPostgresSaver:
    """AsyncPostgresSaver construído dentro do event loop (para astream).

    Deve ser usado como: `async with AsyncPostgresSaver.from_conn_string(...)`.
    Este helper retorna o context manager; o caller faz `setup()`.
    """
    return AsyncPostgresSaver.from_conn_string(_conninfo())


def thread_config(chat_id: int) -> dict:
    """Config do graph com thread_id = chat_id (memória por chat)."""
    return {"configurable": {"thread_id": str(chat_id)}}


__all__ = [
    "build_postgres_checkpointer",
    "build_async_postgres_checkpointer",
    "thread_config",
]