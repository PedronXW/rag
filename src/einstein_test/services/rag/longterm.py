"""Memória de longo prazo do RAG via PostgresStore (H-002).

Fatos e preferências do usuário persistidos por namespace (`user_id`),
recuperados em novas conversas (memória entre chats). O store é injetável:
`InMemoryStore` nos testes, `PostgresStore` em produção.
"""

import os
from functools import lru_cache
from typing import Any

from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore


def _default_store():
    """Store padrão: PostgresStore em produção, InMemoryStore no dev/teste."""
    if os.getenv("RAG_MEMORY_BACKEND", "memory").lower() == "postgres":
        return build_postgres_store()
    return InMemoryStore()


_store: Any | None = None


def get_store() -> Any:
    global _store
    if _store is None:
        _store = _default_store()
    return _store


def set_memory_store(store: Any) -> None:
    global _store
    _store = store


def reset_memory_store() -> None:
    global _store
    _store = None


@lru_cache(maxsize=1)
def _pool():
    """ConnectionPool psycopg para o PostgresStore (paridade com checkpointer)."""
    from psycopg_pool import ConnectionPool
    from sqlalchemy.engine import make_url

    from einstein_test.settings import get_settings

    url = make_url(get_settings().database_url)
    conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
    return ConnectionPool(
        conninfo=conninfo,
        open=True,
        min_size=0,
        max_size=5,
        kwargs={"autocommit": True},
    )


@lru_cache(maxsize=1)
def build_postgres_store():
    """PostgresStore (LangGraph) conectado ao Postgres da app, com setup()."""
    store = PostgresStore(_pool())
    store.setup()
    return store


def _ns(user_id: int) -> tuple[str, ...]:
    return ("user", str(user_id))


def put_memory(user_id: int, key: str, value: dict[str, Any]) -> None:
    """Grava um fato/preferência do usuário na memória de longo prazo."""
    get_store().put(_ns(user_id), key, value)


def get_memory(user_id: int, key: str) -> Any | None:
    """Recupera um fato/preferência do usuário."""
    return get_store().get(_ns(user_id), key)


def list_memory(user_id: int) -> list[dict]:
    """Lista todos os itens de memória do usuário (até 100)."""
    store = get_store()
    items = store.search(_ns(user_id), limit=100)
    return [
        {"key": item.key, "value": dict(item.value) if item.value else {}}
        for item in items
    ]


__all__ = [
    "put_memory",
    "get_memory",
    "list_memory",
    "get_store",
    "set_memory_store",
    "reset_memory_store",
    "build_postgres_store",
]