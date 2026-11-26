"""
Fixtures compartilhadas de testes.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from einstein_test.db.base import Base
from einstein_test.main import create_app

# Fila capturada dos uploads (mock global de enqueue_vectorization).
_queued_file_ids: list[int] = []


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    """Configuração obrigatória mínima para o fails-fast do create_app passar."""
    monkeypatch.setenv("JWT_SECRET", "segredo-dos-testes-0123456789abcdefghijklmnop")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    monkeypatch.setenv("ALLOWED_MODELS", "azure/gpt-5-mini,azure/gpt-6-luna")
    # banco NÃO-postgres por padrão nos testes: _is_postgres()/checkpointer não
    # tentam abrir pool de Postgres (determinístico no CI, sem serviço de banco).
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    # observabilidade Langfuse desligada por padrão nos testes (determinístico)
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    # guardrails/refiners/LGPD desligados por padrão nos testes — os testes que
    # testam essas features os habilitam explicitamente (sem herdar do .env).
    monkeypatch.setenv("GUARDRAILS_ENABLED", "false")
    monkeypatch.setenv("REFINERS_ENABLED", "false")
    monkeypatch.setenv("REQUIRE_CONSENT", "false")
    monkeypatch.setenv("BLOCKED_TOPICS", "")


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch):
    """Limpa o cache de settings após cada teste, DEPOIS do restore do env.

    Depende de `monkeypatch` para que seu teardown rode antes do undo do
    monkeypatch — assim o cache é esvaziado quando o env já voltou ao valor
    original, e a próxima `get_settings()` reconstrói com o env correto.
    """
    from einstein_test.settings import _build_settings

    _build_settings.cache_clear()
    yield
    _build_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Zera o rate limiter em memória entre testes (evita contagem acumulada)."""
    from einstein_test.middleware.rate_limit import RateLimiter

    RateLimiter.reset()
    yield


@pytest.fixture(autouse=True)
def _mock_enqueue(monkeypatch):
    """Mocka a fila de vetorização em todos os testes (sem Redis/Celery).

    Substitui `api.files.enqueue_vectorization` por uma função que coleta os
    file_ids na lista global `_queued_file_ids` — evita .delay() real → Redis.
    """
    global _queued_file_ids
    _queued_file_ids = []
    import einstein_test.api.files as files_api

    monkeypatch.setattr(files_api, "enqueue_vectorization", _queued_file_ids.append)


@pytest.fixture(autouse=True)
def _storage_fake():
    """Usa storage em memória (sem MinIO) em todos os testes.

    O lifespan do create_app chama `ensure_bucket` no startup; sem este fake,
    `get_storage()` criaria um MinioStorage real e cada teste tentaria conectar
    no MinIO. Testes que precisam de storage específico fazem `set_storage`
    explicitamente na própria fixture.
    """
    from einstein_test.storage import InMemoryStorage, set_storage

    set_storage(InMemoryStorage())


@pytest.fixture
def enqueue_spy() -> list[int]:
    """Lista dos file_ids enfileirados nos uploads (mock da fila)."""
    return _queued_file_ids


@pytest.fixture
def app_client():
    """Client do app com SQLite :memory: e headers de usuário autenticado."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    from einstein_test.api.dependencies import configure_session_factory

    configure_session_factory(engine)

    app = create_app()
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        headers = register_and_login(c, email="user@b.com")
        yield c, headers
    engine.dispose()


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    def _factory() -> Session:
        return Session(engine)

    yield _factory
    engine.dispose()


def register_and_login(client: TestClient, email: str, password: str = "senha123") -> dict:
    """Registra e autentica um usuário; retorna headers Authorization."""
    assert client.post(
        "/users", json={"email": email, "password": password}
    ).status_code in (201, 409)
    resp = client.post(
        "/auth/login", json={"email": email, "password": password}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def seed_ready_file(engine, chat_id: int, name: str = "nota.txt") -> int:
    """Insere um File com status=ready direto no banco (helper de testes).

    Útil para testes de mensagens RAG que precisam de ao menos um arquivo
    vetorizado (`ready`) para o fluxo normal do LLM rodar.
    """
    from einstein_test.models.file import File

    with Session(engine) as s:
        f = File(
            chat_id=chat_id, name=name, url=f"s3://einstein/{name}",
            size=5, status="ready",
        )
        s.add(f)
        s.commit()
        return f.id