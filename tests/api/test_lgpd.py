"""F-061 — LGPD: consentimento, DSR (export/delete) e gate de consentimento."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from einstein_test.db.base import Base
from einstein_test.main import create_app
from einstein_test.models import File, Message, User
from tests.conftest import register_and_login


@pytest.fixture
def client():
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
        yield c, engine
    engine.dispose()


def test_register_com_consent(client):
    c, engine = client
    r = c.post("/users", json={"email": "lgpd@b.com", "password": "senha123", "consent": True})
    assert r.status_code == 201
    body = r.json()
    assert body["consent_granted"] is True
    assert body["consent_at"] is not None


def test_consent_revoke(client):
    c, engine = client
    h = register_and_login(c, email="lgpd2@b.com")
    r = c.patch("/users/me/consent", json={"consent": True}, headers=h)
    assert r.json()["consent_granted"] is True
    r = c.patch("/users/me/consent", json={"consent": False}, headers=h)
    assert r.json()["consent_granted"] is False


def test_export_data_inclui_chats_e_mensagens(client):
    c, engine = client
    h = register_and_login(c, email="lgpd3@b.com")
    chat = c.post("/chats", json={"name": "chat"}, headers=h).json()
    c.post(
        f"/chats/{chat['id']}/messages",
        json={"content": "ola"},
        headers=h,
    )
    data = c.get("/users/me/data", headers=h).json()
    assert data["profile"]["email"] == "lgpd3@b.com"
    assert len(data["chats"]) == 1
    assert data["chats"][0]["name"] == "chat"


def test_delete_account_apaga_dados(client):
    c, engine = client
    h = register_and_login(c, email="lgpd4@b.com")
    chat = c.post("/chats", json={"name": "chat"}, headers=h).json()
    c.post(
        f"/chats/{chat['id']}/messages",
        json={"content": "ola"},
        headers=h,
    )
    r = c.delete("/users/me", headers=h)
    assert r.status_code == 200
    with Session(engine) as s:
        assert s.query(User).filter_by(email="lgpd4@b.com").count() == 0
        assert s.query(Message).count() == 0


def test_require_consent_bloqueia_rag(client, monkeypatch):
    """REQUIRE_CONSENT=true: /messages exige consentimento (SSE notice)."""
    from einstein_test.settings import _build_settings

    monkeypatch.setenv("REQUIRE_CONSENT", "true")
    _build_settings.cache_clear()
    c, engine = client
    h = register_and_login(c, email="lgpd5@b.com")
    chat = c.post("/chats", json={"name": "chat"}, headers=h).json()
    # seed file ready para chegar até o ponto do gate
    with Session(engine) as s:
        f = File(chat_id=chat["id"], name="doc.txt", url="s3://b/k", type="text/plain", size=3, status="ready", chunk_ids=[])
        s.add(f)
        s.commit()
    resp = c.post(
        f"/chats/{chat['id']}/messages",
        json={"content": "qual a mensalidade?"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert "consent_required" in resp.text
    # concede consentimento -> agora passa
    c.patch("/users/me/consent", json={"consent": True}, headers=h)
    resp2 = c.post(
        f"/chats/{chat['id']}/messages",
        json={"content": "qual a mensalidade?"},
        headers={**h, "Accept": "text/event-stream"},
    )
    assert "consent_required" not in resp2.text