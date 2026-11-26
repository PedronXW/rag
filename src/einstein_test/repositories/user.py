"""Repositório de User: Protocol + implementação SQLAlchemy.

O repositório persiste o hash já pronto — a política criptográfica (bcrypt)
vive em `einstein_test.security`.
"""

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from einstein_test.models.user import User


class UserRepository(Protocol):
    def create(
        self, email: str, password_hash: str, *, consent_granted: bool = False, consent_at=None
    ) -> User: ...
    def get_by_email(self, email: str) -> User | None: ...
    def delete(self, user_id: int) -> None: ...


class UserRepositorySQLAlchemy:
    """Implementação SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self, email: str, password_hash: str, *, consent_granted: bool = False, consent_at=None
    ) -> User:
        user = User(
            email=email,
            password_hash=password_hash,
            consent_granted=consent_granted,
            consent_at=consent_at,
        )
        self.session.add(user)
        self.session.flush()
        self.session.refresh(user)
        return user

    def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        return self.session.scalar(stmt)

    def delete(self, user_id: int) -> None:
        # bulk delete evita que a ORM tente anular FKs (chats.user_id) ao
        # desvincular relacionamentos já removidos (LGPD/erasure).
        self.session.query(User).filter(User.id == user_id).delete()
        self.session.flush()