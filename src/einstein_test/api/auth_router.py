"""Endpoints de auth: POST /users, POST /auth/login, GET/PATCH /users/me."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from einstein_test.api.auth import create_access_token, get_current_user
from einstein_test.api.dependencies import get_session
from einstein_test.models import User
from einstein_test.repositories.user import UserRepositorySQLAlchemy
from einstein_test.schemas.user import (
    ConsentUpdate,
    TokenRead,
    UserCreate,
    UserLogin,
    UserRead,
)
from einstein_test.services.auth import (
    AuthService,
    EmailAlreadyRegistered,
    InvalidCredentials,
)
from einstein_test.services.user import UserService

router = APIRouter(tags=["auth"])


class UserModelUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, max_length=127)


def _service(session: Session) -> AuthService:
    return AuthService(UserRepositorySQLAlchemy(session))


def _user_service(session: Session) -> UserService:
    return UserService(UserRepositorySQLAlchemy(session))


@router.post("/users", response_model=UserRead, status_code=201)
def register(
    payload: UserCreate, session: Session = Depends(get_session)
) -> UserRead:
    service = _service(session)
    try:
        user = service.register(
            email=payload.email, password=payload.password, consent=payload.consent
        )
    except EmailAlreadyRegistered:
        raise HTTPException(
            status_code=409, detail="email já cadastrado"
        ) from None
    return UserRead.model_validate(user)


@router.patch("/users/me/consent", response_model=UserRead)
def set_consent(
    payload: ConsentUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> UserRead:
    _user_service(session).set_consent(current_user, payload.consent)
    session.flush()
    return UserRead.model_validate(current_user)


@router.get("/users/me/data")
def export_user_data(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    """DSR de exportação: perfil + chats + mensagens + metadados de arquivos."""
    from einstein_test.repositories.chat import ChatRepositorySQLAlchemy
    from einstein_test.repositories.file import FileRepositorySQLAlchemy
    from einstein_test.repositories.message import MessageRepositorySQLAlchemy
    from einstein_test.services.chat import ChatService
    from einstein_test.services.message import MessageService

    chat_svc = ChatService(ChatRepositorySQLAlchemy(session))
    msg_svc = MessageService(MessageRepositorySQLAlchemy(session))
    file_repo = FileRepositorySQLAlchemy(session)
    data = {
        "profile": UserRead.model_validate(current_user).model_dump(mode="json"),
        "consent_granted": current_user.consent_granted,
        "consent_at": current_user.consent_at.isoformat() if current_user.consent_at else None,
        "chats": [],
    }
    for chat in chat_svc.list_chats(current_user.id):
        files = file_repo.list(chat_id=chat.id) or []
        data["chats"].append(
            {
                "id": chat.id,
                "name": chat.name,
                "description": chat.description,
                "created_at": chat.created_at.isoformat() if chat.created_at else None,
                "files": [
                    {"id": f.id, "name": f.name, "type": f.type, "size": f.size, "status": f.status}
                    for f in files
                ],
                "messages": [
                    {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
                    for m in msg_svc.list_messages(chat.id)
                ],
            }
        )
    return data


@router.delete("/users/me", status_code=200)
def delete_account(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    """DSR de apagamento: remove usuário + chats + mensagens + arquivos (LGPD).

    Apaga também os objetos do storage (MinIO) via `FileService.delete_file`.
    """
    from einstein_test.models import Chat, File, Message
    from einstein_test.repositories.chat import ChatRepositorySQLAlchemy
    from einstein_test.repositories.file import FileRepositorySQLAlchemy
    from einstein_test.services.chat import ChatService
    from einstein_test.services.file import FileService
    from einstein_test.vector import VectorRepositorySQLAlchemy

    file_svc = FileService(
        FileRepositorySQLAlchemy(session),
        vector_repository=VectorRepositorySQLAlchemy(session),
    )
    file_repo = FileRepositorySQLAlchemy(session)
    chat_svc = ChatService(ChatRepositorySQLAlchemy(session))
    for chat in chat_svc.list_chats(current_user.id):
        # storage + chunks (hard) via FileService; em seguida hard-delete linhas
        for file_ in file_repo.list(chat_id=chat.id) or []:
            file_svc.delete_file(file_.id)
        session.query(Message).filter(Message.chat_id == chat.id).delete(
            synchronize_session=False
        )
        session.query(File).filter(File.chat_id == chat.id).delete(
            synchronize_session=False
        )
        session.query(Chat).filter(Chat.id == chat.id).delete(
            synchronize_session=False
        )
    _user_service(session).delete_account(current_user.id)
    session.flush()
    return {"status": "ok"}


@router.post("/auth/login", response_model=TokenRead)
def login(payload: UserLogin, session: Session = Depends(get_session)) -> TokenRead:
    service = _service(session)
    try:
        user = service.authenticate(email=payload.email, password=payload.password)
    except InvalidCredentials:
        raise HTTPException(
            status_code=401, detail="usuário ou senha inválidos"
        ) from None
    token = create_access_token(subject=user.email)
    return TokenRead(access_token=token, token_type="bearer")


@router.get("/users/me", response_model=UserRead)
def get_me(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)


@router.patch("/users/me", response_model=UserRead)
def update_me(
    payload: UserModelUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> UserRead:
    try:
        _user_service(session).set_model(current_user, payload.model)
    except ValueError:
        raise HTTPException(
            status_code=422, detail="modelo não suportado"
        ) from None
    # UoW (F-018): `get_session` commita ao final do request
    session.flush()
    return UserRead.model_validate(current_user)