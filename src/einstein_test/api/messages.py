"""Endpoints REST de message."""

import json
from typing import Any
from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from einstein_test.api.auth import get_current_user
from einstein_test.api.dependencies import get_session
from einstein_test.logging_setup import current_request_id, get_logger
from einstein_test.models import User
from einstein_test.repositories.chat import ChatRepositorySQLAlchemy
from einstein_test.repositories.file import FileRepositorySQLAlchemy
from einstein_test.repositories.message import MessageRepositorySQLAlchemy
from einstein_test.repositories.user import UserRepositorySQLAlchemy
from einstein_test.schemas.message import HitlResume, MessageCreate, MessageRead
from einstein_test.schemas.page import Page
from einstein_test.services.chat import ChatNotFoundError, ChatService
from einstein_test.services.file import FileService
from einstein_test.services.message import (
    MAX_HISTORY,
    MessageNotFoundError,
    MessageService,
)
from einstein_test.services.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE,
    MAX_PAGE_SIZE,
    MIN_PAGE,
)
from einstein_test.services.rag import RagService, get_rag_service
from einstein_test.services.rag.graph import _refine_answer
from einstein_test.services.user import UserService
from einstein_test.vector import VectorRepositorySQLAlchemy

logger = get_logger("api.messages")

router = APIRouter(tags=["messages"])


def _chat_service(session: Session) -> ChatService:
    return ChatService(ChatRepositorySQLAlchemy(session))


def _message_service(session: Session) -> MessageService:
    return MessageService(MessageRepositorySQLAlchemy(session))


def _file_service(session: Session) -> FileService:
    return FileService(FileRepositorySQLAlchemy(session))


def _rag_service(session: Session, model: str | None = None) -> RagService:
    return get_rag_service(VectorRepositorySQLAlchemy(session), model=model)


def _rag_readiness_notice(session: Session, chat_id: int) -> str | None:
    """Retorna o notice SSE de readiness (no_ready_files/partial_files) ou None.

    Reutilizado pelas rotas RAG para não rodar com contexto vazio (evita
    alucinação) e por consistência com o /messages.
    """
    readiness = _file_service(session).rag_readiness(chat_id)
    if readiness.code == "no_ready_files":
        return _sse_event(
            {"code": "no_ready_files", "message": "nenhum arquivo pronto para responder — envie arquivos"},
            event="notice",
        )
    if readiness.code == "partial_files":
        return _sse_event(
            {"code": "partial_files", "message": "alguns arquivos ainda em processamento"},
            event="notice",
        )
    return None


def _is_postgres() -> bool:
    """True se o banco configurado é Postgres (habilita memória via checkpointer)."""
    from sqlalchemy.engine import make_url

    from einstein_test.settings import get_settings

    return make_url(get_settings().database_url).drivername in ("postgresql", "postgresql+psycopg")


def _ensure_chat_owned(session: Session, chat_id: int, user_id: int) -> None:
    _chat_service(session).get_chat(chat_id, user_id=user_id)


def _sse_event(data: str, *, event: str | None = None) -> str:
    """Formata um evento SSE (dados JSON-encoded; linhas normalizadas)."""
    encoded = json.dumps(data, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {encoded}\n\n"
    return f"data: {encoded}\n\n"


def _input_guardrail(content: str):
    """Roda o pipeline de guardrails de entrada (gated por GUARDRAILS_ENABLED).

    Retorna o `GuardrailResult`; com guardrails desabilitados (default) retorna
    sempre `allowed`. Usado pelas rotas RAG antes de chamar o LLM.
    """
    from einstein_test.guardrails.base import GuardAction
    from einstein_test.guardrails.input_guardrails import build_input_pipeline
    from einstein_test.settings import get_settings

    cfg = get_settings()
    pipeline = build_input_pipeline(
        enabled=cfg.guardrails_enabled, blocked_topics=cfg.blocked_topics
    )
    result = pipeline.run(content)
    if result.action == GuardAction.block or not result.allowed:
        return result
    return result


def _consent_required() -> bool:
    """True se REQUIRE_CONSENT=true e o usuário ainda não consentiu."""
    from einstein_test.settings import get_settings

    return get_settings().require_consent


def _consent_notice() -> str:
    return _sse_event(
        {
            "code": "consent_required",
            "message": "consentimento de processamento de dados não concedido (LGPD)",
        },
        event="notice",
    )


def _consent_blocked_response() -> StreamingResponse:
    def _stream() -> AsyncIterator[str]:
        yield _consent_notice()
        yield _sse_event("[DONE]")

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _guardrail_blocked_response(result) -> StreamingResponse:
    """SSE de bloqueio por guardrail (sem chamar o LLM, sem persistir)."""
    data = {
        "code": "guardrail_blocked",
        "message": result.reason or "conteúdo bloqueado por guardrail",
        "guard": result.guard,
    }

    def _stream() -> AsyncIterator[str]:
        yield _sse_event(data, event="notice")
        yield _sse_event("[DONE]")

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chats/{chat_id}/messages")
async def create_message(
    chat_id: int,
    payload: MessageCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    request_id = current_request_id()
    guard_result = _input_guardrail(payload.content)
    if not guard_result.allowed:
        return _guardrail_blocked_response(guard_result)
    if _consent_required() and not current_user.consent_granted:
        return _consent_blocked_response()
    user_model = UserService(UserRepositorySQLAlchemy(session)).resolve_model(current_user)
    try:
        await run_in_threadpool(_ensure_chat_owned, session, chat_id, current_user.id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="chat não encontrado") from None
    message_service = _message_service(session)

    async def _setup() -> object:
        return await run_in_threadpool(
            message_service.create_message,
            chat_id=chat_id,
            content=payload.content,
            role="user",
        )

    user_message = await _setup()

    async def _file_readiness() -> object:
        return await run_in_threadpool(_file_service(session).rag_readiness, chat_id)

    readiness = await _file_readiness()
    processing = readiness.processing
    failed = readiness.failed

    async def _history() -> list[str]:
        def _load() -> list[str]:
            from langchain_core.messages import HumanMessage

            from einstein_test.services.rag.graph import trim_history

            contents = [
                m.content
                for m in message_service.list_messages(chat_id=chat_id)
                if m.id != user_message.id
            ]
            msgs = [HumanMessage(content=c) for c in contents]
            trimmed = trim_history(msgs, max_messages=MAX_HISTORY)
            return [m.content for m in trimmed]

        return await run_in_threadpool(_load)

    def _notice() -> str:
        if readiness.code == "no_ready_files":
            return _sse_event(
                {
                    "code": "no_ready_files",
                    "message": "nenhum arquivo pronto para responder — envie arquivos",
                },
                event="notice",
            )
        if readiness.code == "partial_files":
            return _sse_event(
                {
                    "code": "partial_files",
                    "message": "resposta baseada apenas nos arquivos prontos",
                    "processing": processing,
                    "failed": failed,
                },
                event="notice",
            )
        return ""

    async def _stream() -> AsyncIterator[str]:
        if not readiness.can_answer:
            # AC-001: sem arquivos prontos -> notice SSE, sem LLM, sem assistant
            yield _notice()
            logger.info("mensagem sem arquivos prontos", extra={
                "event": "no_ready_files", "chat_id": chat_id,
                "message_id": user_message.id, "request_id": request_id,
            })
            return
        collected: list[str] = []
        sources: list[Any] = []
        try:
            history = await _history()
            notice = _notice()
            if notice:
                logger.info("aviso de arquivos parciais", extra={
                    "event": "partial_files_notice", "chat_id": chat_id,
                    "processing": processing, "failed": failed,
                    "request_id": request_id,
                })
                yield notice
            rag = await run_in_threadpool(_rag_service, session, user_model)
            n_tokens = 0
            use_memory = await run_in_threadpool(_is_postgres)
            async for ev in rag.astream_with_nodes(
                question=user_message.content,
                chat_id=chat_id,
                history=history,
                use_memory=use_memory,
                user_id=current_user.id,
            ):
                if ev["kind"] == "node":
                    yield _sse_event({"node": ev["node"]}, event="node")
                elif ev["kind"] == "token":
                    collected.append(ev["content"])
                    n_tokens += 1
                    yield _sse_event(ev["content"])
                elif ev["kind"] == "done":
                    sources = ev["sources"]
            answer = _refine_answer(
                "".join(collected), sources, user_message.content
            )
            assistant = await run_in_threadpool(
                message_service.create_message,
                chat_id=chat_id,
                content=answer,
                role="assistant",
            )
            logger.info("RAG concluído", extra={
                "event": "rag_done", "chat_id": chat_id,
                "message_id": assistant.id, "n_tokens": n_tokens,
                "request_id": request_id,
            })
            yield _sse_event({"message_id": assistant.id, "sources": sources})
            yield _sse_event("[DONE]")
        except Exception as exc:
            # evita gravar conteúdo do prompt/dados dos arquivos nos logs
            logger.error("falha no RAG", extra={
                "event": "rag_error", "chat_id": chat_id,
                "error": type(exc).__name__,
                "request_id": request_id,
            }, exc_info=True)
            yield _sse_event(
                {"error": "falha ao gerar resposta"},
                event="error",
            )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chats/{chat_id}/agent")
async def create_agent_answer(
    chat_id: int,
    payload: MessageCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    request_id = current_request_id()
    try:
        await run_in_threadpool(_ensure_chat_owned, session, chat_id, current_user.id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="chat não encontrado") from None

    guard_result = _input_guardrail(payload.content)
    if not guard_result.allowed:
        return _guardrail_blocked_response(guard_result)
    if _consent_required() and not current_user.consent_granted:
        return _consent_blocked_response()

    from einstein_test.services.rag.agent import agent_config, build_agent_graph

    user_model = UserService(UserRepositorySQLAlchemy(session)).resolve_model(current_user)
    rag = _rag_service(session, user_model)
    llm = rag.llm
    message_service = _message_service(session)

    async def _stream() -> AsyncIterator[str]:
        notice = await run_in_threadpool(_rag_readiness_notice, session, chat_id)
        if notice:
            yield notice
            yield _sse_event("[DONE]")
            return
        graph = build_agent_graph(
            llm=llm,
            retriever=VectorRepositorySQLAlchemy(session),
            embedder=rag.embedder,
            chat_id=chat_id,
        )
        tools_called: list[str] = []
        answer = ""
        try:
            await run_in_threadpool(
                message_service.create_message,
                chat_id=chat_id,
                content=payload.content,
                role="user",
            )
            async for state in graph.astream(
                {"question": payload.content},
                config=agent_config(chat_id),
                stream_mode="values",
            ):
                msgs = state.get("messages") or []
                for m in msgs:
                    if getattr(m, "tool_calls", None):
                        tools_called.extend(tc["name"] for tc in m.tool_calls)
                if state.get("answer"):
                    answer = state["answer"]
            if answer:
                await run_in_threadpool(
                    message_service.create_message,
                    chat_id=chat_id,
                    content=answer,
                    role="assistant",
                )
            yield _sse_event({"answer": answer, "tools_called": tools_called})
            yield _sse_event("[DONE]")
        except Exception as exc:
            logger.error("falha no agente", extra={
                "event": "agent_error", "chat_id": chat_id,
                "error": type(exc).__name__, "request_id": request_id,
            })
            yield _sse_event({"error": "falha ao gerar resposta"}, event="error")

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _hitl_checkpointer():
    """Checkpointer do HITL cacheado por processo (resume encontra o estado)."""
    if _is_postgres():
        from einstein_test.services.rag.checkpointer import build_postgres_checkpointer

        return build_postgres_checkpointer()

    return _hitl_memory_saver()


@lru_cache(maxsize=1)
def _hitl_memory_saver():
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


@router.post("/chats/{chat_id}/agent/hitl")
async def create_agent_hitl(
    chat_id: int,
    payload: MessageCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        await run_in_threadpool(_ensure_chat_owned, session, chat_id, current_user.id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="chat não encontrado") from None

    guard_result = _input_guardrail(payload.content)
    if not guard_result.allowed:
        raise HTTPException(status_code=422, detail=f"guardrail_blocked: {guard_result.reason}")
    if _consent_required() and not current_user.consent_granted:
        raise HTTPException(status_code=403, detail="consent_required")

    from einstein_test.services.rag.hitl import build_hitl_graph, hitl_config

    user_model = UserService(UserRepositorySQLAlchemy(session)).resolve_model(current_user)
    rag = _rag_service(session, user_model)
    message_service = _message_service(session)
    await run_in_threadpool(
        message_service.create_message,
        chat_id=chat_id,
        content=payload.content,
        role="user",
    )
    checkpointer = await run_in_threadpool(_hitl_checkpointer)
    graph = build_hitl_graph(
        llm=rag.llm,
        retriever=VectorRepositorySQLAlchemy(session),
        embedder=rag.embedder,
        checkpointer=checkpointer,
    )
    cfg = hitl_config(chat_id)
    await run_in_threadpool(
        graph.invoke,
        {"question": payload.content, "chat_id": chat_id},
        cfg,
    )
    snapshot = await run_in_threadpool(graph.get_state, cfg)
    interrupts = snapshot.interrupts
    draft = ""
    if interrupts:
        draft = interrupts[0].value.get("draft", "")
    return {"status": "pending", "chat_id": chat_id, "draft": draft}


@router.post("/chats/{chat_id}/agent/hitl/resume")
async def resume_agent_hitl(
    chat_id: int,
    payload: HitlResume,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    try:
        await run_in_threadpool(_ensure_chat_owned, session, chat_id, current_user.id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="chat não encontrado") from None

    from langgraph.types import Command

    from einstein_test.services.rag.hitl import build_hitl_graph, hitl_config

    user_model = UserService(UserRepositorySQLAlchemy(session)).resolve_model(current_user)
    rag = _rag_service(session, user_model)
    checkpointer = await run_in_threadpool(_hitl_checkpointer)
    graph = build_hitl_graph(
        llm=rag.llm,
        retriever=VectorRepositorySQLAlchemy(session),
        embedder=rag.embedder,
        checkpointer=checkpointer,
    )
    cfg = hitl_config(chat_id)
    snapshot = await run_in_threadpool(graph.get_state, cfg)
    if not snapshot.interrupts:
        raise HTTPException(status_code=409, detail="nenhuma revisão pendente para este chat")
    resume_value = {"approved": payload.approved}
    if not payload.approved:
        resume_value["feedback"] = payload.feedback or "resposta insatisfatória"
    out = await run_in_threadpool(
        graph.invoke,
        Command(resume=resume_value),
        cfg,
    )
    answer = out.get("answer", "")
    if answer:
        await run_in_threadpool(
            _message_service(session).create_message,
            chat_id=chat_id,
            content=answer,
            role="assistant",
        )
    return {"status": "approved" if payload.approved else "regenerated", "answer": answer}


@router.get("/chats/{chat_id}/messages", response_model=Page[MessageRead])
def list_messages(
    chat_id: int,
    page: int = Query(MIN_PAGE, ge=MIN_PAGE, le=MAX_PAGE),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=MIN_PAGE, le=MAX_PAGE_SIZE),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> Page[MessageRead]:
    try:
        _ensure_chat_owned(session, chat_id, current_user.id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="chat não encontrado") from None
    service = _message_service(session)
    items, total = service.list_messages_page(chat_id=chat_id, page=page, page_size=page_size)
    return Page[MessageRead].build(
        items=[MessageRead.model_validate(m) for m in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/messages/{message_id}", response_model=MessageRead)
def get_message(
    message_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> MessageRead:
    service = _message_service(session)
    try:
        message = service.get_message(message_id)
        _ensure_chat_owned(session, message.chat_id, current_user.id)
    except (MessageNotFoundError, ChatNotFoundError):
        raise HTTPException(
            status_code=404, detail="message não encontrado"
        ) from None
    return MessageRead.model_validate(message)


@router.delete("/messages/{message_id}")
def delete_message(
    message_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    service = _message_service(session)
    try:
        message = service.get_message(message_id)
        _ensure_chat_owned(session, message.chat_id, current_user.id)
        service.delete_message(message_id)
    except (MessageNotFoundError, ChatNotFoundError):
        raise HTTPException(
            status_code=404, detail="message não encontrado"
        ) from None
    return {"status": "ok"}

@router.post("/chats/{chat_id}/multi-query")
async def create_multi_query_answer(
    chat_id: int,
    payload: MessageCreate,
    queries: int = Query(default=3, ge=1, le=10),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    request_id = current_request_id()
    try:
        await run_in_threadpool(_ensure_chat_owned, session, chat_id, current_user.id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="chat não encontrado") from None

    guard_result = _input_guardrail(payload.content)
    if not guard_result.allowed:
        return _guardrail_blocked_response(guard_result)
    if _consent_required() and not current_user.consent_granted:
        return _consent_blocked_response()

    from einstein_test.services.rag.multiquery import build_multiquery_graph, multi_config

    user_model = UserService(UserRepositorySQLAlchemy(session)).resolve_model(current_user)
    rag = _rag_service(session, user_model)
    message_service = _message_service(session)

    async def _stream() -> AsyncIterator[str]:
        notice = await run_in_threadpool(_rag_readiness_notice, session, chat_id)
        if notice:
            yield notice
            yield _sse_event("[DONE]")
            return
        graph = build_multiquery_graph(
            llm=rag.llm,
            retriever=VectorRepositorySQLAlchemy(session),
            embedder=rag.embedder,
            chat_id=chat_id,
        )
        try:
            await run_in_threadpool(
                message_service.create_message,
                chat_id=chat_id,
                content=payload.content,
                role="user",
            )
            out = await run_in_threadpool(
                graph.invoke,
                {"question": payload.content, "queries": queries},
                multi_config(chat_id),
            )
            answer = out.get("answer", "")
            sources = out.get("sources", [])
            if answer:
                await run_in_threadpool(
                    message_service.create_message,
                    chat_id=chat_id,
                    content=answer,
                    role="assistant",
                )
            yield _sse_event({"answer": answer, "sources": sources})
            yield _sse_event("[DONE]")
        except Exception as exc:
            logger.error("falha no multi-query", extra={
                "event": "multi_query_error", "chat_id": chat_id,
                "error": type(exc).__name__, "request_id": request_id,
            })
            yield _sse_event({"error": "falha ao gerar resposta"}, event="error")

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chats/{chat_id}/agent/supervisor")
async def create_supervisor_answer(
    chat_id: int,
    payload: MessageCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    request_id = current_request_id()
    try:
        await run_in_threadpool(_ensure_chat_owned, session, chat_id, current_user.id)
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="chat não encontrado") from None

    guard_result = _input_guardrail(payload.content)
    if not guard_result.allowed:
        return _guardrail_blocked_response(guard_result)
    if _consent_required() and not current_user.consent_granted:
        return _consent_blocked_response()

    from einstein_test.services.rag.supervisor import build_supervisor, supervisor_config

    user_model = UserService(UserRepositorySQLAlchemy(session)).resolve_model(current_user)
    rag = _rag_service(session, user_model)
    message_service = _message_service(session)

    async def _stream() -> AsyncIterator[str]:
        notice = await run_in_threadpool(_rag_readiness_notice, session, chat_id)
        if notice:
            yield notice
            yield _sse_event("[DONE]")
            return
        graph = build_supervisor(
            llm=rag.llm,
            retriever=VectorRepositorySQLAlchemy(session),
            embedder=rag.embedder,
            chat_id=chat_id,
        )
        try:
            await run_in_threadpool(
                message_service.create_message,
                chat_id=chat_id,
                content=payload.content,
                role="user",
            )
            out = await run_in_threadpool(
                graph.invoke,
                {"question": payload.content, "chat_id": chat_id},
                supervisor_config(chat_id),
            )
            answer = out.get("answer", "")
            agents = out.get("agents_used", [])
            if answer:
                await run_in_threadpool(
                    message_service.create_message,
                    chat_id=chat_id,
                    content=answer,
                    role="assistant",
                )
            yield _sse_event({"answer": answer, "agents_used": agents})
            yield _sse_event("[DONE]")
        except Exception as exc:
            logger.error("falha no supervisor", extra={
                "event": "supervisor_error", "chat_id": chat_id,
                "error": type(exc).__name__, "request_id": request_id,
            })
            yield _sse_event({"error": "falha ao gerar resposta"}, event="error")

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
