"""Serviço RAG: orquestra retrieve+generate e expõe a factory global.

`RagService` embrulha um LLM (default `ChatOpenAIAdapter`), um retriever
(session-based) e um embedder, expondo `stream`/`astream` via grafo LangGraph.
A factory (`get_rag_service`) cria o adapter por usuário com o modelo
escolhido (F-022); o embedder permanece compartilhado.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from einstein_test.services.rag.graph import (
    build_graph,
    build_graph_async,
    build_graph_with_state,
    build_prompt,
)
from einstein_test.services.rag.llm import (
    LLM_MAX_RETRIES,
    LLM_RETRY_BACKOFF,
    ChatOpenAIAdapter,
    ChatLLM,
    _is_transient,
)
from einstein_test.vector import Embedder, get_embedder
from einstein_test.vector.repository import VectorRepository


class RagService:
    def __init__(
        self,
        llm: ChatLLM | None = None,
        retriever: VectorRepository | None = None,
        embedder: Embedder | None = None,
        checkpointer=None,
    ) -> None:
        self.llm: ChatLLM = llm or ChatOpenAIAdapter()
        self.retriever = retriever
        self.embedder: Embedder = embedder or get_embedder()
        self.checkpointer = checkpointer
        self._graph = None
        self._graph_async = None

    def _embed_question(self, question: str) -> list[float]:
        """Gera o embedding da pergunta com retry transiente (tenacity)."""
        retrying = Retrying(
            stop=stop_after_attempt(LLM_MAX_RETRIES),
            wait=wait_exponential(multiplier=LLM_RETRY_BACKOFF, min=0.05, max=5),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self.embedder.embed([question])[0]
        raise RuntimeError("embedding falhou após retries")

    @property
    def graph(self) -> Any:
        if self.retriever is None:
            raise RuntimeError("retriever não configurado")
        if self._graph is None:
            self._graph = build_graph_with_state(
                llm=self.llm, retriever=self.retriever, embedder=self.embedder,
                async_=False, checkpointer=self.checkpointer,
            )
        return self._graph

    @property
    def graph_async(self) -> Any:
        if self.retriever is None:
            raise RuntimeError("retriever não configurado")
        if self._graph_async is None:
            self._graph_async = build_graph_with_state(
                llm=self.llm, retriever=self.retriever, embedder=self.embedder,
                async_=True, checkpointer=self.checkpointer,
            )
        return self._graph_async

    def stream(
        self,
        question: str,
        chat_id: int,
        history: list[str],
    ) -> Iterator[str]:
        """Gera a resposta RAG em streaming (tokens) via grafo LangGraph."""
        for chunk in self.graph.stream(
            {
                "question": question,
                "history": history,
                "chat_id": chat_id,
                "messages": [HumanMessage(content=question)],
            },
            stream_mode="custom",
            config={"configurable": {"thread_id": chat_id}},
        ):
            yield chunk

    async def astream(
        self,
        question: str,
        chat_id: int,
        history: list[str],
    ) -> AsyncIterator[str]:
        """Gera a resposta RAG em streaming async (tokens) via grafo."""
        async for chunk in self.graph_async.astream(
            {
                "question": question,
                "history": history,
                "chat_id": chat_id,
                "messages": [HumanMessage(content=question)],
            },
            stream_mode="custom",
            config={"configurable": {"thread_id": chat_id}},
        ):
            yield chunk

    async def astream_with_nodes(
        self,
        question: str,
        chat_id: int,
        history: list[str],
        *,
        use_memory: bool = False,
    ) -> AsyncIterator[dict]:
        """Streaming async emitindo eventos de nó + tokens + state final.

        Com `use_memory=True`, o graph async é compilado com um
        `AsyncPostgresSaver` (memória persistente por chat) dentro do event
        loop. Yields dicts: {"kind": "node"|"token"|"done", ...}.
        """
        sources: list[Any] = []
        if use_memory:
            from einstein_test.services.rag.checkpointer import (
                build_async_postgres_checkpointer,
            )

            async with build_async_postgres_checkpointer() as cp:
                await cp.setup()
                graph = build_graph_with_state(
                    llm=self.llm, retriever=self.retriever, embedder=self.embedder,
                    async_=True, checkpointer=cp,
                )
                async for ev in self._astream_events(graph, question, history, chat_id):
                    if ev["kind"] == "done":
                        sources = ev["sources"]
                    yield ev
        else:
            async for ev in self._astream_events(self.graph_async, question, history, chat_id):
                if ev["kind"] == "done":
                    sources = ev["sources"]
                yield ev

    async def _astream_events(self, graph, question, history, chat_id) -> AsyncIterator[dict]:
        from einstein_test.services.rag.callbacks import graph_config

        config = graph_config(chat_id=chat_id)
        sources: list[Any] = []
        async for mode, payload in graph.astream(
            {
                "question": question,
                "history": history,
                "chat_id": chat_id,
                "messages": [HumanMessage(content=question)],
            },
            stream_mode=["updates", "custom"],
            config=config,
        ):
            if mode == "updates":
                for node, update in payload.items():
                    if isinstance(update, dict) and "sources" in update:
                        sources = update["sources"]
                    yield {"kind": "node", "node": node}
            elif mode == "custom":
                yield {"kind": "token", "content": payload}
        yield {"kind": "done", "sources": sources}

    def _build_prompt(
        self,
        question: str,
        chunks: list[Any],
        history: list[str],
    ) -> str:
        return build_prompt(question, chunks, history)

    def _generate_answer(
        self,
        question: str,
        chat_id: int,
        history: list[str],
    ) -> str:
        return "".join(self.stream(question=question, chat_id=chat_id, history=history))


_rag_factory: Callable[..., RagService] | None = None
_shared_embedder: Embedder | None = None


def get_rag_service(
    retriever: VectorRepository, *, model: str | None = None
) -> RagService:
    """Retorna o serviço RAG para o retriever da sessão atual.

    O `ChatOpenAIAdapter` é criado **por usuário** com o `model` escolhido (ou
    o default do ambiente quando `model` é None) — não há mais singleton do
    LLM. O `Embedder` permanece compartilhado (client HTTP de embeddings).
    """
    global _shared_embedder
    if _rag_factory is not None:
        kwargs = {"model": model} if model is not None else {}
        return _rag_factory(retriever=retriever, **kwargs)
    if _shared_embedder is None:
        _shared_embedder = get_embedder()
    return RagService(
        llm=ChatOpenAIAdapter(model=model),
        retriever=retriever,
        embedder=_shared_embedder,
    )


def set_rag_factory(factory: Callable[..., RagService]) -> None:
    global _rag_factory
    _rag_factory = factory


def reset_rag_factory() -> None:
    global _rag_factory
    _rag_factory = None
    global _shared_embedder
    _shared_embedder = None


__all__ = ["RagService", "get_rag_service", "set_rag_factory", "reset_rag_factory"]