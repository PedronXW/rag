"""Streaming rico com astream_events (I-003).

`node_events` consome `astream_events` e emite início/fim de cada nó;
`token_events` emite os tokens reais do LLM (`on_chat_model_stream`). O
contrato SSE (node/token/done/[DONE]) é preservado via `build_sse`.
"""

import json


def build_sse(data, *, event: str | None = None) -> str:
    """Formata um evento SSE (dados JSON-encoded)."""
    encoded = json.dumps(data, ensure_ascii=False, default=str)
    if event:
        return f"event: {event}\ndata: {encoded}\n\n"
    return f"data: {encoded}\n\n"


async def node_events(graph, kwargs):
    """Itera sobre os eventos de início/fim de cada nó (astream_events).

    Yields tuplas ("start"|"end", nome_do_no) em ordem de execução.
    """
    async for event in graph.astream_events(kwargs, version="v2"):
        ev = event.get("event", "")
        name = event.get("name", "")
        if ev == "on_chain_start" and name not in ("LangGraph", "LangGraphGraph"):
            yield ("start", name)
        elif ev == "on_chain_end" and name not in ("LangGraph", "LangGraphGraph"):
            yield ("end", name)


async def token_events(graph, kwargs):
    """Itera sobre os tokens reais do LLM.

    Com um `BaseChatModel` real, captura `on_chat_model_stream` (token a token).
    Com os adaptadores custom do projeto (ChatLLM.stream → str, sem
    on_chat_model_stream), captura o `answer` agregado emitido no
    `on_chain_end` do nó `generate`. Yields strings.
    """
    async for event in graph.astream_events(kwargs, version="v2"):
        ev = event.get("event", "")
        if ev == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", None)
            if content:
                yield content
        elif ev == "on_chain_end" and event.get("name") == "generate":
            data = event.get("data", {}).get("output", {})
            if isinstance(data, dict) and data.get("answer"):
                yield data["answer"]


__all__ = ["build_sse", "node_events", "token_events"]