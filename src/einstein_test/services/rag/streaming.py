"""Helpers de streaming (SSE) para o RAG (H-001).

`stream_tokens` itera sobre os tokens emitidos pelo LLM; `build_sse` formata
um evento Server-Sent Events.
"""

import json


def stream_tokens(llm, messages):
    """Itera sobre os tokens reais emitidos pelo LLM (stream)."""
    for token in llm.stream(messages):
        yield token


async def astream_tokens(llm, messages):
    """Itera sobre os tokens reais emitidos pelo LLM (astream)."""
    async for token in llm.astream(messages):
        yield token


def build_sse(data, *, event: str | None = None) -> str:
    """Formata um evento SSE (dados JSON-encoded)."""
    encoded = json.dumps(data, ensure_ascii=False, default=str)
    if event:
        return f"event: {event}\ndata: {encoded}\n\n"
    return f"data: {encoded}\n\n"


__all__ = ["stream_tokens", "astream_tokens", "build_sse"]