# F-051 — Streaming rico com astream_events

Streaming do RAG via `astream_events`: `node_events` emite início/fim de cada
nó (`on_chain_start`/`on_chain_end`); `token_events` emite os tokens reais do
LLM (`on_chat_model_stream` para BaseChatModel, ou o `answer` agregado para os
adaptadores custom). Contrato SSE preservado.

## Critérios de aceite implementados
- **AC-001/004** — `node_events` emite on_chain_start/end por nó, em ordem.
- **AC-002** — `token_events` emite os tokens reais do LLM.
- **AC-003** — `build_sse` preserva o formato node/token/done.

## Arquivos
- `src/einstein_test/services/rag/rich_stream.py` — `node_events`,
  `token_events`, `build_sse`.

## Observação
Com os adaptadores custom (`ChatLLM.stream → str`, sem `on_chat_model_stream`),
`token_events` captura o `answer` agregado do nó `generate` no `on_chain_end`.
Num `BaseChatModel` real, emite token a token via `on_chat_model_stream`.
- **AC-004** — os eventos de início/fim de cada nó são emitidos, em ordem.
