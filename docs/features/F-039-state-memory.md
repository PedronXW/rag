# F-039 — State avançado e memória de conversa no graph RAG

Migra o graph RAG do einstein-test para o padrão avançado do rag-evolute:
`MessagesState` com reducers, checkpointer `PostgresSaver` (thread_id =
chat_id) e eventos por nó no streaming SSE. Preserva multi-tenant por chat e o
`ChatLLM` Protocol.

## Critérios de aceite implementados

- **AC-001** — `RagState` usa `MessagesState` com canal `messages` via
  `add_messages`; `chunks`/`sources` com reducer de substituição.
- **AC-002** — `build_postgres_checkpointer()` produz um `PostgresSaver`
  (memória persistente); `thread_config(chat_id)` monta o config.
- **AC-003** — Invocações no mesmo `thread_id` (chat_id) acumulam o histórico
  de mensagens via checkpointer.
- **AC-004** — O SSE do `POST /chats/{chat_id}/messages` emite `event: node`
  por nó executado (retrieve/generate), além dos tokens e do [DONE].
- **AC-005** — O [DONE] inclui `sources` (chunk_id, document_id, content)
  recuperadas.
- **AC-006** — Dois `thread_id` distintos não compartilham memória no
  checkpoint.
- **AC-007** — O `ChatLLM`/graph recebe mensagens (não só `prompt` string);
  o prompt chega como `HumanMessage`.

## Arquivos

- `src/einstein_test/services/rag/graph.py` — `RagState(MessagesState)` com
  reducers; `build_graph_with_state(llm, retriever, embedder, checkpointer)`.
- `src/einstein_test/services/rag/checkpointer.py` — `build_postgres_checkpointer`
  e `thread_config`.
- `src/einstein_test/services/rag/service.py` — `RagService` com checkpointer
  e `astream_with_nodes` (eventos por nó + sources).
- `src/einstein_test/api/messages.py` — SSE com `event: node` e sources no
  [DONE].
- `pyproject.toml` — dependência `langgraph-checkpoint-postgres`.

## Como rodar

O endpoint `POST /chats/{chat_id}/messages` agora emite `event: node` por nó,
tokens em `data:`, e `[DONE]` com `message_id` e `sources`. A memória de
conversa é persistida por chat via `PostgresSaver` (thread_id = chat_id)
quando o banco é Postgres; nos testes, usa `MemorySaver`.