# F-043 — Observabilidade Langfuse no RAG

Observabilidade do RAG com Langfuse self-host: serviço no docker-compose
(porta 3000, banco dedicado no Postgres) e callback de trace no graph RAG
clássico (`POST /chats/{chat_id}/messages`). Habilitável via env
`LANGFUSE_ENABLED`; desabilitado nos testes determinísticos.

## Critérios de aceite implementados
- **AC-001** — Serviço `langfuse` no docker-compose (self-host, banco
  dedicado `langfuse` criado via initdb).
- **AC-002** — `get_langfuse_handler()` retorna o `CallbackHandler` quando
  habilitado; `graph_config(chat_id)` injeta `callbacks`.
- **AC-003** — Com `LANGFUSE_ENABLED` desabilitado, o handler é `None` e o
  graph roda sem callbacks (testes determinísticos intactos).
- **AC-004** — O `astream_with_nodes` do `/messages` usa `graph_config`, que
  injeta callbacks quando habilitado.

## Arquivos
- `src/einstein_test/services/rag/callbacks.py` — `langfuse_enabled`,
  `get_langfuse_handler`, `graph_config`.
- `src/einstein_test/services/rag/service.py` — `_astream_events` usa
  `graph_config`.
- `docker-compose.yml` — serviço `langfuse`.
- `docker/initdb/create-langfuse-db.sh` — banco dedicado `langfuse`.

## Como rodar
```bash
docker compose up -d                     # sobe postgres+minio+redis+langfuse
LANGFUSE_ENABLED=true uv run uvicorn ... # habilita traces
# acesse http://localhost:3000 (admin@localhost/admin) para ver os traces
```

## Correções do deep review
- **Client habilitado**: `get_langfuse_handler` inicializa o `Langfuse` client
  com `tracing_enabled=True` antes do `CallbackHandler()` (padrão oficial SDK
  v5) — traces de fato enviados.
- **Imagem compatível**: `langfuse/langfuse:4` no compose (SDK v5 requer
  server ≥ v4.59).
- **Teste reforçado**: AC-002 valida `client._tracing_enabled is True` (não
  falso-verde).
- **Blindagem**: fixture autouse força `LANGFUSE_ENABLED=false` nos testes.
- Removido artefato acidental `=5.0`.
