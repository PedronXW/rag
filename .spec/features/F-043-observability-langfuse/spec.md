# F-043 — Observabilidade Langfuse no RAG

Adiciona observabilidade ao RAG com Langfuse self-host: serviço no
docker-compose (porta 3000, banco dedicado no Postgres) e callback de trace no
graph RAG clássico (`POST /chats/{chat_id}/messages`). Habilitável via env
`LANGFUSE_ENABLED`; desabilitado nos testes determinísticos.

## Histórias
- U-001 — Como dev, quero traces de cada execução do graph RAG no Langfuse,
  **para** depurar e avaliar.
- U-002 — Como dev, quero o Langfuse provisionado self-host no docker-compose,
  **para** observabilidade sem serviço externo.
- U-003 — Como dev, quero que o tracing seja opcional (env), **para** não
  afetar testes determinísticos e ambientes sem Langfuse.

## Critérios
- AC-001 — [U-002] O docker-compose tem um serviço `langfuse` (self-host) com
  banco dedicado.
- AC-002 — [U-001] `get_langfuse_handler()` retorna o handler de callback do
  Langfuse quando habilitado; `graph_config()` injeta `callbacks` no config.
- AC-003 — [U-003] Com `LANGFUSE_ENABLED` desabilitado, o handler é `None` e o
  graph roda sem callbacks (testes determinísticos intactos).
- AC-004 — [U-001] O endpoint `/messages` passa `callbacks` ao invocar o graph
  quando o Langfuse está habilitado.

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Langfuse self-host no compose, banco dedicado) — confirmada
- A-002 — (Tracing no graph RAG clássico /messages) — confirmada
- A-003 — (Habilitável via env LANGFUSE_ENABLED; off nos testes) — confirmada
- A-004 — (Callback via langfuse.langchain.CallbackHandler, como no rag-evolute) — confirmada

## Perguntas em aberto
- (nenhuma)