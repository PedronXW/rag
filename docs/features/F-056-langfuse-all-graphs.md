# F-056 — Tracing Langfuse nos graphs novos

Unifica os `*_config` dos graphs RAG para injetar callbacks do Langfuse (via
`graph_config`), dando traces a `/agent`, `/supervisor`, `/multi-query` e
`/hitl` — antes só o `/messages` tinha.

## Critérios de aceite implementados
- **AC-001** — `agent_config` injeta callbacks quando habilitado (thread_id).
- **AC-002** — `supervisor_config` injeta callbacks (thread `supervisor:{id}`).
- **AC-003** — `multi_config` injeta callbacks (thread `multi:{id}` +
  recursion_limit/max_concurrency).
- **AC-004** — `hitl_config` injeta callbacks (thread `hitl:{id}`).
- **AC-005** — Regressão: 579 testes passando.

## Arquivos
- `services/rag/agent.py`, `supervisor.py`, `multiquery.py`, `hitl.py` —
  `*_config` usam `graph_config` + mantêm thread_id/recursion_limit.

## Observação
Com `LANGFUSE_ENABLED=false` (default dos testes), os configs não têm
`callbacks` — determinístico.
