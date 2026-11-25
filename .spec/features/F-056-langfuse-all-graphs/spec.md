# F-056 — Tracing Langfuse nos graphs novos

Hoje apenas o `/messages` (RagService) injeta callbacks do Langfuse. Os graphs
novos (`/agent`, `/supervisor`, `/multi-query`, `/hitl`) usam `*_config` sem
callbacks — sem traces. Esta feature unifica os `*_config` para injetar os
callbacks do Langfuse (via `graph_config`), mantendo `thread_id`/`recursion_limit`.

## Histórias
- U-001 — Como dev, quero traces do Langfuse em todos os graphs do RAG, **para**
  observabilidade completa.
- U-002 — Como dev, quero que `agent_config`/`supervisor_config`/`multi_config`/
  `hitl_config` incluam callbacks quando habilitado, **para** consistência.
- U-003 — Como dev, quero preservar `thread_id`/`recursion_limit` nos configs,
  **para** não quebrar memória/limites.

## Critérios
- AC-001 — [U-002] `agent_config` injeta `callbacks` do Langfuse quando
  habilitado, mantendo `thread_id`.
- AC-002 — [U-002] `supervisor_config` injeta `callbacks` quando habilitado,
  mantendo `thread_id`.
- AC-003 — [U-002] `multi_config` injeta `callbacks` quando habilitado,
  mantendo `thread_id`/`recursion_limit`.
- AC-004 — [U-002] `hitl_config` injeta `callbacks` quando habilitado,
  mantendo `thread_id`.
- AC-005 — [U-003] Os 574 testes existentes continuam passando (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (callbacks via graph_config, mantendo thread_id/recursion_limit) — confirmada
- A-002 — (Langfuse desabilitado nos testes determinísticos) — confirmada

## Perguntas em aberto
- (nenhuma)