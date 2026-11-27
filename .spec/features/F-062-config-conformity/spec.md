# F-062 — Conformidade de config e observabilidade dos graphs

Garante que todos os graphs do RAG usem o `graph_config` do callbacks (que
injeta callbacks do Langfuse), e remove a duplicação de `thread_config` no
`checkpointer.py`. Consolida a constante de modelo (DEFAULT_CHAT_MODEL vs
settings), se aplicável.

## Histórias
- U-001 — Como dev, quero que todos os graphs RAG usem o mesmo config (com
  callbacks Langfuse), **para** observabilidade completa e consistente.
- U-002 — Como dev, quero remover o `thread_config` duplicado do checkpointer,
  **para** não haver duas formas de montar config.
- U-003 — Como dev, quero uma única fonte para a configuração de modelo, **para**
  não divergir default/settings.

## Critérios
- AC-001 — [U-001] Todos os `*_config` dos graphs (agent, supervisor, multi,
  hitl) usam `graph_config` (com callbacks quando habilitado).
- AC-002 — [U-002] `checkpointer.thread_config` é removido ou delega ao
  `graph_config`.
- AC-003 — [U-001] Com Langfuse habilitado, os configs dos graphs incluem
  `callbacks`; desabilitado, não incluem (determinístico).
- AC-004 — [U-003] A configuração de modelo tem fonte única (settings)
  consumida pelos adapters/llm.
- AC-005 — [U-001..U-003] A suíte de testes passa (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (todos os configs usam graph_config; thread_config removido) — confirmada
- A-002 — (modelo com fonte única em settings) — confirmada

## Perguntas em aberto
- (nenhuma)