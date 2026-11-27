# F-060 — Remoção de módulos órfãos do RAG

Limpeza geral — remover código morto do RAG sem caller em produção. Remover os
módulos órfãos de `services/rag` (evaluation, ragas_eval, react, rich_stream,
streaming), os re-exports mortos do `__init__.py` e `build_graph`/`build_graph_async`
(usados só pelos órfãos). Manter `mcp/` (funcionalidade bacana) e a avaliação
via Langfuse (backend do eval).

## Histórias
- U-001 — Como dev, quero remover módulos sem caller em produção, **para**
  reduzir manutenção e confusão.
- U-002 — Como dev, quero remover re-exports mortos do `__init__.py`, **para**
  ter uma API pública enxuta.
- U-003 — Como dev, quero que `mcp/` e a avaliação via Langfuse sejam mantidos,
  **para** não perder funcionalidade.

## Critérios
- AC-001 — [U-001] `services/rag/evaluation.py` é removido (sem caller em src).
- AC-002 — [U-001] `services/rag/ragas_eval.py` é removido (avaliação via Langfuse).
- AC-003 — [U-001] `services/rag/react.py`, `rich_stream.py` e `streaming.py` são
  removidos (sem caller em src).
- AC-004 — [U-002] `build_graph`/`build_graph_async` saem do `__init__.py` (mortos);
  os re-exports da feature Ollama (ChatOllamaAdapter, build_chat_llm,
  is_ollama_model) e de config (LLM_MAX_RETRIES/LLM_RETRY_BACKOFF) são mantidos
  (API pública validada por testes).
- AC-005 — [U-003] `mcp/` continua funcional; a avaliação via Langfuse continua.
- AC-006 — [U-001/U-003] A suíte de testes passa (regressão) com os testes
  dos módulos removidos também removidos.

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Remover módulos órfãos de services/rag) — confirmada
- A-002 — (Manter mcp/ e avaliação via Langfuse) — confirmada
- A-003 — (hooks de teste mantidos; sem remoção de set/reset_memory_store) — confirmada

## Perguntas em aberto
- (nenhuma)