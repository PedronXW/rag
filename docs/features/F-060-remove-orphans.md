# F-060 — Remoção de módulos órfãos do RAG

Limpeza geral — remove código morto do RAG sem caller em produção.

## Critérios de aceite implementados
- **AC-001** — `services/rag/evaluation.py` removido (sem caller em src).
- **AC-002** — `services/rag/ragas_eval.py` removido (avaliação via Langfuse).
- **AC-003** — `services/rag/react.py`, `rich_stream.py` e `streaming.py` removidos.
- **AC-004** — `build_graph`/`build_graph_async` saem do `__init__.py`; os
  re-exports da feature Ollama (ChatOllamaAdapter, build_chat_llm, is_ollama_model)
  e de config (LLM_MAX_RETRIES/LLM_RETRY_BACKOFF) são mantidos.
- **AC-005** — `mcp/` continua funcional; avaliação via Langfuse continua.
- **AC-006** — Regressão: 621 testes passando; testes dos módulos removidos
  também removidos/ajustados.

## Arquivos
- Removidos: `services/rag/{evaluation,ragas_eval,react,rich_stream,streaming}.py`.
- Removidos: `graph.build_graph`/`build_graph_async`, re-exports mortos.
- `services/rag/__init__.py` — API pública enxuta (mantém Ollama/config).
- Testes deduplicados/ajustados (remove import de órfãos).