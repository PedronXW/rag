# F-061 — Deduplicação estrutural do RAG

Elimina duplicações reais entre módulos do RAG: pool de conexão Postgres único,
constantes de retrieval de fonte única, `eval/*` usando `retrieve_chunks`,
prompts usando `build_prompt`, e `_is_transient` importado do retrieval.

## Critérios de aceite implementados
- **AC-001** — `_pool()` de conexão Postgres centralizado no `checkpointer.py`;
  `longterm._pool` o reutiliza (fonte única).
- **AC-002** — `CHUNKS_PER_FILE`/`MAX_CHUNKS` só no `retrieval.py`;
  `eval/generation.py` usa `retrieve_chunks`; `eval/cli.py` usa as constantes
  (sem hardcode).
- **AC-003** — `managed`/`multiquery`/`supervisor` usam `build_prompt` do graph
  (sem prompts `<contexto>` manuais).
- **AC-004** — `_is_transient` importado de `retrieval.py` no service (sem
  re-export em cadeia via llm).
- **AC-005** — Agregação de métricas única via `EvalReport.aggregates` em `eval/`.
- **AC-006** — Regressão: 627 testes passando.

## Arquivos
- `services/rag/checkpointer.py`, `longterm.py` — pool único.
- `services/rag/graph.py` — constantes importadas do retrieval (removida
  rededaração).
- `services/rag/{managed,multiquery,supervisor}.py` — `build_prompt`.
- `services/rag/service.py` — `_is_transient` do retrieval.
- `eval/generation.py`, `eval/cli.py` — `retrieve_chunks`/constantes.