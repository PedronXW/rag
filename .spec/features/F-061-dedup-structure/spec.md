# F-061 — Deduplicação estrutural do RAG

Elimina duplicações reais entre módulos do RAG: `_pool()` de ConnectionPool
(em checkpointer e longterm), constantes `CHUNKS_PER_FILE`/`MAX_CHUNKS`
(em graph e retrieval), busca de chunks manual em `eval/generation.py` e
hardcode em `eval/cli.py`, `aggregate()` múltiplos, `_is_transient`
re-exportado em cadeia, e prompts `<contexto>` manuais (managed/multiquery/
supervisor) que não usam `build_prompt`.

## Histórias
- U-001 — Como dev, quero uma única fonte para ConnectionPool Postgres, **para**
  não divergir a config de conexão.
- U-002 — Como dev, quero uma única fonte para constantes de retrieval e para a
  busca de chunks, **para** que mudanças de política se propaguem.
- U-003 — Como dev, quero que os prompts usem `build_prompt` (fonte única),
  **para** consistência de mitigação de prompt injection.
- U-004 — Como dev, quero um único `aggregate`, **para** menos manutenção.

## Critérios
- AC-001 — [U-001] Existe um pool de conexão Postgres único (compartilhado por
  checkpointer e longterm).
- AC-002 — [U-002] `CHUNKS_PER_FILE`/`MAX_CHUNKS` têm uma única fonte
  (`retrieval.py`); `eval/generation.py` e `eval/cli.py` usam `retrieve_chunks`
  (sem busca manual/hardcode).
- AC-003 — [U-003] `managed`/`multiquery`/`supervisor` usam `build_prompt` do
  graph (sem prompts `<contexto>` manuais).
- AC-004 — [U-002] `_is_transient` é importado da fonte única (`retrieval.py`),
  sem re-export em cadeia.
- AC-005 — [U-004] Existe um único `aggregate` (em `eval/`), reutilizado.
- AC-006 — [U-001..U-004] A suíte de testes passa (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (pool único; constantes e retrieve em retrieval.py) — confirmada
- A-002 — (prompts usam build_prompt; aggregate único em eval/) — confirmada

## Perguntas em aberto
- (nenhuma)