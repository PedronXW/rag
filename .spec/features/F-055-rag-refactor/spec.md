# F-055 — Refatorar RAG: helper de retrieval único + streaming rico

Consolida dívida técnica do RAG: (1) um helper único de retrieval
(`retrieval.py`) elimina o drift de constantes `chunks_per_file/max_chunks`
hardcoded em 4 módulos e a duplicação de `_chunk_to_dict`/`_is_transient`;
(2) o `/messages` passa a usar `astream_events` (rich_stream) para eventos por
nó + tokens reais, mantendo o contrato SSE.

## Histórias
- U-001 — Como dev, quero um único helper de retrieval, **para** que mudanças de
  política (`chunks_per_file`/`max_chunks`) se propaguem a todos os módulos.
- U-002 — Como dev, quero eliminar a duplicação de `_chunk_to_dict` e
  `_is_transient`, **para** menos manutenção e drift.
- U-003 — Como usuário, quero streaming rico (`astream_events`) disponível via
  `RagService`, **para** eventos por nó + tokens, sem quebrar o contrato SSE do
  `/messages` (que mantém tokens individuais).

## Critérios
- AC-001 — [U-001] Existe `retrieval.py` com `retrieve_chunks`/`chunk_to_dict`/
  `_is_transient`; os módulos do RAG usam essas funções (sem hardcode).
- AC-002 — [U-002] Não há `_chunk_to_dict`/`_is_transient` duplicados nos
  módulos do RAG (fonte única).
- AC-003 — [U-003] O `/messages` usa `astream_events` para emitir eventos por
  nó + tokens, mantendo o contrato SSE (node/token/done/[DONE]).
- AC-004 — [U-001/U-003] Os 567 testes existentes continuam passando
  (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (retrieval.py como fonte única; módulos refatorados) — confirmada
- A-002 — (rich_stream conectado ao /messages via astream_events) — confirmada
- A-003 — (Contrato SSE preservado) — confirmada

## Perguntas em aberto
- (nenhuma)