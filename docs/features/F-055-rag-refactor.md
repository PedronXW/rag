# F-055 — Refatorar RAG: helper de retrieval único + streaming rico

Consolida dívida técnica do RAG: um helper único de retrieval elimina o drift
de constantes hardcoded (chunks_per_file/max_chunks) e a duplicação de
`_chunk_to_dict`/`_is_transient`; o streaming rico (`astream_events`) fica
disponível via `RagService.astream_with_nodes(rich=True)`.

## Critérios de aceite implementados
- **AC-001** — `retrieval.py` expõe `retrieve_chunks`/`chunk_to_dict`/
  `_is_transient`; os módulos usam essas funções (sem hardcode).
- **AC-002** — Não há `_chunk_to_dict`/`_is_transient` duplicados (fonte única).
- **AC-003** — `RagService.astream_with_nodes(rich=True)` usa `astream_events`;
  o `/messages` mantém o streaming padrão com tokens individuais.
- **AC-004** — Regressão: 574 testes passando.

## Arquivos
- `src/einstein_test/services/rag/retrieval.py` — fonte única de retrieve/
  chunk_to_dict/_is_transient/constantes.
- Refatorados: `graph.py`, `agent.py`, `hitl.py`, `multiquery.py`,
  `supervisor.py`, `llm.py`, `retry.py`.
- `service.py` — `astream_with_nodes(rich=True)` via `_astream_rich`.

## Como rodar
Streaming rico (eventos por nó + tokens via `astream_events`):
`rag.astream_with_nodes(question, chat_id, history, rich=True)`.
