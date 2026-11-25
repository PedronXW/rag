# F-058 — Integrar memória gerenciada e documentar módulos órfãos

Integra a memória de longo prazo **gerenciada** (`managed.memory_node`) ao
RagService (via `_with_memory`, agora delegando ao store). Documenta os módulos
órfãos (`react`, `ragas_eval`) como ferramentas CLI/alternativas.

## Critérios de aceite implementados
- **AC-001** — RagService usa a memória gerenciada (store) no fluxo.
- **AC-002** — `managed.memory_node`/`build_graph_with_memory_store` usados
  pelo RagService (via `_with_memory`).
- **AC-003** — `react` e `ragas_eval` com docstrings de ferramentas.
- **AC-004** — Regressão: 589 testes passando.

## Arquivos
- `services/rag/service.py` — `_with_memory` delega a `managed.memory_node`.
- `services/rag/managed.py`, `react.py`, `ragas_eval.py` — integrados/documentados.

## Observação
A memória gerenciada (store no graph) agora é a fonte da memória longa no
fluxo do `/messages` (antes `list_memory` duplicava a lógica).
