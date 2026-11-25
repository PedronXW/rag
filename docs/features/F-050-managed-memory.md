# F-050 — Memória longa como estado gerenciado no graph

Aprofunda a memória de longo prazo (H-002): o `store` é injetado no graph e um
nó de memória recupera os fatos do usuário para o estado/prompt. Isolada por
usuário (namespace user_id); endpoints H-002 preservados.

## Critérios de aceite implementados
- **AC-001** — `build_graph_with_memory_store` compila o graph com o store
  injetado.
- **AC-002** — `memory_node` recupera a memória do store e inclui no estado.
- **AC-003** — A memória entra no prompt de geração.
- **AC-004** — Memória isolada por usuário (multi-tenant).
- **AC-005** — Endpoints GET/PUT /users/{id}/memory continuam (regressão).

## Arquivos
- `src/einstein_test/services/rag/managed.py` — `memory_node`,
  `build_graph_with_memory_store`.

## Observação
O `memory_node` usa o store injetado (não o global) para ler a memória;
`list_memory` do H-002 continua usado pelos endpoints.
