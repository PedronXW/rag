# F-057 — IsLastStep real e build_prompt unificado

`summary_node` passa a receber `is_last_step` (campo gerenciado `IsLastStep`)
e só gera resumo na última etapa; `build_prompt` do graph vira a fonte única
(com `summary`/`memory` opcionais), eliminando o drift do prompt.

## Critérios de aceite implementados
- **AC-001** — `summary_node(is_last_step=...)` não gera resumo fora da última
  etapa; `IsLastStep` usado como campo gerenciado.
- **AC-002** — `build_prompt` único aceita `summary`/`memory`; `summary` o
  reutiliza (`build_prompt_with_summary` delega).
- **AC-003** — Sumarização gera resumo quando o histórico excede o limite.
- **AC-004** — Regressão: 584 testes passando.

## Arquivos
- `services/rag/graph.py` — `build_prompt` com `summary`/`memory` opcionais.
- `services/rag/summary.py` — `summary_node(is_last_step)`,
  `build_prompt_with_summary` delega ao graph.

## Observação
`IsLastStep` do langgraph já é `Annotated[bool, ManagedValue]`; é usado como
tipo de campo direto (`is_last_step: IsLastStep`), não com `Annotated` extra.
