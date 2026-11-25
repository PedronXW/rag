# F-057 — IsLastStep real e build_prompt unificado

`summary_node` referencia `IsLastStep` apenas como docstring, sem usar de fato
o campo gerenciado. E `summary.build_prompt_with_summary` diverge do
`graph.build_prompt`. Esta feature implementa o gate de `IsLastStep` de verdade
no nó de sumarização e unifica o prompt (um único `build_prompt` com suporte a
resumo/memória).

## Histórias
- U-001 — Como dev, quero que `summary_node` use `IsLastStep` (campo gerenciado)
  de fato, **para** que o resumo só rode na última etapa.
- U-002 — Como dev, quero um único `build_prompt` (com resumo/memória opcionais),
  **para** eliminar o drift entre summary e graph.
- U-003 — Como usuário, quero que a sumarização continue funcionando, **para**
  não regredir o H-003.

## Critérios
- AC-001 — [U-001] `summary_node` usa `IsLastStep` como campo gerenciado; não
  gera resumo quando não é a última etapa.
- AC-002 — [U-002] Existe um `build_prompt` único (graph) que aceita
  resumo/memória opcionais; `summary` reutiliza-o (sem prompt divergente).
- AC-003 — [U-003] A sumarização continua gerando resumo quando o histórico
  excede o limite.
- AC-004 — [U-003] Os 579 testes existentes continuam passando (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (IsLastStep como campo gerenciado no nó) — confirmada
- A-002 — (build_prompt único com resumo/memória opcionais) — confirmada

## Perguntas em aberto
- (nenhuma)