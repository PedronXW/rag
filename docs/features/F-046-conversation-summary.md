# F-046 — Sumarização de conversa no RAG

Quando o histórico da conversa excede o limite (mesmo após trim), um nó gera
um `SystemMessage` de resumo dos turnos antigos e o usa como contexto inicial
no prompt — em vez de descartar informação. `IsLastStep` (campo gerenciado)
permite rodar a sumarização só na última etapa.

## Critérios de aceite implementados
- **AC-001** — `should_summarize` detecta histórico acima do limite.
- **AC-002** — `build_prompt_with_summary` coloca o resumo antes das mensagens
  recentes no prompt.
- **AC-003** — `maybe_summarize` é o nó gateado por `IsLastStep` (última etapa).
- **AC-004** — Histórico dentro do limite não gera resumo (sem custo extra).
- **AC-005** — `summarize_old_turns` combina resumo + últimas N mensagens
  (parceria com trim).

## Arquivos
- `src/einstein_test/services/rag/summary.py` — `should_summarize`,
  `summarize_old_turns`, `maybe_summarize`, `build_prompt_with_summary`.

## Observação
`SummaryMessage` não existe nesta versão do langchain-core; o resumo é
representado como `SystemMessage` (padrão real de resumo do LangGraph).

## Correções do deep review
- **Integração real**: `RagService._with_summary` aplica a sumarização no fluxo
  do `/messages` (gera resumo quando o histórico excede o limite, substituindo
  turnos antigos por `SystemMessage` + recentes).
- **`summary_node`** implementado (gate `IsLastStep` via campo gerenciado).
- **Bug de slicing corrigido**: `history = list(history)` normalizado; edge case
  `max_messages=0` protegido.
- **Testes reforçados**: integração no fluxo real + caminhos dentro/fora do
  limite.
