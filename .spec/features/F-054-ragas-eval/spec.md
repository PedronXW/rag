# F-054 — Avaliação LLM-as-judge com RAGAS

Complementa o harness determinístico (H-005/F-038) com avaliação
LLM-as-judge usando RAGAS: métricas `faithfulness` e `answer_correctness`
calculadas por um LLM juiz, sobre o graph RAG. O juiz é injetável (stub nos
testes; LLM real em produção).

## Histórias
- U-001 — Como dev, quero avaliar o RAG com métricas LLM-judge (RAGAS),
  **para** medir fidelidade e correção de forma mais robusta que heurística.
- U-002 — Como dev, quero que o juiz seja injetável, **para** testes
  determinísticos sem LLM real.
- U-003 — Como dev, quero rodar a avaliação RAGAS via CLI, **para** CI.

## Critérios
- AC-001 — [U-001] `evaluate_ragas` calcula `faithfulness` e
  `answer_correctness` usando um LLM juiz sobre os casos.
- AC-002 — [U-002] O juiz é injetável (stub determinístico nos testes).
- AC-003 — [U-001] O harness roda o graph RAG para obter contexto + resposta.
- AC-004 — [U-003] O harness é executável via CLI/script (exit 0 no limiar).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (RAGAS com métricas faithfulness/answer_correctness; juiz injetável) — confirmada
- A-002 — (Testes com juiz stub determinístico) — confirmada

## Perguntas em aberto
- (nenhuma)