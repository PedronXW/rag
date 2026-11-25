# F-054 — Avaliação LLM-as-judge com RAGAS

Complementa o harness determinístico com avaliação LLM-as-judge: métricas
`faithfulness` e `answer_correctness` calculadas por um juiz injetável sobre o
graph RAG. Juiz stub determinístico nos testes; `BaseChatModel`/RAGAS em
produção.

## Critérios de aceite implementados
- **AC-001** — `evaluate_ragas` calcula faithfulness e answer_correctness.
- **AC-002** — O juiz é injetável (Protocol `Judge`).
- **AC-003** — `collect_case` roda o graph para obter resposta + contexto.
- **AC-004** — CLI (`python -m einstein_test.services.rag.ragas_eval`) com exit
  0 no limiar.

## Arquivos
- `src/einstein_test/services/rag/ragas_eval.py` — `evaluate_ragas`,
  `collect_case`, `aggregate`, `main`.

## Como rodar
```bash
uv run python -m einstein_test.services.rag.ragas_eval --cases 5 --min 0.5
```

## Observação
O juiz stub é determinístico (faithfulness=1 se há resposta e contexto; senão
0). Para produção, injete um juiz RAGAS real (que usa um `BaseChatModel`).
