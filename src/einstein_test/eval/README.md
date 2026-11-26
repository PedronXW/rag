# eval — harness de avaliação do RAG (F-038)

Mede, por evidência, a qualidade do RAG: retrieval, assertividade, fidelidade,
cobertura, robustez e perf/custo. **Não faz parte do runtime da API** — é uma
ferramenta independente.

## Arquivos

- `cli.py` — entry point `uv run python -m einstein_test.eval` (backends
  `fake`/`real`/`langfuse`, judges `fake`/`llm`, saídas `md`/`html`/`json`).
- `dataset.py` — leitura do dataset gold (`eval/dataset/dev.jsonl`).
- `runner.py` — orquestra a avaliação por pergunta.
- `generation.py` — geração de resposta (RAG in-process).
- `judge.py` — juiz (fake determinístico ou LLM).
- `metrics.py` — `recall@5/12`, `MRR@5`, `hit@1`, correctness, faithfulness,
  coverage, similarity, robustez.
- `report.py` — relatório final.

## Como rodar

```bash
uv run python -m einstein_test.eval --backend fake --limit 2 --format md   # CI (sem chave)
uv run python -m einstein_test.eval --backend real --judge llm \
  --format html --out eval/reports/latest.html                             # real
uv run python -m einstein_test.eval --backend langfuse --judge llm         # Langfuse (experimento)
```

O `--backend real`/`--backend langfuse` limpam as linhas criadas no banco ao
final e só rodam em banco local (exceto com `EVAL_ALLOW_NON_LOCAL=1`).

## Backend Langfuse (F-059)

`--backend langfuse` roda o RAG real por pergunta e publica como **experimento
Langfuse**: espelha o dataset gold em `einstein-rag-eval` (upsert por id) e
registra scores no trace de cada item — veredito do juiz (`correctness`,
`faithfulness`, `coverage`, 1–5) e métricas de retrieval (`recall@5/12`,
`MRR@5`, `hit@1`). Requer `LANGFUSE_ENABLED=true` e a infra self-hosted
(`docker compose up langfuse`); o nome do dataset vem de `EVAL_DATASET_NAME`
(default `einstein-rag-eval`).
