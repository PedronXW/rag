# F-059 — Langfuse para avaliação (scores por trace + experimentos)

Usa o Langfuse como plataforma de avaliação do RAG: scores por trace
(`create_score`) na avaliação contínua, push de dataset gold e experimentos
(`run_experiment`) com scores por item, e `run_batched_evaluation` para
datasets offline. Desabilitado por padrão (determinístico nos testes).

## Critérios de aceite implementados
- **AC-001** — `register_trace_score` cria score no trace; `evaluate_ragas`
  aceita `score_manager` para registrar por caso.
- **AC-002** — `run_batched_eval` usa `run_batched_evaluation` do Langfuse.
- **AC-003** — Com Langfuse desabilitado, nada é registrado.
- **AC-004** — Regressão: testes existentes continuam passando.
- **AC-005** — `push_dataset` espelha o dataset gold em um dataset Langfuse
  (upsert por `q.id`).
- **AC-006** — `run_experiment` roda task + evaluators no Langfuse; evaluators
  de veredito (`correctness`/`faithfulness`/`coverage`) e de retrieval
  (`recall@5/12`, `MRR@5`, `hit@1`) registram scores por item.

## Arquivos
- `services/rag/eval_langfuse.py` — `register_trace_score`, `push_dataset`,
  `run_experiment`, `make_verdict_evaluator`, `make_retrieval_evaluator`,
  `run_batched_eval`, `langfuse_eval_enabled`.
- `services/rag/ragas_eval.py` — `evaluate_ragas(score_manager=...)`.
- `eval/cli.py` — backend `--backend langfuse`.

## Como rodar
```bash
# experimento (push dataset + scores por item)
uv run python -m einstein_test.eval --backend langfuse --judge llm
```

```python
# scores por trace (avaliação contínua)
evaluate_ragas(cases, judge=..., score_manager=register_trace_score, trace_id="...")

# batched (dataset offline)
run_batched_eval(mapper=..., evaluators=[...])

# push + experimento (SDK)
push_dataset(name="einstein-rag-eval", questions=[...])
run_experiment(name="einstein-rag-eval", data=[...], task=..., evaluators=[...])
```
