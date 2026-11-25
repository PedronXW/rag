# F-059 — Langfuse para avaliação (scores por trace + batched)

Usa o Langfuse como plataforma de avaliação do RAG: scores por trace
(`create_score`) na avaliação contínua e `run_batched_evaluation` para
datasets offline. Desabilitado por padrão (determinístico nos testes).

## Critérios de aceite implementados
- **AC-001** — `register_trace_score` cria score no trace; `evaluate_ragas`
  aceita `score_manager` para registrar por caso.
- **AC-002** — `run_batched_eval` usa `run_batched_evaluation` do Langfuse.
- **AC-003** — Com Langfuse desabilitado, nada é registrado.
- **AC-004** — Regressão: 594 testes passando.

## Arquivos
- `services/rag/eval_langfuse.py` — `register_trace_score`, `run_batched_eval`,
  `langfuse_eval_enabled`.
- `services/rag/ragas_eval.py` — `evaluate_ragas(score_manager=...)`.

## Como rodar
```python
# scores por trace (avaliação contínua)
evaluate_ragas(cases, judge=..., score_manager=register_trace_score, trace_id="...")

# batched (dataset offline)
run_batched_eval(mapper=..., evaluators=[...])
```