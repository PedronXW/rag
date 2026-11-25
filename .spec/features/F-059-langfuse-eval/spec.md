# F-059 — Langfuse para avaliação (scores por trace + batched)

Usa o Langfuse como plataforma de avaliação do RAG: (1) **scores por trace** —
o harness (evaluate_ragas) registra `create_score` no trace do graph quando
habilitado (avaliação contínua); (2) **batched** — `run_batched_evaluation` do
Langfuse para avaliar datasets offline.

## Histórias
- U-001 — Como dev, quero que cada execução avaliada do RAG registre scores no
  Langfuse (trace), **para** acompanhar qualidade na UI.
- U-002 — Como dev, quero avaliar datasets offline via `run_batched_evaluation`
  do Langfuse, **para** avaliação declarativa.
- U-003 — Como dev, quero que o eval continue determinístico nos testes, **para**
  não depender de Langfuse real.

## Critérios
- AC-001 — [U-001] `evaluate_ragas`/harness registra scores no Langfuse (via
  `create_score`) por caso avaliado, quando habilitado.
- AC-002 — [U-002] Existe um runner de `run_batched_evaluation` do Langfuse para
  datasets.
- AC-003 — [U-003] Com Langfuse desabilitado, o harness roda sem registrar
  (determinístico).
- AC-004 — [U-003] Os 589 testes existentes continuam passando (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (scores por trace + run_batched_evaluation) — confirmada
- A-002 — (Langfuse desabilitado nos testes; client stub/mock) — confirmada

## Perguntas em aberto
- (nenhuma)