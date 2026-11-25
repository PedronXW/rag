# F-048 — Robustez: retry de nós e avaliação do RAG

Fechar o ciclo de robustez e qualidade: (1) um decorator/helper de **retry de
nós** do graph (re-executa um nó com backoff em falhas transientes, além do
retry do LLM); (2) um **harness de avaliação determinístico** (relevance,
fidelity, accuracy) que roda o graph com stubs e gera scores — sem depender de
outro LLM.

## Histórias
- U-001 — Como dev, quero re-executar nós do graph com backoff em falhas
  transientes, **para** tolerar falhas de rede/storage sem derrubar a execução.
- U-002 — Como dev, quero avaliar o RAG com métricas determinísticas, **para**
  medir regressão de qualidade.
- U-003 — Como dev, quero que a avaliação rode com stubs (sem LLM real),
  **para** CI determinística.

## Critérios
- AC-001 — [U-001] Existe um helper/decorator `retry_node` que re-executa um
  nó com backoff em falhas transientes.
- AC-002 — [U-001] O nó com retry retorna o resultado após sucesso na
  re-execução (falha transiente é absorvida).
- AC-003 — [U-002] Existe um harness `evaluate_rag` que roda o graph e calcula
  `relevance`, `fidelity` e `accuracy`.
- AC-004 — [U-003] O harness usa stubs determinísticos (FakeChatLLM/retriever),
  sem LLM real.
- AC-005 — [U-002] O harness é executável via CLI/script (exit 0 no limiar).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Retry de nós + avaliação determinística) — confirmada
- A-002 — (Harness sem LLM judge; métricas relevance/fidelity/accuracy) — confirmada
- A-003 — (Testes com stubs) — confirmada

## Perguntas em aberto
- (nenhuma)