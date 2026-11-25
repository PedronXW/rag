# F-048 — Robustez: retry de nós e avaliação do RAG

Fechar o ciclo de robustez e qualidade: (1) `retry_node` re-executa um nó com
backoff em falhas transientes; (2) harness de avaliação determinístico
(relevance, fidelity, accuracy) que roda o graph com stubs, sem LLM judge.

## Critérios de aceite implementados
- **AC-001** — `retry_node(node, retries, backoff)` re-executa o nó em falha
  transiente (TimeoutError/ConnectionError/OSError).
- **AC-002** — Após sucesso na re-execução, a falha é absorvida e o resultado
  retornado.
- **AC-003** — `evaluate_rag` calcula relevance, fidelity e accuracy.
- **AC-004** — O harness usa stubs (FakeChatLLM/retriever/embedder).
- **AC-005** — `python -m einstein_test.services.rag.evaluation` é CLI (exit 0
  no limiar).

## Arquivos
- `src/einstein_test/services/rag/retry.py` — `retry_node`.
- `src/einstein_test/services/rag/evaluation.py` — `evaluate_rag`, `aggregate`,
  `main`.

## Como rodar
```bash
uv run python -m einstein_test.services.rag.evaluation --cases 5 --min 0.0
```

## Correções do deep review
- **retry integrado ao fluxo real**: `_retrieve_sync`/`_retrieve_async` do
  graph envolvem a busca com `retry_node` (absorve falhas transientes de
  embedding/busca).
- **fidelity determinística**: a resposta é grounded se está contida no
  contexto recuperado (o graph básico não emite `grounded`).
- **retry.py limpo**: removido código morto pós-loop; `retries<1` levanta
  `ValueError`.
