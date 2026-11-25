# F-058 — Integrar memória gerenciada e documentar módulos órfãos

Integra a memória de longo prazo **gerenciada** (`managed.memory_node` + store
no graph) ao fluxo do RAG, como complemento ao `_with_memory` atual. Documenta
os módulos órfãos (`react.py` create_agent, `ragas_eval.py`) como ferramentas
CLI/alternativas — sem dead code silencioso.

## Histórias
- U-001 — Como dev, quero que a memória gerenciada (store no graph) seja
  integrada ao RagService, **para** memória longa mais idiomática.
- U-002 — Como dev, quero que os módulos órfãos (`react`, `ragas_eval`) tenham
  documentação clara como ferramentas, **para** não haver dead code confuso.
- U-003 — Como usuário, quero que a memória longa continue funcionando, **para**
  não regredir H-002/I-002.

## Critérios
- AC-001 — [U-001] O RagService passa a usar a memória gerenciada (store) no
  fluxo, recuperando a memória do usuário no prompt.
- AC-002 — [U-001] `managed.memory_node`/`build_graph_with_memory_store` são
  usados pelo RagService (não só isolados).
- AC-003 — [U-002] `react` e `ragas_eval` têm documentação (docstring/README)
  explicitando que são ferramentas CLI/alternativas.
- AC-004 — [U-003] Os 584 testes existentes continuam passando (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (memória gerenciada integrada ao RagService) — confirmada
- A-002 — (react/ragas_eval documentados como ferramentas) — confirmada

## Perguntas em aberto
- (nenhuma)