# F-053 — Rate limit e readiness nas rotas RAG

Aplica às rotas RAG novas (`/agent`, `/multi-query`, `/supervisor`) o mesmo
gate de readiness (`rag_readiness`/`no_ready_files`/`partial_files`) que o
`/messages` tem, e garante rate limiting consistente — evitando respostas com
contexto vazio (alucinação) e uso abusivo.

## Histórias
- U-001 — Como usuário, quero que rotas RAG sem arquivos prontos emitam um
  notice (`no_ready_files`/`partial_files`) e não respondam com contexto vazio,
  **para** evitar alucinação.
- U-002 — Como dev, quero consistência de comportamento entre as rotas RAG,
  **para** que nenhuma rota rode sem readiness.
- U-003 — Como usuário, quero rate limiting nas rotas RAG, **para** evitar
  abuso.

## Critérios
- AC-001 — [U-001] As rotas `/agent`, `/multi-query` e `/supervisor` checam
  `rag_readiness` e emitem notice `no_ready_files` quando não há arquivos
  prontos (sem chamar o LLM).
- AC-002 — [U-001] Quando há arquivos parciais, emitem notice `partial_files`.
- AC-003 — [U-002] Com arquivos prontos, a rota roda normalmente (resposta
  gerada).
- AC-004 — [U-003] As rotas RAG são cobertas por rate limiting (429 quando
  excede).
- AC-005 — [U-003] O `/messages` continua funcionando (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (readiness nas rotas /agent, /multi-query, /supervisor) — confirmada
- A-002 — (Rate limiting via middleware existente) — confirmada
- A-003 — (Testes com stubs) — confirmada

## Perguntas em aberto
- (nenhuma)