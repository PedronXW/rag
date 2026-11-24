# F-044 — Streaming token-real e trim_messages no RAG

Melhora o RAG multi-turno: (1) o histórico do `/messages` passa por
`trim_messages` (mantém as últimas N mensagens / até K tokens, descartando as
mais antigas) para não explodir o contexto; (2) o streaming passa a usar
`stream_mode="messages"` (tokens reais do LLM) mantendo o contrato SSE atual
(event node + token + done + [DONE]).

## Histórias
- U-001 — Como usuário, quero que conversas longas mantenham contexto útil sem
  estourar o limite do modelo, **para** que o RAG não degrade com histórico
  grande.
- U-002 — Como usuário, quero ver os tokens da resposta chegando em tempo real,
  **para** ter feedback imediato (streaming token-real).
- U-003 — Como dev, quero preservar o contrato SSE existente, **para** não
  quebrar clientes.

## Critérios
- AC-001 — [U-001] O histórico do `/messages` é limitado por `trim_messages`
  (mantém últimas N mensagens / até K tokens).
- AC-002 — [U-002] O `/messages` usa `stream_mode="messages"` para emitir
  tokens reais do LLM.
- AC-003 — [U-002] O SSE continua emitindo `event: node` por nó + tokens +
  `done` + `[DONE]` (contrato preservado).
- AC-004 — [U-001] Após o trim, o histórico passa a caber no limite de tokens
  (mensagens mais antigas são descartadas).
- AC-005 — [U-003] Os testes de SSE existentes continuam passando (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (trim_messages no /messages, via langchain_core) — confirmada
- A-002 — (stream_mode=messages no /messages, contrato SSE preservado) — confirmada
- A-003 — (Testes com FakeChatLLM determinístico) — confirmada

## Perguntas em aberto
- (nenhuma)