# F-051 — Streaming rico com astream_events

Evolui o streaming do RAG para usar `astream_events` do LangGraph, que emite
eventos granulares: `on_chain_start`/`on_chain_end` por nó e
`on_chat_model_stream` (tokens reais do LLM). Mantém o contrato SSE do
`/messages` (event node + token + done + [DONE]).

## Histórias
- U-001 — Como usuário, quero ver tokens reais do LLM em streaming, **para**
  feedback imediato.
- U-002 — Como dev, quero usar `astream_events` para eventos granulares por nó
  e por token, **para** observabilidade mais rica.
- U-003 — Como usuário, quero que o contrato SSE existente seja preservado,
  **para** não quebrar clientes.

## Critérios
- AC-001 — [U-002] Existe um helper que consome `astream_events` e emite
  eventos de nó (`on_chain_start`/`on_chain_end`) e de token
  (`on_chat_model_stream`).
- AC-002 — [U-001] Os tokens reais do LLM são emitidos em streaming.
- AC-003 — [U-003] O formato SSE do `/messages` (event node + token + done +
  [DONE]) é preservado.
- AC-004 — [U-002] Os eventos de início/fim de cada nó são emitidos, em ordem.

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (astream_events com on_chain_start/end e on_chat_model_stream) — confirmada
- A-002 — (Contrato SSE preservado) — confirmada
- A-003 — (Testes com stubs determinísticos) — confirmada

## Perguntas em aberto
- (nenhuma)