# F-039 — State avançado e memória de conversa no graph RAG

Migra o graph RAG do einstein-test (atualmente `retrieve → generate` com
`RagState` simples e histórico via `history: list[str]`) para o padrão
avançado do rag-evolute: `MessagesState` com reducers, checkpointer
`PostgresSaver` (thread_id = chat_id) e eventos por nó no streaming SSE.
Preserva multi-tenant por chat e o `ChatLLM` Protocol existente.

## Histórias
- U-001 — Como usuário, quero que a conversa seja persistida via checkpointer
  (thread_id = chat_id), **para** que o graph recupere o estado real entre
  mensagens.
- U-002 — Como dev, quero o estado do graph como `MessagesState` com reducers,
  **para** acumular histórico e chunks corretamente entre nós.
- U-003 — Como usuário, quero que o SSE emita eventos por nó + tokens +
  [DONE] com sources, **para** acompanhar o progresso e a base da resposta.
- U-004 — Como dev, quero preservar multi-tenant: o checkpointer e a memória
  são por chat (thread_id = chat_id).

## Critérios
- AC-001 — [U-002] `RagState` passa a usar `MessagesState` com canal
  `messages` via reducer `add_messages`; `chunks`/`sources` com reducer de
  substituição.
- AC-002 — [U-001] `PostgresSaver` configurado como checkpointer do graph,
  com `thread_id = chat_id` para memória persistente de conversa.
- AC-003 — [U-001] O histórico de mensagens do chat é recuperado do
  checkpoint ao invocar o graph com um `thread_id` (chat_id).
- AC-004 — [U-003] O endpoint SSE emite `event: node` por nó executado
  (retrieve/grade/generate), além dos tokens e do [DONE].
- AC-005 — [U-003] O [DONE] do SSE inclui `sources` (chunk_id, document_id,
  content) recuperadas.
- AC-006 — [U-004] Dois chats distintos (thread_id distintos) não compartilham
  memória no checkpoint.
- AC-007 — [U-002] O `ChatLLM` Protocol e o `ChatOpenAIAdapter` passam a
  receber mensagens (não só `prompt` string) para alimentar o `MessagesState`.

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Contrato SSE muda: adiciona eventos por nó e sources no [DONE]) — confirmada
- A-002 — (Memória via PostgresSaver com thread_id = chat_id) — confirmada
- A-003 — (Reducers completos: add_messages + substituição de chunks/sources) — confirmada
- A-004 — (Preservar multi-tenant e o ChatLLM Protocol) — confirmada

## Perguntas em aberto
- (nenhuma)