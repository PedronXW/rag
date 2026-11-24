# F-042 — Multi-query com subgraphs no RAG

Multi-query RAG com subgraphs e execução paralela (map-reduce): o graph
reescreve a pergunta em N variações (LLM), dispara um retrieve paralelo por
variação via `Send`, e funde os resultados (dedup por chunk_id + rank por
score) antes de gerar. Subgraph de retrieval reutilizável. Nova rota
`POST /chats/{chat_id}/multi-query`.

## Histórias
- U-001 — Como usuário, quero que o graph reescreva a pergunta em N variações,
  **para** cobrir diferentes ângulos da mesma intenção.
- U-002 — Como usuário, quero que as N buscas rodem em paralelo (map via
  `Send`), **para** aproveitar a concorrência.
- U-003 — Como usuário, quero que os resultados sejam fundidos (dedup + rank
  por score), **para** que o contexto final seja não-redundante e ordenado.
- U-004 — Como usuário, quero um subgraph de retrieval reutilizável como nó,
  **para** compor sem duplicar lógica.

## Critérios
- AC-001 — [U-004] Existe um subgraph de retrieval (compilado) usado como nó
  no graph principal.
- AC-002 — [U-001] O nó de query-rewriting gera N variações da pergunta via
  LLM quando `queries > 1`.
- AC-003 — [U-002] O graph usa `Send` para disparar um retrieve por variação
  em paralelo.
- AC-004 — [U-002] O estado acumula os chunks de todas as variações via
  reducer.
- AC-005 — [U-003] A fusão deduplica chunks repetidos (chunk_id) e ranqueia
  por score máximo.
- AC-006 — [U-001/U-004] `POST /chats/{chat_id}/multi-query` aceita
  `{question, queries}` e responde com a resposta e sources.

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Multi-query completo com Send) — confirmada
- A-002 — (Usar ChatLLM para rewrite) — confirmada
- A-003 — (Nova rota /multi-query; /messages e /agent intactos) — confirmada
- A-004 — (Fusão: dedup por chunk_id + rank por score máximo) — confirmada

## Perguntas em aberto
- (nenhuma)