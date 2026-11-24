# F-041 — Human-in-the-loop no graph RAG

Adiciona o padrão human-in-the-loop ao graph RAG: o graph pausa após gerar a
resposta (antes de persistir) e aguarda aprovação humana via `interrupt()` +
`Command(resume=...)`. Nova rota `POST /chats/{chat_id}/agent/hitl` (inicia e
pausa) e `POST /chats/{chat_id}/agent/hitl/resume` (aprova/rejeita com
feedback). O estado pendente persiste via checkpointer.

## Histórias
- U-001 — Como usuário, quero que o graph pause após gerar a resposta e
  aguarde aprovação, **para** que nenhuma resposta seja entregue sem revisão.
- U-002 — Como usuário, quero aprovar, rejeitar ou editar a resposta no resume,
  **para** controlar o que é entregue.
- U-003 — Como usuário, quero que o estado pendente persista (via checkpointer
  + thread_id = chat_id), **para** que a retomada seja uma chamada independente.

## Critérios
- AC-001 — [U-001] Existe um graph com nó `review` que usa `interrupt()` após a
  geração.
- AC-002 — [U-001] Invocar com `thread_id` sem resume retorna estado pendente
  (`snapshot.next` + `interrupts`) e NÃO persiste a resposta.
- AC-003 — [U-002] `Command(resume={"approved": True})` retoma e entrega a
  resposta gerada (persistindo a message assistant).
- AC-004 — [U-002] `Command(resume={"approved": False, "feedback": ...})` refaz
  a geração considerando o feedback.
- AC-005 — [U-003] O estado pendente persiste entre a invocação pausada e o
  resume (mesmo `thread_id`).
- AC-006 — [U-001/U-003] A rota `POST /chats/{chat_id}/agent/hitl` devolve o
  estado `pending` (draft); `POST /chats/{chat_id}/agent/hitl/resume` retoma.

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Interrupt após geração) — confirmada
- A-002 — (Rota /agent/hitl + /resume) — confirmada
- A-003 — (Memória via checkpointer com thread_id = chat_id, como G-001) — confirmada
- A-004 — (Resume aceita approved + feedback) — confirmada

## Perguntas em aberto
- (nenhuma)