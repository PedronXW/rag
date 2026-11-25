# F-050 — Memória longa como estado gerenciado no graph

Aprofunda a memória de longo prazo (H-002): em vez de apenas pré-pendurar a
memória no prompt via `_with_memory`, o `PostgresStore`/`InMemoryStore` é
injetado como estado gerenciado do graph e um nó dedica leitura/escrita de
memória. Integração mais profunda com o LangGraph.

## Histórias
- U-001 — Como dev, quero que o `store` de memória longa seja injetado no graph
  (estado gerenciado), **para** que os nós leiam/escrevam memória diretamente.
- U-002 — Como dev, quero um nó de memória que recupere a memória do usuário e
  a inclua no estado, **para** uso no prompt.
- U-003 — Como usuário, quero que a memória longa continue funcionando
  (multi-tenant, isolada por usuário), **para** não regredir a H-002.
- U-004 — Como usuário, quero que os endpoints de memória continuem, **para**
  controlar o que o sistema lembra.

## Critérios
- AC-001 — [U-001] O graph RAG aceita o `store` como estado gerenciado
  (injetável), e os nós têm acesso a ele.
- AC-002 — [U-001/U-002] Existe um nó de memória que recupera a memória do
  usuário do store e a inclui no estado (chunks/prompt).
- AC-003 — [U-002] A memória recuperada entra no prompt de geração.
- AC-004 — [U-003] A memória é isolada por usuário (namespace user_id) —
  multi-tenant preservado.
- AC-005 — [U-004] Os endpoints GET/PUT /users/{id}/memory continuam
  funcionando (regressão H-002).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Store injetado como estado gerenciado; nó de memória no graph) — confirmada
- A-002 — (Memória isolada por usuário, namespace user_id) — confirmada
- A-003 — (Testes com store injetável; endpoints preservados) — confirmada

## Perguntas em aberto
- (nenhuma)