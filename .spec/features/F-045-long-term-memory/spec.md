# F-045 — Memória de longo prazo com PostgresStore

Memória de longo prazo do RAG via `PostgresStore` (LangGraph): fatos e
preferências do usuário persistidos por namespace (`user_id`), recuperados em
novas conversas (memória **entre chats**, diferente do checkpointer por
thread). Exposição via endpoints REST de memória e integração no prompt do
graph.

## Histórias
- U-001 — Como usuário, quero que fatos/preferências meus persistam entre
  conversas, **para** que o RAG lembre de mim em novos chats.
- U-002 — Como dev, quero um `PostgresStore` por usuário (namespace), **para**
  memória de longo prazo isolada por usuário.
- U-003 — Como usuário, quero ver/editar minha memória via API, **para**
  controlar o que o sistema lembra.
- U-004 — Como usuário, quero que a memória de longo prazo entre no prompt do
  RAG, **para** respostas personalizadas.

## Critérios
- AC-001 — [U-002] Existe um `PostgresStore` (LangGraph) com `put`/`get` por
  namespace (`user_id`).
- AC-002 — [U-001] `put_memory(user_id, key, value)` grava um fato/preferência.
- AC-003 — [U-001] `get_memory(user_id, key)` recupera o fato persistido.
- AC-004 — [U-003] `GET /users/{id}/memory` lista a memória do usuário;
  `PUT /users/{id}/memory` grava um item.
- AC-005 — [U-004] O graph do RAG recupera a memória de longo prazo do usuário
  e a inclui no prompt de geração.
- AC-006 — [U-002] Dois usuários distintos não compartilham memória
  (namespaces isolados).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (PostgresStore por usuário, namespace = user_id) — confirmada
- A-002 — (Endpoints GET/PUT /users/{id}/memory) — confirmada
- A-003 — (Memória entra no prompt do RAG) — confirmada
- A-004 — (Testes com store injetável; Postgres em produção) — confirmada

## Perguntas em aberto
- (nenhuma)