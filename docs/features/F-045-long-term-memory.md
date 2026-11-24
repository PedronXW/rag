# F-045 — Memória de longo prazo com PostgresStore

Memória de longo prazo do RAG via `PostgresStore` (LangGraph): fatos e
preferências do usuário persistidos por namespace (`user_id`), recuperados em
novas conversas (memória entre chats). Exposição via endpoints REST e
integração no prompt do RAG.

## Critérios de aceite implementados
- **AC-001** — `build_postgres_store()` (PostgresStore do LangGraph).
- **AC-002** — `put_memory(user_id, key, value)` grava um fato/preferência.
- **AC-003** — `get_memory(user_id, key)` recupera o fato persistido.
- **AC-004** — `GET /users/{id}/memory` lista; `PUT /users/{id}/memory` grava.
- **AC-005** — O `RagService` recupera a memória e a inclui no prompt
  (`_with_memory`), via `user_id`.
- **AC-006** — Namespaces isolados por usuário.

## Arquivos
- `src/einstein_test/services/rag/longterm.py` — store injetável, put/get/list.
- `src/einstein_test/api/memory.py` — endpoints GET/PUT /users/{id}/memory.
- `src/einstein_test/services/rag/service.py` — `_with_memory` + `user_id`.
- `src/einstein_test/api/messages.py` — passa `user_id` ao streaming.
- `src/einstein_test/main.py` — registra o router de memória.

## Como rodar
```bash
# endpoints
PUT /users/{id}/memory   {"key": "preferencia", "value": {"estilo": "curto"}}
GET /users/{id}/memory   # lista os fatos do usuário
```
A memória entra no prompt do RAG automaticamente (via user_id).

## Correções do deep review
- **PostgresStore corrigido**: usa `ConnectionPool` + `PostgresStore(_pool())` +
  `setup()` (paridade com checkpointer) — `from_conn_string` era contextmanager.
- **Prompt injection mitigado**: memória marcada como "dado não confiável",
  serializada em JSON.
- **list_memory** com limite explícito (100).
- **Teste AC-001** reforçado (mocka pool/setup, valida tipo).
