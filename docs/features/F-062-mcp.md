# F-062 — MCP (Model Context Protocol)

Expõe as tools do RAG via Model Context Protocol usando o SDK **Python** `mcp`
(`MCPServer`). Sem serviço Node — encaixa na stack Python do projeto.

## Tools expostas
- `rag_search(chat_id, question)` — busca vetorial por chunks (wrapper de `retrieve_chunks`).
- `fetch_document(chat_id, file_id)` — conteúdo de um arquivo do chat (multi-tenant).
- `get_memory(user_id, key)` / `put_memory(user_id, key, value)` — memória de longo prazo.
- `list_chats(user_id)` / `create_chat(user_id, name)`.

Multi-tenant por `chat_id`/`user_id` passados como argumento (cliente de confiança).

## Executar
```bash
uv run python -m einstein_test.mcp                 # stdio (default)
uv run python -m einstein_test.mcp --transport http  # streamable HTTP em :8001
```

## Critérios de aceite
- **AC-001** — `build_server()` registra as tools `rag_search`, `fetch_document`,
  `get_memory`, `put_memory`, `list_chats`, `create_chat`.
- **AC-002** — `rag_search` retorna os chunks relevantes (retrieval real).
- **AC-003** — `get_memory`/`put_memory` e `list_chats`/`create_chat` funcionam.
- **AC-004** — Engine/session injetável para testes; regressão preservada.

## Arquivos
- `mcp/server.py` — `build_server()`, `main()`, `set_engine()` (injeção p/ testes).
- `mcp/__init__.py` — entry point do módulo.