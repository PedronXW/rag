# F-040 — Agente react com tools no RAG

Agente react com ferramentas: o LLM decide em loop (via `ToolNode` e
`Command(goto=...)`) quais ferramentas chamar até finalizar com uma resposta.
Tools: `search_chunks` (busca vetorial por chat) e `fetch_document` (conteúdo
de um arquivo do chat), multi-tenant. Nova rota `POST /chats/{chat_id}/agent`;
o `POST /chats/{chat_id}/messages` (RAG clássico) permanece intacto.

## Critérios de aceite implementados
- **AC-001** — `build_tools(retriever, embedder, chat_id)` cria `search_chunks`
  e `fetch_document` via `@tool`.
- **AC-002/007** — `search_chunks` busca chunks do chat; `fetch_document`
  retorna conteúdo apenas se o arquivo pertence ao chat (multi-tenant).
- **AC-003** — O graph do agente roda loop de tools (ToolMessage volta) e
  finaliza com resposta.
- **AC-004** — O agente usa `Command(goto="tools"|"finalize")` (sem router fixo).
- **AC-005** — `POST /chats/{chat_id}/agent` (SSE) responde com `answer` +
  `tools_called` no [DONE].
- **AC-006** — `POST /chats/{chat_id}/messages` (RAG clássico) continua
  funcionando.

## Arquivos
- `src/einstein_test/services/rag/agent.py` — tools, `AgentState`,
  `build_agent_graph`, `agent_config`.
- `src/einstein_test/api/messages.py` — rota `POST /chats/{chat_id}/agent`.
- `src/einstein_test/services/rag/llm.py` — `FakeChatLLM.bind_tools` (teste).

## Como rodar
`POST /chats/{chat_id}/agent` (SSE): emite tokens da resposta e `[DONE]` com
`{answer, tools_called}`.
- **AC-007** — As tools rejeitam acesso a arquivos/chunks de outro chat (multi-tenant).

## Segurança (correções do deep review)
- **Multi-tenant (AC-007)**: `fetch_document` usa `get_chunks_for_chat(chat_id,
  file_id)` (novo método no repositório) que filtra por `File.chat_id` e
  `File.deleted_at` — impede ler arquivos de outros chats.
- **Prompt injection**: `_agent_node` injeta `SystemMessage` instruindo que
  tool results são dados não confiáveis e que `fetch_document` só usa file_id
  do `search_chunks` do mesmo chat.
- **Loop limitado**: `agent_config` define `recursion_limit=10`.
- **Tool-calling real**: `ChatOpenAIAdapter.bind_tools` (delega ao langchain);
  `ChatLLM` Protocol declara `bind_tools`.
