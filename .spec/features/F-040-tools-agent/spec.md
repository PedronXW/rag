# F-040 — Agente react com tools no RAG

Adiciona um agente react ao RAG do einstein-test: o LLM decide em loop
(via `ToolNode` e `Command(goto=...)`) quais ferramentas chamar até produzir a
resposta final. Ferramentas: `search_chunks` (busca vetorial por chat) e
`fetch_document` (conteúdo de um arquivo do chat), ambas multi-tenant. Nova
rota `POST /chats/{chat_id}/agent`; o `POST /chats/{chat_id}/messages` (RAG
clássico) permanece intacto.

## Histórias
- U-001 — Como usuário, quero que o LLM decida dinamicamente quais ferramentas
  usar (buscar chunks, buscar documento) para responder, **para** que o fluxo
  não dependa de roteador pré-programado.
- U-002 — Como usuário, quero que o agente rode em loop (chama tool, observa
  resultado, decide próximo passo) e finalize com uma resposta, **para**
  permitir requisições de múltiplos passos.
- U-003 — Como usuário, quero que as tools respeitem o multi-tenant (só dados
  do meu chat), **para** que não haja vazamento entre chats.
- U-004 — Como usuário, quero que o `POST /chats/{chat_id}/messages` (RAG
  clássico) continue funcionando, **para** não quebrar o contrato existente.

## Critérios
- AC-001 — [U-001] Existe um nó com `ToolNode` registrando `search_chunks` e
  `fetch_document` (via `@tool` do langchain).
- AC-002 — [U-001/U-003] `search_chunks(question, chat_id)` busca chunks do
  chat via retriever (filtra por chat_id); `fetch_document(file_id, chat_id)`
  retorna o conteúdo do arquivo se pertencer ao chat.
- AC-003 — [U-002] O graph do agente compila e, quando o LLM decide chamar uma
  tool, o resultado volta ao estado e o loop continua (múltiplos steps
  observáveis via `ToolMessage`).
- AC-004 — [U-002] O agente finaliza com `answer`; usa `Command(goto=...)`
  para rotear (sem router fixo).
- AC-005 — [U-004] `POST /chats/{chat_id}/agent` responde (SSE) com a resposta
  do agente e a lista de `tools_called` no [DONE].
- AC-006 — [U-003/U-004] `POST /chats/{chat_id}/messages` continua funcionando
  (RAG clássico, regressão).
- AC-007 — [U-003] As tools rejeitam acesso a arquivos/chunks de outro chat
  (multi-tenant).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (StateGraph + Command + ToolNode, adaptado do rag-evolute F-005) — confirmada
- A-002 — (Tools: search_chunks + fetch_document, multi-tenant) — confirmada
- A-003 — (Nova rota /agent; /messages clássico intacto) — confirmada
- A-004 — (Testes com FakeChatLLM que emite tool_calls determinísticos) — confirmada

## Perguntas em aberto
- (nenhuma)