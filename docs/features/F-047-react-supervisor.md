# F-047 — create_react_agent e supervisor multi-agente

Duas abordagens agentic: `create_react_agent` (pre-built do langgraph-prebuilt)
e um **supervisor** que orquestra sub-agentes especializados
(retrieval/doc/evaluate) via StateGraph. Rota `POST /chats/{chat_id}/agent/supervisor`.

## Critérios de aceite implementados
- **AC-001** — `build_react_agent` usa `create_react_agent` (pre-built) com
  tools do chat.
- **AC-002** — `build_supervisor` tem nó `route` que roteia a pergunta.
- **AC-003** — Sub-agentes (retrieval + finalize) executam e consolidam a
  resposta com `agents_used`.
- **AC-004** — `POST /chats/{chat_id}/agent/supervisor` (SSE) responde com
  `{answer, agents_used}` + [DONE].
- **AC-005** — Rotas existentes preservadas (regressão).

## Arquivos
- `src/einstein_test/services/rag/react.py` — `build_react_agent`.
- `src/einstein_test/services/rag/supervisor.py` — `build_supervisor`.
- `src/einstein_test/api/messages.py` — rota `/agent/supervisor`.

## Observação
`create_react_agent` está deprecado no langgraph 1.x (movido para
`langchain.agents.create_agent`); o `langchain` não está instalado, então o
módulo `react.py` fica como referência e o supervisor (StateGraph) é o caminho
ativo.

## Correções do deep review
- **Roteamento real**: `build_supervisor` usa `add_conditional_edges` —
  pergunta "retrieval" vai ao sub-agente de retrieval; "general" pula direto
  ao finalize (sem custo de busca).
- **`agents_used` confiável**: reflete o caminho realmente executado.
- **Namespace de thread**: `supervisor:{chat_id}` (evita colisão com o RAG).
