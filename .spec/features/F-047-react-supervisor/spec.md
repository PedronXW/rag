# F-047 — create_react_agent e supervisor multi-agente

Evolui o agentic RAG com duas abordagens complementares: (1) `create_react_agent`
(pre-built do langgraph-prebuilt) como alternativa ao StateGraph manual do
G-002; (2) um **supervisor** que orquestra agentes especializados
(retrieval/doc/evaluate) via subgraphs. Nova rota
`POST /chats/{chat_id}/agent/supervisor` (SSE); rotas existentes preservadas.

## Histórias
- U-001 — Como dev, quero comparar `create_react_agent` (pre-built) com o
  StateGraph manual, **para** escolher a melhor abstração.
- U-002 — Como usuário, quero que um supervisor orquestre agentes
  especializados, **para** resolver perguntas que exigem múltiplas habilidades.
- U-003 — Como usuário, quero que o supervisor delegue a sub-agentes
  (retrieval, doc, evaluate) e consolide a resposta, **para** composição
  avançada.
- U-004 — Como usuário, quero a rota `/agent/supervisor` sem quebrar as
  existentes.

## Critérios
- AC-001 — [U-001] Existe um `create_react_agent` compilado (com tools do chat,
  multi-tenant) que roda e responde.
- AC-002 — [U-003] Existe um supervisor (StateGraph/subgraph) que roteia a
  pergunta para sub-agentes especializados.
- AC-003 — [U-003] Os sub-agentes (retrieval/doc/evaluate) executam e retornam
  resultados ao supervisor, que consolida a resposta final.
- AC-004 — [U-002] `POST /chats/{chat_id}/agent/supervisor` responde (SSE) com
  a resposta consolidada e os agentes utilizados.
- AC-005 — [U-004] As rotas existentes (`/messages`, `/agent`, `/agent/hitl`,
  `/multi-query`) continuam funcionando (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (create_react_agent + supervisor) — confirmada
- A-002 — (Rota /agent/supervisor, SSE) — confirmada
- A-003 — (Sub-agentes: retrieval/doc/evaluate) — confirmada
- A-004 — (Testes com FakeChatLLM determinístico) — confirmada

## Perguntas em aberto
- (nenhuma)