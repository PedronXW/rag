# F-052 — Migrar create_react_agent para create_agent

O `create_react_agent` do langgraph-prebuilt está **deprecado** no LangGraph
1.x (movido para `langchain.agents.create_agent`). Esta feature adiciona
`langchain` e migra o `react.py` (H-004) de "referência" para funcional usando
`langchain.agents.create_agent`.

## Histórias
- U-001 — Como dev, quero usar `langchain.agents.create_agent` (novo) em vez do
  `create_react_agent` deprecado, **para** não usar API legada.
- U-002 — Como dev, quero que o agente criado seja funcional (compila e
  responde), **para** que não seja só referência.
- U-003 — Como usuário, quero que o agente respeite o multi-tenant, **para**
  não regredir o H-004.

## Critérios
- AC-001 — [U-001] `langchain` é adicionado como dependência.
- AC-002 — [U-001] `build_agent` usa `langchain.agents.create_agent` (não
  `create_react_agent` deprecado).
- AC-003 — [U-002] O agente criado compila e responde (invoke funcional).
- AC-004 — [U-003] O agente respeita o multi-tenant (chat_id).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (create_agent do langchain.agents substitui create_react_agent) — confirmada
- A-002 — (Testes com stub determinístico) — confirmada

## Perguntas em aberto
- (nenhuma)