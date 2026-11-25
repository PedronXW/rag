# F-052 — Migrar create_react_agent para create_agent

Migra o agente react para `langchain.agents.create_agent` (novo), substituindo
o `create_react_agent` deprecado do LangGraph 1.x. `build_agent` é o caminho
ativo; `build_react_agent` é mantido como referência legada.

## Critérios de aceite implementados
- **AC-001** — `langchain` adicionado como dependência.
- **AC-002** — `build_agent` usa `langchain.agents.create_agent`.
- **AC-003** — `build_agent` é chamável (create_agent), com `react_config`.
- **AC-004** — `react_config` é por chat (multi-tenant), thread `react:{chat_id}`.

## Arquivos
- `src/einstein_test/services/rag/react.py` — `build_agent` (create_agent),
  `build_react_agent` (legado), `react_config`.
- `pyproject.toml` — dependência `langchain`.

## Observação
`create_agent` requer um `BaseChatModel` real (com `bind_tools`); o adapter
custom do projeto (ChatLLM → str) não é um BaseChatModel, então `build_agent`
é funcional quando recebe um modelo do langchain.
