# F-049 — InjectedToolArg nas tools do RAG

Refatora as tools do agente para o padrão idiomático do LangGraph:
`InjectedToolArg`. `retriever`/`embedder`/`chat_id` são injetados pelo runtime
na chamada (ocultos do schema do LLM), com fallback para os valores da closure.

## Critérios de aceite implementados
- **AC-001** — `search_chunks`/`fetch_document` declaram `InjectedToolArg`.
- **AC-002** — Os valores são injetados na invocação (com fallback à closure).
- **AC-003** — O agente respeita o multi-tenant (chat_id).
- **AC-004** — `create_react_agent`/supervisor continuam funcionando.

## Arquivos
- `src/einstein_test/services/rag/agent.py` — `build_tools` com `InjectedToolArg`.

## Observação
O `AgentState` ganhou `chat_id`/`retriever`/`embedder`; as tools usam
`Annotated[..., InjectedToolArg]` com default None → fallback aos valores do
factory (compatibilidade com callers diretos).
