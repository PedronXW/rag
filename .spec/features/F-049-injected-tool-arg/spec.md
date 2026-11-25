# F-049 — InjectedToolArg nas tools do RAG

Refatora as tools do agente (G-002) para o padrão idiomático do LangGraph:
`InjectedToolArg` do langchain-core. Em vez de capturar `retriever`/`embedder`/
`chat_id` via closure, o runtime injeta esses valores na hora da chamada da
tool — tools mais limpas, testáveis e declarativas.

## Histórias
- U-001 — Como dev, quero que as tools do agente recebam `retriever`/
  `embedder`/`chat_id` via `InjectedToolArg`, **para** não depender de closure.
- U-002 — Como dev, quero que o runtime injete automaticamente os valores ao
  chamar a tool, **para** que a assinatura da tool seja declarativa.
- U-003 — Como usuário, quero que o agente continue funcionando (multi-tenant
  por chat preservado), **para** não quebrar o comportamento.

## Critérios
- AC-001 — [U-001] As tools (`search_chunks`, `fetch_document`) declaram
  `retriever`/`embedder`/`chat_id` com `InjectedToolArg`.
- AC-002 — [U-002] O runtime injeta os valores automaticamente ao invocar a
  tool (sem closure).
- AC-003 — [U-003] O agente continua funcionando e respeita o multi-tenant
  (só dados do chat).
- AC-004 — [U-003] O `create_react_agent`/supervisor que usa as tools continua
  funcional (regressão).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (InjectedToolArg do langchain-core para retriever/embedder/chat_id) — confirmada
- A-002 — (Padrão idiomático; sem quebrar o agente) — confirmada
- A-003 — (Testes com stubs determinísticos) — confirmada

## Perguntas em aberto
- (nenhuma)