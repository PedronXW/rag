# F-062 — Conformidade de config e observabilidade dos graphs

Garante que todos os graphs RAG usem o `graph_config` (callbacks Langfuse) e
remove a duplicação do `thread_config` no checkpointer.

## Critérios de aceite implementados
- **AC-001** — `agent_config`/`supervisor_config`/`multi_config`/`hitl_config`
  usam `graph_config` (com callbacks quando habilitado).
- **AC-002** — `checkpointer.thread_config` delega ao `graph_config` (removida
  duplicação).
- **AC-003** — Com Langfuse habilitado, os configs incluem `callbacks`;
  desabilitado, não incluem (determinístico).
- **AC-004** — Config de modelo de chat com fonte única: `ChatOpenAIAdapter`
  usa `settings.openai_chat_model` como default.
- **AC-005** — Regressão: 634 testes passando.

## Arquivos
- `services/rag/checkpointer.py` — `thread_config` delega ao `graph_config`.
- `services/rag/{agent,supervisor,multiquery,hitl}.py` — já usavam `graph_config`.

## Observação
O AC-001 já havia sido implementado na J-001 (F-056); esta feature remove o
`thread_config` duplicado e consolida a fonte única de modelo (settings).