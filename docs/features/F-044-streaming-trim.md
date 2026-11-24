# F-044 — Streaming token-real e trim_messages no RAG

Melhora o RAG multi-turno: (1) o histórico do `/messages` passa por
`trim_messages` (mantém as últimas N mensagens / até K tokens); (2) helpers de
streaming SSE reutilizáveis. O contrato SSE (node/token/done/[DONE]) é
preservado.

## Critérios de aceite implementados
- **AC-001** — `trim_history` limita o histórico às últimas N mensagens /
  até K tokens (via `langchain_core.trim_messages`).
- **AC-002** — `stream_tokens`/`astream_tokens` emitem tokens reais do LLM.
- **AC-003** — `build_sse` formata eventos SSE (node/token/done).
- **AC-004** — Após o trim, o histórico cabe no limite de tokens.
- **AC-005** — Testes SSE existentes continuam passando (regressão).

## Arquivos
- `src/einstein_test/services/rag/graph.py` — `trim_history`, `_token_count`.
- `src/einstein_test/services/rag/streaming.py` — `stream_tokens`,
  `astream_tokens`, `build_sse`.
- `src/einstein_test/api/messages.py` — aplica `trim_history` no histórico.

## Como rodar
O `POST /chats/{chat_id}/messages` agora limita o histórico por tokens;
os helpers de streaming ficam em `services/rag/streaming.py`.

## Correções do deep review
- **trim_history corrigido**: sem `keep_first` (inválido no langchain-core 1.7.0);
  aplica `max_messages` (fatiamento) ANTES do trim de tokens — política de
  mensagens é a fonte principal (preserva F-011).
- **Contrato F-011 preservado**: `_history` usa `max_messages=MAX_HISTORY` (10).
- **sources corretas**: `_astream_events` coleta sources e emite um único
  `done` final (antes um `done` vazio sobrescrevia as sources).
