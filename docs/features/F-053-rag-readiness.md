# F-053 — Rate limit e readiness nas rotas RAG

Aplica às rotas RAG (`/agent`, `/multi-query`, `/supervisor`) o mesmo gate de
readiness (`rag_readiness`) que o `/messages` tem, e adiciona rate limiting
específico por rota — evitando respostas com contexto vazio (alucinação) e uso
abusivo.

## Critérios de aceite implementados
- **AC-001** — As rotas `/agent`, `/multi-query`, `/supervisor` emitem notice
  `no_ready_files` quando não há arquivos prontos (sem chamar o LLM).
- **AC-002** — Emitem notice `partial_files` quando há arquivos parciais.
- **AC-003** — Com arquivos prontos, a rota roda normalmente.
- **AC-004** — Rate limiting específico por rota RAG (429 quando excede).
- **AC-005** — O `/messages` continua funcionando (regressão).

## Arquivos
- `src/einstein_test/api/messages.py` — `_rag_readiness_notice` + gate nas rotas.
- `src/einstein_test/middleware/rate_limit.py` — limites e mapeamento de rota
  para `/agent`, `/multi-query`, `/supervisor`, `/hitl`, `/resume`.

## Como rodar
Rotas RAG sem arquivos prontos agora retornam `notice: no_ready_files` +
`[DONE]` (sem gastar LLM); cada rota tem limite de requisições/min próprio.
