# F-060 — Modelos locais via Ollama

Permite usar LLMs locais (Ollama) como modelo de chat do RAG, por usuário,
sem depender de chave/endpoint externos. Modelos locais são identificados pelo
prefixo `ollama/` em `ALLOWED_MODELS`/`User.model` (ex.: `ollama/llama3.2`).

## Como funciona

- `ChatOllamaAdapter` (`services/rag/llm.py`) — subclasse do
  `ChatOpenAIAdapter` que fala com o **endpoint OpenAI-compatível** do Ollama
  (`OLLAMA_BASE_URL`, default `http://localhost:11434/v1`) usando `ollama` como
  api_key fictícia (o Ollama não exige chave). Reusa o mesmo streaming, retry
  transiente no 1º token e tool-calling do adapter OpenAI.
- `build_chat_llm(model)` — factory que roteia: `ollama/<nome>` →
  `ChatOllamaAdapter`; qualquer outro → `ChatOpenAIAdapter`. Usada em
  `get_rag_service` (RAG por usuário), no LLM-judge do eval e nas rotas RAG
  (todas propagam o modelo do usuário).
- `OllamaEmbedder` (`vector/embedder.py`) — embeddings locais via `/v1/embeddings`
  do Ollama (`OLLAMA_EMBED_MODEL`, default `nomic-embed-text`, 768 dims). Como o
  schema fixa `vector(1536)`, os vetores são **zero-padded** para 1536 — a
  similaridade por cosseno (e o índice HNSW) permanece correta, sem migration.
  Selecionado via `EMBEDDER=ollama`.
- `validate_settings` — `OPENAI_API_KEY` deixa de ser obrigatória quando
  `ALLOWED_MODELS` contém ao menos um modelo `ollama/` (setup 100% local).
- docker-compose — serviço `ollama` (vol. `ollama-data`) e `OLLAMA_BASE_URL`/
  `EMBEDDER`/`OLLAMA_EMBED_MODEL` injetados na API/worker.

## Critérios de aceite
- **AC-001** — `ChatOllamaAdapter` funciona sem `OPENAI_API_KEY`, remove o
  prefixo `ollama/` do nome do modelo e aponta para `OLLAMA_BASE_URL`.
- **AC-002** — `build_chat_llm` roteia `ollama/*` → `ChatOllamaAdapter` e os
  demais → `ChatOpenAIAdapter`.
- **AC-003** — `get_rag_service`, as rotas RAG e o LLM-judge criam o adapter
  correto conforme o modelo do usuário.
- **AC-004** — `validate_settings` aceita setup somente-ollama sem chave; segue
  exigindo chave sem modelos locais.
- **AC-005** — `OllamaEmbedder` zero-pad para `vector(1536)` sem exigir chave;
  `get_embedder` roteia por `EMBEDDER`.
- **AC-006** — Regressão: testes existentes continuam passando.

## Como usar
```bash
# 1. suba o Ollama e puxe os modelos (chat + embedding)
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text

# 2. libere o modelo na allowlist e o embedder local (.env)
EMBEDDER=ollama
ALLOWED_MODELS=azure/gpt-5-mini,azure/gpt-6-luna,ollama/llama3.2

# 3. usuário escolhe o modelo local
curl -X PATCH /users/me -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"model": "ollama/llama3.2"}'
```
