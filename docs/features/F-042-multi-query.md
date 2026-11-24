# F-042 — Multi-query com subgraphs no RAG

Multi-query RAG com subgraphs e map-reduce: o graph reescreve a pergunta em N
variações (LLM), dispara um retrieve paralelo por variação via `Send`, e funde
os resultados (dedup por chunk_id + rank por score máximo) antes de gerar.
Subgraph de retrieval reutilizável. Rota `POST /chats/{chat_id}/multi-query`.

## Critérios de aceite implementados
- **AC-001** — `build_retrieval_subgraph` (subgraph de retrieval compilado).
- **AC-002** — rewrite gera N variações via LLM quando `queries > 1`.
- **AC-003** — `Send` dispara um retrieve por variação (map paralelo).
- **AC-004** — O canal `chunks` acumula via reducer `operator.add`.
- **AC-005** — `fuse_chunks` deduplica por chunk_id e ranqueia por score máximo.
- **AC-006** — `POST /chats/{chat_id}/multi-query` responde `{answer, sources}` +
  [DONE].

## Arquivos
- `src/einstein_test/services/rag/multiquery.py` — subgraph de retrieve,
  rewrite, Send, fuse_chunks, `build_multiquery_graph`.
- `src/einstein_test/api/messages.py` — rota `/multi-query`.

## Como rodar
`POST /chats/{id}/multi-query?queries=3` (SSE) → `{answer, sources}` + [DONE].

## Correções do deep review
- **Subgraph real como nó**: `run_retrieval` usa `build_retrieval_subgraph`
  compilado (AC-001 de fato).
- **Thread-safety**: retrievals paralelos (Send) rodam com `max_concurrency=1`
  (evita acesso concorrente à Session SQLAlchemy em Postgres).
- **Score real**: `_chunk_to_dict` calcula cosine entre query e chunk embedding
  (FileChunk não tem score próprio) — o ranking por score máximo funciona.
- **Prompt injection no rewrite**: marca a pergunta do usuário como dado não
  confiável.
- **Namespace de thread**: `multi:{chat_id}` não colide com a memória RAG.
