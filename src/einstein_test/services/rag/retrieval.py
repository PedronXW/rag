"""Helpers centralizados de retrieval e transientes (I-refactor).

Fonte única para:
- `retrieve_chunks`: busca vetorial com `CHUNKS_PER_FILE`/`MAX_CHUNKS`.
- `chunk_to_dict`: conversão de chunk (objeto/dict) em dict serializável.
- `_is_transient`: classificação de exceções transientes (rede/storage/timeout).

Elimina o drift de constantes hardcoded e a duplicação de `_chunk_to_dict`/
`_is_transient` entre os módulos do RAG.
"""

from typing import Any

CHUNKS_PER_FILE = 2
MAX_CHUNKS = 12


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def chunk_to_dict(c: Any, query_embedding: list[float] | None = None) -> dict:
    """Converte um chunk (objeto ou dict) em dict serializável.

    Se `query_embedding` for fornecido e o chunk tiver `.embedding`, calcula o
    score real via cosine; senão usa o score do chunk (default 0.0).
    """
    if isinstance(c, dict):
        base = {
            "chunk_id": c.get("id") or c.get("chunk_id"),
            "document_id": c.get("file_id") or c.get("document_id"),
            "content": c.get("content", ""),
        }
        if "score" in c:
            base["score"] = c["score"]
        return base

    score = getattr(c, "score", 0.0)
    if query_embedding is not None and getattr(c, "embedding", None) is not None:
        from einstein_test.vector.similarity import cosine_similarity

        score = round(cosine_similarity(query_embedding, list(c.embedding)), 6)
    return {
        "chunk_id": getattr(c, "id", None),
        "document_id": getattr(c, "file_id", None),
        "content": getattr(c, "content", str(c)),
        "score": score,
    }


def retrieve_chunks(question: str, retriever, embedder, chat_id: int) -> list[dict]:
    """Busca os chunks mais similares do chat e os converte em dicts serializáveis."""
    embedding = embedder.embed([question])[0]
    chunks = (
        retriever.search_similar(
            embedding=embedding,
            chat_id=chat_id,
            chunks_per_file=CHUNKS_PER_FILE,
            max_chunks=MAX_CHUNKS,
        )
        or []
    )
    return [chunk_to_dict(c, embedding) for c in chunks]


__all__ = ["retrieve_chunks", "chunk_to_dict", "_is_transient", "CHUNKS_PER_FILE", "MAX_CHUNKS"]