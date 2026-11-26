"""Módulo de vetorização: embedder, extração, chunking e repositório pgvector."""

import os

from dotenv import load_dotenv

from einstein_test.vector.chunker import TextChunker
from einstein_test.vector.embedder import (
    Embedder,
    FakeEmbedder,
    OpenAIEmbedder,
    OllamaEmbedder,
)
from einstein_test.vector.extractor import TEXT_EXTENSIONS, TextExtractor
from einstein_test.vector.repository import VectorRepository, VectorRepositorySQLAlchemy

_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Retorna o embedder configurado (OpenAI por padrão; fake nos testes).

    Com `EMBEDDER=ollama`, usa o `OllamaEmbedder` (modelos locais, sem chave) —
    necessário para setups 100% locais (ex.: Ollama + chat local).
    """
    global _embedder
    if _embedder is None:
        _embedder = _build_embedder()
    return _embedder


def _build_embedder() -> Embedder:
    load_dotenv()
    if os.getenv("EMBEDDER", "openai").strip().lower() == "ollama":
        return OllamaEmbedder()
    return OpenAIEmbedder()


def set_embedder(embedder: Embedder) -> None:
    global _embedder
    _embedder = embedder


__all__ = [
    "Embedder",
    "FakeEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "TEXT_EXTENSIONS",
    "TextChunker",
    "TextExtractor",
    "VectorRepository",
    "VectorRepositorySQLAlchemy",
    "get_embedder",
    "set_embedder",
]