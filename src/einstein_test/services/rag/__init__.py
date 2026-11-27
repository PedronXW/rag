"""Serviço RAG — pacote com submódulos coesos (F-023).

- `llm.py`: Protocol/fakes/adapter + retry + `allowed_models()`.
- `graph.py`: grafo LangGraph (sync/async) sem duplicação.
- `service.py`: `RagService` + factory.

Este módulo re-exporta a API pública usada por callers e testes
(`from einstein_test.services.rag import ...`).
"""

from einstein_test.services.rag.llm import (
    LLM_MAX_RETRIES,
    LLM_RETRY_BACKOFF,
    ChatLLM,
    ChatOpenAIAdapter,
    ChatOllamaAdapter,
    FakeChatLLM,
    allowed_models,
    build_chat_llm,
    is_ollama_model,
)
from einstein_test.services.rag.service import (
    RagService,
    get_rag_service,
    reset_rag_factory,
    set_rag_factory,
)

__all__ = [
    "ChatLLM",
    "FakeChatLLM",
    "ChatOpenAIAdapter",
    "ChatOllamaAdapter",
    "build_chat_llm",
    "is_ollama_model",
    "RagService",
    "get_rag_service",
    "set_rag_factory",
    "reset_rag_factory",
    "allowed_models",
    "LLM_MAX_RETRIES",
    "LLM_RETRY_BACKOFF",
]