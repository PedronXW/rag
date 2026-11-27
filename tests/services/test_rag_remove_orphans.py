"""K-001 — Testes da remoção de módulos órfãos do RAG."""

import importlib
import os


def test_modulos_orfaos_removidos():
    """AC-001/002/003: os módulos órfãos não importam em produção."""
    for mod in ("evaluation", "ragas_eval", "react", "rich_stream", "streaming"):
        with __import__("pytest").raises(ImportError):
            importlib.import_module(f"einstein_test.services.rag.{mod}")
            assert False, f"{mod} deveria ter sido removido"


def test_nao_hay_arquivo_orfaos():
    """AC-001/002/003: os arquivos dos módulos órfãos não existem."""
    base = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "einstein_test", "services", "rag"
    )
    for mod in ("evaluation", "ragas_eval", "react", "rich_stream", "streaming"):
        assert not os.path.exists(os.path.join(base, f"{mod}.py")), f"{mod}.py ainda existe"


def test_init_imports_publicos_reduzidos():
    """AC-004: o __init__.py do rag não exporta build_graph/build_graph_async."""
    from einstein_test.services.rag import __all__ as rag_all

    assert "build_graph" not in rag_all
    assert "build_graph_async" not in rag_all
    # API viva preservada
    assert "RagService" in rag_all
    assert "get_rag_service" in rag_all
    assert "allowed_models" in rag_all
    # feature Ollama preservada
    assert "ChatOllamaAdapter" in rag_all


def test_mcp_continua():
    """AC-005: o módulo mcp permanece e é importável."""
    import einstein_test.mcp.server as server

    assert server is not None


def test_eval_langfuse_continua():
    """AC-005: a avaliação via Langfuse permanece."""
    from einstein_test.services.rag.eval_langfuse import langfuse_eval_enabled

    assert callable(langfuse_eval_enabled)