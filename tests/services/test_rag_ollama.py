"""F-060 — Adaptação para modelos locais via Ollama.

Valida o `ChatOllamaAdapter`, a factory `build_chat_llm` (roteio por prefixo
`ollama/`) e a flexibilização da validação de settings para setups locais.
"""


def test_is_ollama_model():
    from einstein_test.services.rag.llm import is_ollama_model

    assert is_ollama_model("ollama/llama3.2") is True
    assert is_ollama_model("azure/gpt-5-mini") is False
    assert is_ollama_model(None) is False


def test_chat_ollama_adapter_nao_exige_chave_e_remove_prefixo():
    from einstein_test.services.rag.llm import ChatOllamaAdapter

    adapter = ChatOllamaAdapter(model="ollama/llama3.2")
    assert adapter.model == "llama3.2"
    assert adapter.api_key == "ollama"
    assert adapter.base_url == "http://localhost:11434/v1"
    assert hasattr(adapter, "stream") and hasattr(adapter, "astream")
    assert hasattr(adapter, "bind_tools")


def test_chat_ollama_adapter_model_default():
    from einstein_test.services.rag.llm import ChatOllamaAdapter

    adapter = ChatOllamaAdapter()
    assert adapter.model == "llama3.2"


def test_build_chat_llm_roteia_ollama():
    from einstein_test.services.rag.llm import (
        ChatOllamaAdapter,
        ChatOpenAIAdapter,
        build_chat_llm,
    )

    assert isinstance(build_chat_llm(model="ollama/llama3.2"), ChatOllamaAdapter)
    assert isinstance(
        build_chat_llm(model="azure/gpt-5-mini"), ChatOpenAIAdapter
    )


def test_validate_settings_aceita_somente_ollama_sem_chave():
    from einstein_test.settings import Settings, validate_settings

    s = Settings(
        database_url="postgresql://u:p@localhost/db",
        jwt_secret="s" * 32,
        openai_api_key="",
        allowed_models=["ollama/llama3.2"],
    )
    validate_settings(s)  # não deve levantar


def test_validate_settings_ainda_exige_chave_sem_ollama():
    from einstein_test.settings import Settings, SettingsError, validate_settings

    s = Settings(
        database_url="postgresql://u:p@localhost/db",
        jwt_secret="s" * 32,
        openai_api_key="",
        allowed_models=["azure/gpt-5-mini"],
    )
    try:
        validate_settings(s)
    except SettingsError as e:
        assert "OPENAI_API_KEY" in str(e)
    else:
        raise AssertionError("deveria exigir OPENAI_API_KEY")


def test_public_api_expoe_adapter_ollama():
    import einstein_test.services.rag as rag

    assert hasattr(rag, "ChatOllamaAdapter")
    assert hasattr(rag, "build_chat_llm")
    assert hasattr(rag, "is_ollama_model")
